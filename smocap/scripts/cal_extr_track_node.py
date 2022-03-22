#!/usr/bin/env python3

# ./scripts/cal_extr_track_node.py  _camera:=ueye_enac_z_2
#
# Calibrate camera extrinsics using a known track
#

import os, sys, logging
import math, numpy as np
import rospy, tf2_ros,  geometry_msgs.msg, rospkg

import cv2, cv2.aruco as aruco
LOG = logging.getLogger('calibrate_extrisinc')

import common_vision.utils as cv_u
import common_vision.rospy_utils as cv_rpu
import common_vision.camera as cv_c
import common_vision.bird_eye as cv_be
import two_d_guidance as tdg

MY_DICT = aruco.DICT_5X5_1000
MY_BOARD_PARAMS_A4 = {'squaresX':7,
                      'squaresY':5,
                      'squareLength':0.04,
                      'markerLength':0.03} # 5px/mm

class ExtrCalibPipeline(cv_u.Pipeline):
    show_none=0
    def __init__(self, cam, robot_name):
        cv_u.Pipeline.__init__(self)

        self.ar_dict = aruco.Dictionary_get(MY_DICT)
        self.gridboard = aruco.CharucoBoard_create(**MY_BOARD_PARAMS_A4, dictionary=self.ar_dict)

        self.bird_eye = None
        
        self.display_mode = 3
        self.img = None
        self.corners = None
        self.rvec, self.tvec = None, None
        self.ref_2_cam_T = None
        tdg_wd = rospkg.RosPack().get_path('two_d_guidance')
        self.track_path = tdg.Path(load=os.path.join(tdg_wd, 'paths/roboteck/track.npz'))
    
    def _detect_track(self, img, cam, thresh=180):
        blurred_img = cv2.GaussianBlur(img, (9, 9), 0)
        # find white area
        ret, self.threshold1 = cv2.threshold(blurred_img, thresh, 255, cv2.THRESH_BINARY)
        cnts, hierarchy = cv2.findContours(self.threshold1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        self.cnt_max = max(cnts, key=cv2.contourArea)
        self.tl, self.br = np.min(self.cnt_max, axis=0).squeeze(), np.max(self.cnt_max, axis=0).squeeze()
        
    def _detect_and_localize_board(self, img, cam):
        # detect markers
        self.corners, self.ids, rejectedImgPoints = cv2.aruco.detectMarkers(self.img, self.ar_dict)
        # refine markers detection using board description
        res = aruco.refineDetectedMarkers(self.img, self.gridboard, self.corners, self.ids, rejectedImgPoints, cam.K, cam.D)
        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = res
        
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=detectedCorners,
            markerIds=detectedIds,
            image=self.img,
            board=self.gridboard)
            
        ret, self.rvec, self.tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.gridboard,
                                                                       cam.K, cam.D, None, None, False)
        self.board_2_cam_T = cv_u.T_of_t_r(self.tvec.squeeze(), self.rvec)
        self.ref_2_board_t = [0., 0., -0.005]         # ref (track frame) to ar board transform
        self.ref_2_board_R = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
        self.ref_2_board_T = cv_u.T_of_t_R(self.ref_2_board_t, self.ref_2_board_R)
        self.ref_2_cam_T = np.dot(self.board_2_cam_T, self.ref_2_board_T)
        cam.set_pose_T(self.ref_2_cam_T)
        
    def _process_image(self, img, cam, stamp, stop_when_localized=True):
        self.img = img
        if self.ref_2_cam_T is None or not stop_when_localized:
            #self._detect_track(img, cam) # TODO
            self._detect_and_localize_board(img, cam)

        

    def _draw_path(self, cam, img):
        pts_world = np.array([[x, y, 0] for x, y in self.track_path.points])
        pts_img = cam.project(pts_world).squeeze()
        for _p in pts_img:
            cv2.circle(img, tuple(_p.astype(int)), 4, (0,0,255), -1)
        
        
    def draw_debug_bgr(self, cam, img_cam=None):
        cnt_color, cntmax_color, thickness = (255,0,0), (0, 255,0), 3
        if self.img is None:
            return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        else:
            if self.display_mode == 1: # input
                debug_img = self.img.copy()
            elif self.display_mode == 2: # 
                debug_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
                #debug_img = cv2.cvtColor(self.threshold1, cv2.COLOR_GRAY2BGR)
                #cv2.drawContours(debug_img, self.cnt_max, -1, cntmax_color, 3)
                #cv2.rectangle(debug_img, tuple(self.tl), tuple(self.br), (0,0,255), 1)

                if self.corners is not None:
                    aruco.drawDetectedMarkers(image=debug_img, corners=self.corners)
                if self.rvec is not None:
                    cv2.drawFrameAxes(debug_img, cam.K, cam.D, self.rvec, self.tvec, 0.1)
                self._draw_path(cam, debug_img)
                 
            elif self.display_mode == 3: # 
                if self.bird_eye is None :
                    debug_img = self.img.copy()
                else:
                    chroma_blue, chroma_green = (187, 71, 0), (64, 177, 0)
                    debug_img = self.bird_eye.undist_unwarp_img(cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR), cam, fill_bg=chroma_green)
                    ui = cv_be.UnwarpedImage()
                    ui.draw_grid(debug_img, self.bird_eye, cam, 0.1)
                    ui.draw_path(debug_img, self.bird_eye, cam, self.track_path.points)

        self.draw_timing(debug_img)
        return debug_img

    def has_extrinsics(self):
        return self.ref_2_cam_T is not None
    
class BeParamTrack:
    x0, y0, dx, dy = -0.1, 0.3, 1.2, 0.8 # bird eye area in local floor plane frame
    max_dist = 3.
    w = 640                           # bird eye image width (pixel coordinates)
    s = dy/w                          # scale
    h = int(dx/s)                     # bird eye image height
    
class Node(cv_rpu.SimpleVisionPipeNode):

    def __init__(self, compress_debug=True):
       cv_rpu.SimpleVisionPipeNode.__init__(self, ExtrCalibPipeline, self.pipe_cbk, img_fmt="passthrough", fetch_extrinsics=False)
       self.img_pub = cv_rpu.CompressedImgPublisher(self.cam, '/smocap/calib_extr/image')
       self.broadcaster = tf2_ros.StaticTransformBroadcaster()
       self.calibrated = False
       self.start()

    def pipe_cbk(self):
        #self.img_pub.publish(self, self.cam, "bgr8")
        pass
    
    def periodic(self):
        #print('proc: {:.1f}ms'.format(self.pipeline.lp_proc*1e3))
        if self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)
        if self.pipeline.has_extrinsics():
            if not self.calibrated:
                self.calibrated = True
                self.bird_eye = cv_be.BirdEye(self.cam, BeParamTrack(), cache_filename='/tmp/be_cfg.npz', force_recompute=False)
                self.pipeline.bird_eye = self.bird_eye # bleeeee....
            self.send_transform()
    
    def send_transform(self):
        t_ref2cam, q_ref2cam = cv_u.tq_of_T(self.pipeline.ref_2_cam_T)
        static_transformStamped = geometry_msgs.msg.TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = "ueye_enac_z_2_optical_frame"
        
        static_transformStamped.transform.translation.x = t_ref2cam[0]
        static_transformStamped.transform.translation.y = t_ref2cam[1]
        static_transformStamped.transform.translation.z = t_ref2cam[2]
        
        static_transformStamped.transform.rotation.x = q_ref2cam[0]
        static_transformStamped.transform.rotation.y = q_ref2cam[1]
        static_transformStamped.transform.rotation.z = q_ref2cam[2]
        static_transformStamped.transform.rotation.w = q_ref2cam[3]
        self.broadcaster.sendTransform(static_transformStamped)

            
    
def main(args):
    name = 'smocap_cal_extr_node'
    rospy.init_node(name)
    rospy.loginfo('{} starting'.format(name))
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run(low_freq=1)


if __name__ == '__main__':
    main(sys.argv)
