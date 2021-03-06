#!/usr/bin/env python

'''
Looks like this is not finished...
'''


import sys, numpy as np, rospy, cv2

import smocap
import smocap.rospy_utils, smocap.real_time_utils

import pdb

class Marker:
    def __init__(self, _id, corner):
        self.id = _id
        self.set_pose(np.eye(4))

    def set_pose(self, T_m2w):
        self.T_m2w = T_m2w


# class Observation:
#     def __init__(self, cam, marker):
#         self.cam = cam
#         self.marker = marker
        
#     def set_m2c(self, t_m2c, r_m2c):
#         self.t_m2c, self.r_m2c = t_m2c, r_m2c
#         self.T_m2c = smocap.utils.T_of_t_r(t_m2c, r_m2c)

class Observations:
    def __init__(self, corners, mids, t_m2cs, r_m2cs, objpoints):
        self.corners, self.mids, self.t_m2cs, self.r_m2cs, self.objpoints = corners, mids, t_m2cs, r_m2cs, objpoints

        
class ArucoMocap:
    def __init__(self, cam_sys, marker_size):
        self.cam_sys = cam_sys
        self.cameras = cam_sys.get_cameras() # remove me
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params =  cv2.aruco.DetectorParameters_create()
        #self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.marker_size = marker_size
        self.markers = {}
        #self.obs_by_markers = {}
        n_cam = len(cam_sys.get_cameras())
        self.obs_by_cams = [None for i in range(n_cam)]


    def detect_markers_in_full_frame(self, img, cam_idx, stamp):
        '''
        Called by frame searcher thread
        '''
        corners, mids, rejectedImgPoints = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)
        #print('ff detected markers: {}'.format(mids))
        cam = self.cam_sys.get_camera(cam_idx)
        try:
            for mid, corner in zip(mids, corners):
                if mid[0] not in self.markers:
                    self.markers[mid[0]] = Marker(mid[0], corner)
            #pdb.set_trace()
            #r_m2cs, t_m2cs, objpoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, cam.K, cam.D)
            r_m2cs, t_m2cs = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, cam.K, cam.D)
            objpoints = None
            self.obs_by_cams[cam_idx] = Observations(corners, mids, t_m2cs, r_m2cs, objpoints)
        except TypeError: # no marker detected
            #print(corners, mids, rejectedImgPoints)
            self.obs_by_cams[cam_idx] = None
            #print('no marker detected in cam {}'.format(cam_idx))
            return

    
    def process_image(self, img, cam_idx):
        cam = self.cam_sys.get_camera(cam_idx)
        T_c2w = cam.cam_to_world_T
        o =  self.obs_by_cams[cam._id]
        if o is not None:
            for mid, r_m2c, t_m2c in zip(o.mids, o.r_m2cs, o.t_m2cs):
                T_m2c = smocap.utils.T_of_t_r(t_m2c, r_m2c)
                T_m2w = np.dot(T_c2w, T_m2c)
                #pdb.set_trace()
                self.markers[mid[0]].set_pose(T_m2w)
        else:
            rospy.loginfo('no markers observation in cam {}'.format(cam_idx))
            
    def draw(self, img, cam):
        o =  self.obs_by_cams[cam._id]
        if o is not None:
            cv2.aruco.drawDetectedMarkers(img, o.corners, o.mids)
            for r_m2c, t_m2c in zip(o.r_m2cs, o.t_m2cs):
                cv2.aruco.drawAxis(img, cam.K, cam.D, r_m2c, t_m2c, self.marker_size)
        return img
        

class SMoCapNode:

    def __init__(self):
        cam_names = rospy.get_param('~cameras', 'camera_1').split(',')
        marker_size = rospy.get_param('~marker_size', 1.)
        rospy.loginfo('using marker size {} m'.format(marker_size))
        self.low_freq = 2.
        self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(cam_names)
        self.profiler = smocap.real_time_utils.Profiler(self.cam_sys.nb_cams())

        self.img_pub = smocap.rospy_utils.ImgPublisher(self.cam_sys)
        self.pose_pub = smocap.rospy_utils.PosePublisher()
        self.fov_pub = smocap.rospy_utils.FOVPublisher(self.cam_sys)
        self.stat_pub = smocap.rospy_utils.StatusPublisher()

        self.mocap = ArucoMocap(self.cam_sys, marker_size)
        self.frame_searchers = [smocap.real_time_utils.FrameSearcher(i, self.mocap, self.profiler) for i, c in enumerate(self.cam_sys.get_cameras())]
        
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)
        
        
    def periodic(self):
        self.img_pub.publish(self.cam_lst.get_images_as_rgb(), self.profiler, self.mocap)
        self.fov_pub.publish()
        self.stat_pub.publish(self.profiler)
        
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            print('ROSInterruptException')
        print 'goodbye'
        for f in self.frame_searchers:
            f.stop()
            
    def on_image(self, img, (cam_idx, stamp, seq)):
        self.profiler.signal_start(cam_idx, stamp, seq)
        #img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.frame_searchers[cam_idx].put_image(img, seq, stamp)
        self.mocap.process_image(img, cam_idx)
        self.profiler.signal_done(cam_idx, rospy.Time.now())
        try: # publish the first tracked marker as /smocap/est
            mid = self.mocap.markers.keys()[0]
            self.pose_pub.publish(self.mocap.markers[mid].T_m2w)
        except IndexError:
            pass # no marker tracked


        
def main(args):
  rospy.init_node('smocap_node')
  rospy.loginfo('smocap node starting')
  rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
  SMoCapNode().run()

if __name__ == '__main__':
    main(sys.argv)
