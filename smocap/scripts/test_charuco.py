#!/usr/bin/env python

import sys, numpy as np, rospy, rospkg, time, tf2_ros, geometry_msgs.msg
import cv2, cv2.aruco
import pdb

import smocap.camera
import smocap.camera_system
import smocap.rospy_utils

class CalibrationNode:
    
    def __init__(self, cam_sys, board, dictionary):
        self.cam_sys = cam_sys
        self.board, self.dictionary = board, dictionary
        self.cam_listener = smocap.rospy_utils.CamerasListener(cams=cam_sys.cam_names, cbk=self.on_image)
        self.has_image = False
        
    def on_image(self, img, info):
        if not self.has_image:
            self.img = img
            self.has_image = True
        
    def periodic(self):
        pass

    def run(self):
        #rate = rospy.Rate(1)
        #while not rospy.is_shutdown():
        #    self.periodic()
        #    rate.sleep()
        while not self.has_image:
            time.sleep(0.1)
        cam = self.cam_sys.get_camera(0)
        img_res, world_to_cam_t, world_to_cam_q = localize_aruco_board(self.img, cam.K, cam.D, self.dictionary, self.board)
        cv2.imshow('board', img_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cam.set_location(world_to_cam_t, world_to_cam_q)

        
class CamSysTFBroadcaster:

    def __init__(self, cam_sys):
        rospy.loginfo(' CamSysTFBroadcaster {}'.format('0'))
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        cam = cam_sys.get_camera(0)
        static_transformStamped.child_frame_id = '{}_optical_frame'.format(cam.name)
        
        smocap.rospy_utils.transf_msg_of_tq( static_transformStamped.transform, cam.world_to_cam_t, cam.world_to_cam_q)
        broadcaster.sendTransform(static_transformStamped)
        
    def run(self):
        rospy.spin()

def make_board(dictionary, nrow, ncol, display=False):
    board = cv2.aruco.CharucoBoard_create(ncol, nrow, .025, .0125, dictionary)
    if display:
        img = board.draw((200*ncol,200*nrow))
        cv2.imwrite('charuco.png',img)
        cv2.imshow('board', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return board


def localize_aruco_board(img_gray, camera_matrix, dist_coeffs, dictionary, board):
    if len(img_gray.shape) == 3:
        img_res = np.array(img_gray)
    else:
        img_res = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    # Detect markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_gray, dictionary)
    print('detected {} markers {}'.format(len(ids.ravel()),  ids.ravel()))
    #cv2.aruco.drawDetectedMarkers(img_res, corners, ids)

    # Refine markers detection
    detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = cv2.aruco.refineDetectedMarkers(img_gray, board, corners, ids, rejectedImgPoints)
    print('refined {} markers {}'.format(len(detectedIds.ravel()), detectedIds.ravel()))
    #cv2.aruco.drawDetectedMarkers(img_res, detectedCorners, detectedIds)

    # Find chessboard
    charucoretval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(detectedCorners, detectedIds, img_gray, board,
                                                                                    cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
    #charucoretval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, img, board,
    #                                                                                cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
    print('chessboard corners ids: {}'.format(charucoIds.ravel()))
    img_res = cv2.aruco.drawDetectedCornersCharuco(img_res, charucoCorners, charucoIds, (0,255,0))    

    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, dist_coeffs)  # pose estimation from a charuco board
    #print retval, rvec, tvec
    cv2.aruco.drawAxis(img_res, camera_matrix, dist_coeffs, rvec, tvec, 100)

    world_to_cam_T = smocap.utils.T_of_t_r(tvec.squeeze(), rvec)
    world_to_cam_t, world_to_cam_q = smocap.utils.tq_of_T(world_to_cam_T)
    print(' world_to_cam_t {} world_to_cam_q {}'.format(world_to_cam_t, world_to_cam_q))

    return img_res, world_to_cam_t, world_to_cam_q


def main(args):
    swd = rospkg.RosPack().get_path('smocap')
    camera_name = 'ueye_enac_112'
    camera_intrinsic_path = swd+'/params/ricou/{}.yaml'.format(camera_name)
    nrow, ncol, dictionary = 7, 9, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = make_board(dictionary, nrow, ncol)

    cam_sys = smocap.camera_system.CameraSystem(cam_names=[camera_name])
    cam = cam_sys.get_camera(0)
    cam.load_intrinsics(camera_intrinsic_path)
    
    if 0:
        img_path = swd+'/test/charuco/scene_05.png'
        img_gray = cv2.imread(img_path)
        camera_matrix, dist_coeffs, w, h = smocap.camera.load_intrinsics(camera_intrinsic_path)
    
        img_res, world_to_cam_t, world_to_cam_q = localize_aruco_board(img_gray, cam.K, cam.D, dictionary, board)
       
        cv2.imshow('scene', img_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        try:
            rospy.init_node('test_charuco', anonymous=True)
            CalibrationNode(cam_sys, board, dictionary).run()
            CamSysTFBroadcaster(cam_sys).run()
        except rospy.ROSInterruptException: pass

    
if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=300)
    main(sys.argv)
