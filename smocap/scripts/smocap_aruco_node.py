#!/usr/bin/env python

import sys, numpy as np, rospy, cv2

import smocap
import smocap.rospy_utils



class Marker:
    def __init__(self, _id):
        self.id = _id

    def set_pose(self, T_m2w):
        self.T_m2w = T_m2w


class ArucoMocap:
    def __init__(self, cam_sys):
        self.cam_sys = cam_sys
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params =  cv2.aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.marker_size = 1.
        self.markers = {}
        
    def process_image(self, img, cam_idx):
        self.corners, self.ids, self.rejectedImgPoints = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)
        
        cam = self.cam_sys.get_cameras()[cam_idx]
        T_c2w = cam.cam_to_world_T
        self.r_m2cs, self.t_m2cs, self.objpoints = cv2.aruco.estimatePoseSingleMarkers(self.corners, self.marker_size, cam.K, cam.D)
        try:
            for mid, r_m2c, t_m2c in zip(self.ids, self.r_m2cs, self.t_m2cs):
                if mid[0] not in self.markers:
                    self.markers[mid[0]] = Marker(mid[0])
                T_m2c = smocap.utils.T_of_t_r(t_m2c, r_m2c)
                T_m2w = np.dot(T_c2w, T_m2c)
                self.markers[mid[0]].set_pose(T_m2w)
        except TypeError: # no marker detected
            #print(self.corners, self.ids, self.rejectedImgPoints)
            print('no marker detected')
            
    def draw(self, img, cam):
        cv2.aruco.drawDetectedMarkers(img, self.corners, self.ids)
        if self.ids is not None:
            for r_m2c, t_m2c in zip(self.r_m2cs, self.t_m2cs):
                cv2.aruco.drawAxis(img, cam.K, cam.D, r_m2c, t_m2c, self.marker_size)
        return img
        

class SMoCapNode:

    def __init__(self, cam_names= ['camera_1']):
        self.low_freq = 2.
        self.img_pub = smocap.rospy_utils.ImgPublisher()
        self.pose_pub = smocap.rospy_utils.PosePublisher()
        self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(cam_names)
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)
        self.mocap = ArucoMocap(self.cam_sys)
        
    def periodic(self, cam_idx=0):
        img = self.cam_lst.get_image(cam_idx)
        if  img is not None:
            img = self.mocap.draw(img, self.cam_sys.get_camera(cam_idx))
            self.img_pub.publish(img)
                
    def run(self):
        rate = rospy.Rate(1./self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass    
    
    def on_image(self, img, cam_idx):
        self.image = img # save image for low rate display callback
        self.mocap.process_image(img, cam_idx)
        self.pose_pub.publish(self.mocap.markers[0].T_m2w)


        
def main(args):
  rospy.init_node('smocap_node')
  rospy.loginfo('smocap node starting')
  SMoCapNode().run()

if __name__ == '__main__':
    main(sys.argv)
