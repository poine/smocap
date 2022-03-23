#!/usr/bin/env python3

import math, os, sys , time, numpy as np, rospy, sensor_msgs.msg, cv2, cv_bridge

def main(args):
  rospy.init_node('test_blob_detector_node')
  rospy.loginfo('test_blob_detector_node starting')
  cam_name = 'ueye_enac_z_2'
  cam_img_topic = f'/{cam_name}/image_raw'
  cam_img_msg = rospy.wait_for_message(cam_img_topic, sensor_msgs.msg.Image)
  rospy.loginfo(f'received image from {cam_name}')
  bridge = cv_bridge.CvBridge()
  img = bridge.imgmsg_to_cv2(cam_img_msg, 'passthrough')
  filename = '/tmp/smocap_fail.png'
  rospy.loginfo(f'saving to {filename}')
  cv2.imwrite(filename, img)
  
if __name__ == '__main__':
    main(sys.argv)

