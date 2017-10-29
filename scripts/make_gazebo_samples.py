#!/usr/bin/env python
import logging, sys, os, rospy, sensor_msgs.msg, math, numpy as np, threading, time, cv_bridge, cv2

import hog_remote, utils

LOG = logging.getLogger('make_gazebo_samples')

'''
   Programmatically send marker in Gazebo to defined locations and save resulting pictures
'''


def main(args):
    output_dir = '/home/poine/work/smocap.git/test/gazebo_samples_no_distortion'
    world_poses = [[[0., 0, 0], [0., 0, 0]],
                   [[1., 0, 0], [0., 0, 0]],
                   [[1., 0, 0], [0, 0, math.pi/2]],
                   [[1., 1, 0], [0, 0, 0]],
                   [[0., 1, 0], [0, 0, 0]],
                   [[0., 1, 0.5], [0, 0, 0]]]

    camera_name = '/smocap/camera'
    msg = rospy.wait_for_message(camera_name+'/camera_info', sensor_msgs.msg.CameraInfo)
    cam_info_filename = os.path.join(output_dir, 'camera_info.yaml')
    utils.write_camera_model(cam_info_filename, msg)
    LOG.info(' wrote camera calibration to {}'.format(cam_info_filename))

    remote = hog_remote.Node()
    remote_thread = threading.Thread(target=remote.run)
    remote_thread.start()
    bridge = cv_bridge.CvBridge()
    
    for i, (w_to_m_t, w_to_m_eu) in enumerate(world_poses):
        print i
        remote.set_position(w_to_m_t)
        remote.set_orientation(w_to_m_eu)
        time.sleep(0.1)
        while not remote.marker_has_arrived():
            time.sleep(0.1)
        msg = rospy.wait_for_message(camera_name+'/image_raw', sensor_msgs.msg.Image)
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            print(e)
        cv2.imwrite(os.path.join(output_dir, 'image_{:02d}.png'.format(i)), cv_image)   
        #print msg
        
    rospy.signal_shutdown('done')

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)
     rospy.init_node('smocap_make_gazebo_samples')
     main(sys.argv)
