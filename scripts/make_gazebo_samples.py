#!/usr/bin/env python
import logging, sys, os, rospy, sensor_msgs.msg, math, numpy as np, threading, time, cv_bridge, cv2, tf2_ros, yaml

import hog_remote, utils, smocap
import pdb

LOG = logging.getLogger('make_gazebo_samples')

'''
   Programmatically send marker in Gazebo to defined locations and save resulting pictures
'''


def make_samples(**args):
    output_dir = args.get('output_dir', '/home/poine/work/smocap.git/test/gazebo_samples_no_distortion')
    world_poses =  args.get('poses', [[[0., 0, 0], [0., 0, 0]],
                                      [[1., 0, 0], [0., 0, 0]],
                                      [[1., 0, 0], [0, 0, math.pi/2]],
                                      [[1., 1, 0], [0, 0, 0]],
                                      [[0., 1, 0], [0, 0, 0]],
                                      [[0., 1, 0.5], [0, 0, 0]]])

    camera_name, img_encoding = args.get('camera_name', 'smocap/camera'), 'mono8'
    cam = smocap.Camera(camera_name, img_encoding)

    # Retrieve camera intrinsic
    LOG.info(' retrieving camera calibration for {}'.format(camera_name))
    msg = rospy.wait_for_message('/{}/camera_info'.format(camera_name), sensor_msgs.msg.CameraInfo)
    cam_info_filename = os.path.join(output_dir, 'camera_info.yaml')
    utils.write_camera_model(cam_info_filename, msg)
    LOG.info(' wrote camera calibration to {}'.format(cam_info_filename))

    # Retrieve camera extrinsic
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    cam_frame = '{}_optical_frame'.format(camera_name)
    while not cam.is_localized():
        try:
            world_to_camo_transf = tf_buffer.lookup_transform(target_frame=cam_frame, source_frame='world', time=rospy.Time(0))
            world_to_camo_t, world_to_camo_q = utils.t_q_of_transf_msg(world_to_camo_transf.transform)
            cam.set_location(world_to_camo_t, world_to_camo_q)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo_throttle(1., " waiting to get camera location")
    rospy.loginfo('  retrieved pose ({})'.format(cam_frame))
    cam_extr_filename = os.path.join(output_dir, 'camera_extr.yaml')
    extr_yaml = {'camera_name':'blah', 'world_to_cam_t':world_to_camo_t, 'world_to_cam_q':world_to_camo_q}
    with open(cam_extr_filename, 'w') as f:
        #f.write("%YAML:1.0\n")
        yaml.dump(extr_yaml, f)

    # Starts remote
    remote = hog_remote.Node()
    remote_thread = threading.Thread(target=remote.run)
    remote_thread.start()
    bridge = cv_bridge.CvBridge()
    true_poses = []
    def my_pass(msg):
        #print ('in my_pass')
        pass
    rospy.Subscriber(camera_name+'/image_raw', sensor_msgs.msg.Image, my_pass, queue_size=None)
    for i, (w_to_m_t, w_to_m_eu) in enumerate(world_poses):
        print('setting pose for {}'.format(i))
        remote.set_position(w_to_m_t)
        remote.set_orientation(w_to_m_eu)
        time.sleep(0.1)
        while not remote.marker_has_arrived():
            time.sleep(0.1)
        time.sleep(5.)
        print('fetching image for pose {}'.format(i))
        msg = rospy.wait_for_message(camera_name+'/image_raw', sensor_msgs.msg.Image)
        true_pose = remote.get_marker_pose()
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            print(e)
        img_filename = 'image_{:02d}.png'.format(i)
        cv2.imwrite(os.path.join(output_dir, img_filename), cv_image)   
        true_poses.append((img_filename, true_pose))
        print(true_poses[-1])
        #print msg
        
    truth_filename = os.path.join(output_dir, 'truth.yaml')
    truth_yaml = {}
    for img_filename, true_pose in true_poses:
        truth_yaml[img_filename] = {}
        truth_yaml[img_filename]['t'] =  true_pose[0]
        truth_yaml[img_filename]['q'] =  true_pose[1]
    with open(truth_filename, 'w') as f:
        yaml.dump(truth_yaml, f)

    rospy.signal_shutdown('done')

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)
     rospy.init_node('smocap_make_gazebo_samples')
     make_samples(sys.argv)
