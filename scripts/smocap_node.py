#!/usr/bin/env python
import roslib
#roslib.load_manifest('my_package')
import sys , numpy as np, rospy, cv2, sensor_msgs.msg, geometry_msgs.msg, cv_bridge
import tf.transformations, tf2_ros
import pdb

import smocap, utils

def round_pt(p): return (int(p[0]), int(p[1]))

class SMoCapNode:

    def __init__(self):
        self.publish_image = rospy.get_param('~publish_image', True)
        self.publish_est =   rospy.get_param('~publish_est', True)
        camera_name =        rospy.get_param('~camera', '/camera')
        self.detect_rgb =    rospy.get_param('~detect_rgb', False)
        detect_min_area =    rospy.get_param('~detect_min_area', 2)

        self.last_frame_time = None
        self.processing_duration = None
        self.fps, self.fps_lp = 0., 0.95

        self.smocap = smocap.SMoCap(self.detect_rgb, undistort=True, min_area=detect_min_area)

        cam_info_msg = rospy.wait_for_message(camera_name+'/camera_info', sensor_msgs.msg.CameraInfo)
        self.smocap.set_camera_calibration(np.array(cam_info_msg.K).reshape(3,3), np.array(cam_info_msg.D), cam_info_msg.width, cam_info_msg.height)
        rospy.loginfo(' retrieved camera calibration')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        while not self.smocap.camera.is_localized():
            try:
                world_to_camo_transf = self.tf_buffer.lookup_transform('world', 'camera_optical_frame', rospy.Time(0))
                world_to_camo_t, world_to_camo_q = utils.t_q_of_transf_msg(world_to_camo_transf.transform)
                self.smocap.set_world_to_cam(world_to_camo_t, world_to_camo_q)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.loginfo_throttle(1., " waiting to get camera location")
        rospy.loginfo(' retrieved camera location')
            
        if self.publish_image:
            self.image_pub = rospy.Publisher("/smocap/image_debug", sensor_msgs.msg.Image, queue_size=1)

        if self.publish_est:
            self.est_cam_pub = rospy.Publisher('/smocap/est_cam', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)
            self.est_world_pub = rospy.Publisher('/smocap/est_world', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)
            
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(camera_name+'/image_raw', sensor_msgs.msg.Image, self.img_callback, queue_size=1)
        
    def draw_debug_image(self, img, draw_true_markers=False):
        img_with_keypoints = self.smocap.draw_debug_on_image(img)

        # draw projected markers
        if draw_true_markers:
            for i in range(4):
                loc = round_pt(self.smocap.projected_markers_img[i])
                cv2.circle(img_with_keypoints, loc, 5, (0,255,0), 1)
                cv2.putText(img_with_keypoints, self.smocap.markers_names[i], loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 

        txt = '{:5.1f} fps, {} detected, (d:{:.0f} ms)'.format(self.fps, len(self.smocap.keypoints), self.smocap.marker.detection_duration*1e3)
        h, w, d = img_with_keypoints.shape    
        loc = (10, h-20)
        cv2.putText(img_with_keypoints, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) 
            
        return img_with_keypoints


    def do_publish_est(self):
        msg = geometry_msgs.msg.PoseWithCovarianceStamped()
        msg.header.frame_id = "camera_optical_frame"
        msg.header.stamp = self.last_frame_time#rospy.Time.now()
        utils.position_and_orientation_from_T(msg.pose.pose.position, msg.pose.pose.orientation, self.smocap.marker.cam_to_body_T)
        #std_xy, std_z, std_rxy, std_rz = 0.05, 0.01, 0.5, 0.05
        std_xy, std_z, std_rxy, std_rz = 0.1, 0.01, 0.5, 0.1
        msg.pose.covariance[0]  = msg.pose.covariance[7] = std_xy**2
        msg.pose.covariance[14] = std_z**2
        msg.pose.covariance[21] = msg.pose.covariance[28] = std_rxy**2
        msg.pose.covariance[35] = std_rz**2
        self.est_cam_pub.publish(msg)
        msg.header.frame_id = "world"
        utils.position_and_orientation_from_T(msg.pose.pose.position, msg.pose.pose.orientation, self.smocap.world_to_body_T)
        self.est_world_pub.publish(msg)
        
        
    def img_callback(self, msg):
        if self.last_frame_time is not None:
            self.processing_duration = msg.header.stamp - self.last_frame_time
            self.fps = self.fps_lp*self.fps + (1-self.fps_lp)/self.processing_duration.to_sec()
        self.last_frame_time = msg.header.stamp
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8" if self.detect_rgb else "mono8")
        except cv_bridge.CvBridgeError as e:
            print(e)

        #markers_world = np.array([self.tfl.transformPoint(m_b, "base_link", "world") for m_b in self.smocap.markers_body])
        #print('markers world\n{}'.format(markers_world))

        #body_to_world_t, body_to_world_q = self.tfl.get('/world', '/base_link')#self.tfl.get('/base_link', '/world')
        #body_to_world_T = utils.T_of_quat_t(body_to_world_q, body_to_world_t)
        #print('body_to_world tf\n{}'.format(body_to_world_T))

        #img_points = cv2.projectPoints(markers_world, self.smocap.world_to_cam_r, np.array(self.smocap.world_to_cam_t), self.smocap.K, self.smocap.D)[0]
        #print('projected image points\n{}'.format(img_points.squeeze()))
        #print('projected image points\n{}'.format(self.smocap.projected_markers_img.squeeze()))
        
        self.smocap.detect_keypoints(cv_image)
        #print('detected img points\n{}'.format(self.smocap.detected_kp_img))
        if self.smocap.keypoints_detected():
            self.smocap.identify_keypoints()
            #print('identified img points\n{}'.format(self.smocap.detected_kp_img[self.smocap.kp_of_marker]))
            #self.smocap.detected_kp_img[self.smocap.kp_of_marker[self.smocap.marker_c]]
            #pdb.set_trace()
            if self.smocap.keypoints_identified():
                #self.smocap.track_pnp()
                self.smocap.track()
        
        
        if self.publish_est and self.smocap.marker.cam_to_body_T is not None: self.do_publish_est()
        
        if self.publish_image:
            debug_image = self.draw_debug_image(cv_image)
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
            except cv_bridge.CvBridgeError as e:
                print(e)



                
def main(args):
  rospy.init_node('smocap_node')
  rospy.loginfo('smocape starting')
  sn = SMoCapNode()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)



