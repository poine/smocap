#!/usr/bin/env python
import roslib
#roslib.load_manifest('my_package')
import sys , numpy as np, rospy, cv2, sensor_msgs.msg, geometry_msgs.msg, cv_bridge
import tf.transformations
import pdb

import smocap, utils

def round_pt(p): return (int(p[0]), int(p[1]))

class SMoCapNode:

    def __init__(self, img_topic='/smocap/camera/image_raw'):
        self.show_gui =       rospy.get_param('~show_gui', False)
        self.publish_image =  rospy.get_param('~publish_image', True)
        self.publish_thruth = rospy.get_param('~publish_thruth', True)
        self.publish_est =    rospy.get_param('~publish_est', True)
        
        if self.publish_image:
            self.image_pub = rospy.Publisher("/smocap/image_debug", sensor_msgs.msg.Image, queue_size=1)

        if self.publish_thruth:
            self.thruth_pub = rospy.Publisher('/smocap/thruth', geometry_msgs.msg.PoseStamped, queue_size=1)
            self.thl = utils.TruthListener()

        if self.publish_est:
            self.est_pub = rospy.Publisher('/smocap/est', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)
            
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(img_topic, sensor_msgs.msg.Image, self.img_callback)

        self.tfl = utils.TfListener()

        self.smocap = smocap.SMoCap()
            
        self.last_frame_time = None
        self.processing_duration = None
        self.fps, self.fps_lp = 0., 0.5


    def draw_debug_image(self, img, draw_true_markers=False):
        img_with_keypoints = self.smocap.draw_debug_on_image(img)

        # draw projected markers
        if draw_true_markers:
            for i in range(4):
                loc = round_pt(self.smocap.projected_markers_img[i])
                cv2.circle(img_with_keypoints, loc, 5, (0,255,0), 1)
                cv2.putText(img_with_keypoints, self.smocap.markers_names[i], loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 

            
        if self.processing_duration is not None:
            loc = (10, 780)
            cv2.putText(img_with_keypoints, '{:5.1f} fps'.format(self.fps), loc, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) 
            
        return img_with_keypoints


    def do_publish_thruth(self):
        msg = geometry_msgs.msg.PoseStamped()
        msg.pose = self.thl.pose
        msg.header.frame_id = "/world"
        msg.header.stamp = rospy.Time.now()
        self.thruth_pub.publish(msg)

    def do_publish_est(self):
        msg = geometry_msgs.msg.PoseWithCovarianceStamped()
        utils.position_and_orientation_from_T(msg.pose.pose.position, msg.pose.pose.orientation, self.smocap.cam_to_body_T)
        msg.header.frame_id = "/camera_optical_frame"#"/world"
        msg.header.stamp = rospy.Time.now()
        self.est_pub.publish(msg)
        
    def img_callback(self, msg):
        if self.last_frame_time is not None:
            self.processing_duration = msg.header.stamp - self.last_frame_time
            self.fps = self.fps_lp*self.fps + (1-self.fps_lp)/self.processing_duration.to_sec()
        self.last_frame_time = msg.header.stamp
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            print(e)

        if self.smocap.world_to_cam_T is None:
            try:
                world_to_camo_t, world_to_camo_q = self.tfl.get('/world', '/camera_optical_frame')
                self.smocap.set_world_to_cam(world_to_camo_t, world_to_camo_q)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
        
        #markers_world = np.array([self.tfl.transformPoint(m_b, "base_link", "world") for m_b in self.smocap.markers_body])
        #print('markers world\n{}'.format(markers_world))

        #body_to_world_t, body_to_world_q = self.tfl.get('/world', '/base_link')#self.tfl.get('/base_link', '/world')
        #body_to_world_T = utils.T_of_quat_t(body_to_world_q, body_to_world_t)
        #print('body_to_world tf\n{}'.format(body_to_world_T))
        if self.publish_thruth:
            body_to_world_T = self.thl.get_body_to_world_T()
            #print('body_to_world gz\n{}'.format(self.thl.get_body_to_world_T()))
            if body_to_world_T is not None:
                self.smocap.project(body_to_world_T)
                #print('markers world 2\n{}'.format(self.smocap.markers_world))
                #markers_world_err = self.smocap.markers_world[:,:3] - markers_world
                #print('markers world err\n{}'.format(np.linalg.norm(markers_world_err, axis=1)))
        

        #img_points = cv2.projectPoints(markers_world, self.smocap.world_to_cam_r, np.array(self.smocap.world_to_cam_t), self.smocap.K, self.smocap.D)[0]
        #print('projected image points\n{}'.format(img_points.squeeze()))
        #print('projected image points\n{}'.format(self.smocap.projected_markers_img.squeeze()))
        
        self.smocap.detect_keypoints(cv_image)
        #print('detected img points\n{}'.format(self.smocap.detected_kp_img))
        self.smocap.identify_keypoints()
        #print('identified img points\n{}'.format(self.smocap.detected_kp_img[self.smocap.kp_of_marker]))
        #self.smocap.detected_kp_img[self.smocap.kp_of_marker[self.smocap.marker_c]]
        #pdb.set_trace()
        if self.publish_thruth and body_to_world_T is not None:
            err_img_points = np.mean(np.linalg.norm(self.smocap.projected_markers_img - self.smocap.detected_kp_img[self.smocap.kp_of_marker], axis=1))
        #print('err projected img points {:.1f} pixel'.format(err_img_points))

        self.smocap.track()
        
        
     
        #print

        if self.publish_thruth: self.do_publish_thruth()
        if self.publish_est: self.do_publish_est()
        
        if self.show_gui or self.publish_image:
            debug_image = self.draw_debug_image(cv_image)
        if self.show_gui:
            cv2.imshow("Image window", debug_image)
            cv2.waitKey(3)

        if self.publish_image:
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
            except cv_bridge.CvBridgeError as e:
                print(e)



                
def main(args):
  rospy.init_node('smocap_node', anonymous=True)
  sn = SMoCapNode()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)



