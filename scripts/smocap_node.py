#!/usr/bin/env python
import roslib
#roslib.load_manifest('my_package')
import sys , numpy as np, rospy, cv2, sensor_msgs.msg, geometry_msgs.msg, cv_bridge
import tf.transformations
import pdb

import smocap, utils

def round_pt(p): return (int(p[0]), int(p[1]))

class SMoCapNode:

    def __init__(self):
        self.show_gui =      rospy.get_param('~show_gui', False)
        self.publish_image = rospy.get_param('~publish_image', True)
        self.publish_truth = rospy.get_param('~publish_truth', True)
        self.publish_est =   rospy.get_param('~publish_est', True)
        camera = rospy.get_param('~camera', '/camera')
        detect_rgb = rospy.get_param('~detect_rgb', False)
        detect_min_area = rospy.get_param('~detect_min_area', 2)

        self.last_frame_time = None
        self.processing_duration = None
        self.fps, self.fps_lp = 0., 0.95

        self.smocap = smocap.SMoCap(detect_rgb, detect_min_area)

        if self.publish_image:
            self.image_pub = rospy.Publisher("/smocap/image_debug", sensor_msgs.msg.Image, queue_size=1)

        if self.publish_truth:
            self.truth_pub = rospy.Publisher('/smocap/truth', geometry_msgs.msg.PoseStamped, queue_size=1)
            self.thl = utils.TruthListener()

        if self.publish_est:
            self.est_cam_pub = rospy.Publisher('/smocap/est_cam', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)
            self.est_world_pub = rospy.Publisher('/smocap/est_world', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)
            
        self.tfl = utils.TfListener()

        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(camera+'/image_raw', sensor_msgs.msg.Image, self.img_callback)
        self.camera_info_sub = rospy.Subscriber(camera+'/camera_info', sensor_msgs.msg.CameraInfo, self.cam_info_callback)
        

            


    def cam_info_callback(self, msg):
        print msg.K, msg.D
        self.camera_info_sub.unregister()
        self.camera_info_sub = None
        self.smocap.set_camera_calibration(np.array(msg.K).reshape(3,3), np.array(msg.D))
        
    def draw_debug_image(self, img, draw_true_markers=False):
        img_with_keypoints = self.smocap.draw_debug_on_image(img)

        # draw projected markers
        if draw_true_markers:
            for i in range(4):
                loc = round_pt(self.smocap.projected_markers_img[i])
                cv2.circle(img_with_keypoints, loc, 5, (0,255,0), 1)
                cv2.putText(img_with_keypoints, self.smocap.markers_names[i], loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 

        txt = '{:5.1f} fps, {} detected, (d:{:.0f} ms)'.format(self.fps, len(self.smocap.keypoints), self.smocap.marker.detection_duration*1e3)
        h, w, d = img.shape    
        loc = (10, h-20)
        cv2.putText(img_with_keypoints, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) 
            
        return img_with_keypoints


    def do_publish_truth(self):
        msg = geometry_msgs.msg.PoseStamped()
        msg.pose = self.thl.pose
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()
        self.truth_pub.publish(msg)

    def do_publish_est(self):
        msg = geometry_msgs.msg.PoseWithCovarianceStamped()
        msg.header.frame_id = "camera_optical_frame"#"/world"
        msg.header.stamp = self.last_frame_time#rospy.Time.now()
        utils.position_and_orientation_from_T(msg.pose.pose.position, msg.pose.pose.orientation, self.smocap.cam_to_body_T)
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
        print msg.header
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
        if self.publish_truth:
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
        
        if 1:
            self.smocap.detect_keypoints(cv_image)
        else:
            u0, v0, du, dv = 700, 700, 100, 100
            self.smocap.detect_keypoints(cv_image[u0:u0+du, v0:v0+dv])
        #print('detected img points\n{}'.format(self.smocap.detected_kp_img))
        if self.smocap.keypoints_detected():
            self.smocap.identify_keypoints()
            #print('identified img points\n{}'.format(self.smocap.detected_kp_img[self.smocap.kp_of_marker]))
            #self.smocap.detected_kp_img[self.smocap.kp_of_marker[self.smocap.marker_c]]
            #pdb.set_trace()
            if self.publish_truth and body_to_world_T is not None and self.smocap.projected_markers_img is not None:
                err_img_points = np.mean(np.linalg.norm(self.smocap.projected_markers_img - self.smocap.detected_kp_img[self.smocap.kp_of_marker], axis=1))
                #print('err projected img points {:.1f} pixel'.format(err_img_points))
            if self.smocap.keypoints_identified():
                self.smocap.track()
        
        
     
        #print

        if self.publish_truth: self.do_publish_truth()
        if self.publish_est and self.smocap.cam_to_body_T is not None: self.do_publish_est()
        
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
  rospy.init_node('smocap_node')
  sn = SMoCapNode()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)



