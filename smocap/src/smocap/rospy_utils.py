import numpy as np
import rospy, sensor_msgs.msg, geometry_msgs.msg, cv_bridge, tf2_ros

import smocap.utils

class CamerasListener:
    def __init__(self, **kwargs):
        cam_names = kwargs.get('cams', ['camera_1'])
        for cam_idx, cam_name in enumerate(cam_names):
            cam_img_topic = '/{}/image_raw'.format(cam_name)
            rospy.Subscriber(cam_img_topic, sensor_msgs.msg.Image, self.img_callback, cam_idx, queue_size=1)
            rospy.loginfo(' subscribed to ({})'.format(cam_img_topic))
        self.img_cbk = kwargs.get('cbk', None)
        self.bridges = [cv_bridge.CvBridge() for c in cam_names]
        self.images = [None for c in cam_names]
            

    def img_callback(self, msg, camera_idx):
        # cv_bridge is not thread safe: each video stream has its own bridge
        if self.img_cbk is not None:
            try:
                self.images[camera_idx] = self.bridges[camera_idx].imgmsg_to_cv2(msg, "passthrough")
                self.img_cbk(self.images[camera_idx], camera_idx)
            except cv_bridge.CvBridgeError as e:
                print(e)

    def get_image(self, cam_idx):
        return self.images[cam_idx]
                
        
class ImgPublisher:
    def __init__(self):
        img_topic = "/smocap/image_debug"
        rospy.loginfo(' publishing on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()

    def publish(self, img):
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "rgb8"))


class PosePublisher:
    def __init__(self):
         self.pose_pub = rospy.Publisher('/smocap/est_marker', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)

    def publish(self, T_m2w):
        msg = geometry_msgs.msg.PoseWithCovarianceStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()#self.last_frame_time
        smocap.utils._position_and_orientation_from_T(msg.pose.pose.position, msg.pose.pose.orientation, T_m2w)
        std_xy, std_z, std_rxy, std_rz = 0.1, 0.01, 0.5, 0.1
        msg.pose.covariance[0]  = msg.pose.covariance[7] = std_xy**2
        msg.pose.covariance[14] = std_z**2
        msg.pose.covariance[21] = msg.pose.covariance[28] = std_rxy**2
        msg.pose.covariance[35] = std_rz**2
        self.pose_pub.publish(msg)
        

class CamSysRetriever:
    def fetch(self, cam_names):
        cam_sys = smocap.camera_system.CameraSystem(cam_names=cam_names)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        for i, cam_name in enumerate(cam_names):
            cam = cam_sys.cameras[i]
            rospy.loginfo(' retrieving camera: "{}" configuration'.format(cam_name))
            cam_info_topic = '/{}/camera_info'.format(cam_name)
            cam_info_msg = rospy.wait_for_message(cam_info_topic, sensor_msgs.msg.CameraInfo)
            cam.set_calibration(np.array(cam_info_msg.K).reshape(3,3), np.array(cam_info_msg.D), cam_info_msg.width, cam_info_msg.height)
            rospy.loginfo('  retrieved intrinsics ({})'.format(cam_info_topic))
            while not cam.is_localized():
                cam_frame = '{}_optical_frame'.format(cam.name)
                try:
                    world_to_camo_transf = self.tf_buffer.lookup_transform(target_frame=cam_frame, source_frame='world', time=rospy.Time(0))
                    world_to_camo_t, world_to_camo_q = smocap.utils.t_q_of_transf_msg(world_to_camo_transf.transform)
                    cam.set_location(world_to_camo_t, world_to_camo_q)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.loginfo_throttle(1., " waiting to get camera location")
            rospy.loginfo('  retrieved extrinsics ({})'.format(cam_frame))
        return cam_sys
