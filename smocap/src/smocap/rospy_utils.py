import numpy as np
import rospy, sensor_msgs.msg, geometry_msgs.msg, visualization_msgs.msg, std_msgs.msg, cv_bridge, tf2_ros, cv2
import pdb

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
        # cv_bridge is not thread safe, so each video stream has its own bridge
        if self.img_cbk is not None:
            try:
                self.images[camera_idx] = self.bridges[camera_idx].imgmsg_to_cv2(msg, "passthrough")
                self.img_cbk(self.images[camera_idx], (camera_idx, msg.header.stamp, msg.header.seq))
            except cv_bridge.CvBridgeError as e:
                print(e)

    def get_images(self):
        return self.images
        
    def get_image(self, cam_idx):
        return self.images[cam_idx]
                
    def get_images_as_rgb(self):
        ''' return a list of RGB images - used for drawing debug '''
        rgb_images = []
        for img in self.images:
            if img is None:            # no image, make a black one
                rgb_image = np.zeros((480, 640, 3)) 
            elif len(img.shape) ==  2: # mono image
                rgb_image = np.zeros((img.shape[0],img.shape[1],3))
                for i in range(3):
                    rgb_image[:,:,i] = img
                rgb_images.append(rgb_image)
            else:                      # rgb image, just copy it
                rgb_images.append(img)
        return rgb_images






def make_2d_line(p0, p1, spacing=200, endpoint=True):
    dist = np.linalg.norm(p1-p0)
    n_pt = dist/spacing
    if endpoint: n_pt += 1
    return np.stack([np.linspace(p0[j], p1[j], n_pt, endpoint=endpoint) for j in range(2)], axis=-1)

def get_points_on_plane(rays, plane_n, plane_d):
    return np.array([-plane_d/np.dot(ray, plane_n)*ray for ray in rays])



class ImgPublisher:
    def __init__(self, cam_sys):
        img_topic = "/smocap/image_debug"
        rospy.loginfo(' publishing on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        self.cam_sys = cam_sys
        w, h = np.sum([cam.w for cam in  cam_sys.get_cameras()]), np.max([cam.h for cam in cam_sys.get_cameras()])
        self.img = 255*np.ones((h, w, 3), dtype='uint8')

    def publish(self, imgs, profiler, mocap):
        for cam, img in zip(self.cam_sys.get_cameras(), imgs):
            if  img is not None:
                mocap.draw(img, cam)
        x0 = 0
        for i, (img, cam) in enumerate(zip(imgs, self.cam_sys.get_cameras())):
            if img is not None:
                h, w = img.shape[0], img.shape[1]; x1 = x0+w
                cam_area = self.img[:h,x0:x1]
                self.img[:h,x0:x1] = img
                cv2.rectangle(cam_area, (0, 0), (x1-1, h-1), (0,255,0), 2)
                
                txt, loc, fs, fcol = '{}'.format(cam.name), (10, 40), 1.1, (0, 255, 0)
                cv2.putText(self.img[:,x0:x1], txt, loc, cv2.FONT_HERSHEY_SIMPLEX, fs, fcol, 2)
                txt, loc, fs = 'fps: {:.1f}'.format(profiler.fps[i]), (10, 90), 0.9
                cv2.putText(self.img[:,x0:x1], txt, loc, cv2.FONT_HERSHEY_SIMPLEX, fs, fcol, 2)
                txt, loc = 'skipped: {}'.format(profiler.skipped[i]), (10, 130)
                cv2.putText(self.img[:,x0:x1], txt, loc, cv2.FONT_HERSHEY_SIMPLEX, fs, fcol, 2)
                txt, loc = 'seq: {}'.format(profiler.last_frame_seq[i]), (10, 170)
                cv2.putText(self.img[:,x0:x1], txt, loc, cv2.FONT_HERSHEY_SIMPLEX, fs, fcol, 2)
                txt, loc = 'ff dur: {:.3f} s'.format(profiler.ff_duration[i]), (10, 210)
                cv2.putText(self.img[:,x0:x1], txt, loc, cv2.FONT_HERSHEY_SIMPLEX, fs, fcol, 2)
                x0 = x1
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.img, "rgb8"))


class FOVPublisher:
    def __init__(self, cam_sys):
        self.cams_fov_pub = rospy.Publisher('/smocap/cams_fov', visualization_msgs.msg.MarkerArray, queue_size=1)
        self.cam_fov_msg = visualization_msgs.msg.MarkerArray()
        for idx_cam, (cam) in enumerate(cam_sys.get_cameras()):
            img_corners = np.array([[0., 0], [cam.w, 0], [cam.w, cam.h], [0, cam.h], [0, 0]])
            borders_img = np.zeros((0,2))
            for i in range(len(img_corners)-1):
                borders_img = np.append(borders_img, make_2d_line(img_corners[i], img_corners[i+1], endpoint=True), axis=0)
            # ideal border of image ( aka undistorted ) in pixels
            borders_undistorted = cv2.undistortPoints(borders_img.reshape((1, len(borders_img), 2)), cam.K, cam.D, None, cam.K)
            # border of image in optical plan
            borders_cam = [np.dot(cam.invK, [u, v, 1]) for (u, v) in borders_undistorted.squeeze()]
            # border of image projected on floor plane (in cam frame)
            borders_floor_plane_cam = get_points_on_plane(borders_cam, cam.fp_n, cam.fp_d)
            # border of image projected on floor plane (in world frame)
            borders_floor_plane_world = np.array([np.dot(cam.cam_to_world_T[:3], p.tolist()+[1]) for p in borders_floor_plane_cam])
                
            marker = visualization_msgs.msg.Marker()
            marker.header.frame_id = "world"
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.id = idx_cam
            marker.text = cam.name
            marker.scale.x = 0.01
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = 0
            marker.pose.position.y = 0
            marker.pose.position.z = 0
            for x, y, z in borders_floor_plane_world:
                p1 = geometry_msgs.msg.Point()
                p1.x=x; p1.y=y;p1.z=z
                marker.points.append(p1)
            self.cam_fov_msg.markers.append(marker)

                
            
    def publish(self):
        self.cams_fov_pub.publish(self.cam_fov_msg)
        

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


class PoseArrayPublisher:
    def __init__(self):
        
        self.poses_pub = rospy.Publisher('/smocap/est_markers', geometry_msgs.msg.PoseArray, queue_size=1)

    
    def publish(self, Ts_m2w):
        msg = geometry_msgs.msg.PoseArray()
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()
        for T_m2w in Ts_m2w:
            pose = geometry_msgs.msg.Pose()
            smocap.utils._position_and_orientation_from_T(pose.position, pose.orientation, T_m2w)
            msg.poses.append(pose)
            
        self.poses_pub.publish(msg)
  
        
        
class StatusPublisher:
    def __init__(self):
        self.status_pub = rospy.Publisher('/smocap/status', std_msgs.msg.String, queue_size=1)

    def publish(self, _profiler):
        txt = '\nTime: {}\n'.format(rospy.Time.now().to_sec())
        txt += 'Timing:\n'
        for i, (fps, skipped, fs_dur) in enumerate(zip(_profiler.fps, _profiler.skipped, _profiler.ff_duration)):
            txt += (' camera {}: fps {:4.1f} skipped {:d}\n'.format(i, fps, skipped))
            #pdb.set_trace()
            txt += ('   ff: {:.3f}s ({:.1f}fps)\n'.format(fs_dur, 1./fs_dur))
        self.publish_txt(txt)

    def publish_txt(self, txt):
        self.status_pub.publish(txt)
          
class CamSysRetriever:
    def fetch(self, cam_names):
        cam_sys = smocap.camera_system.CameraSystem(cam_names=cam_names)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        for i, cam_name in enumerate(cam_names):
            cam = cam_sys.cameras[i]
            # Retrieve camera instrinsic 
            rospy.loginfo(' retrieving camera: "{}" configuration'.format(cam_name))
            cam_info_topic = '/{}/camera_info'.format(cam_name)
            cam_info_msg = rospy.wait_for_message(cam_info_topic, sensor_msgs.msg.CameraInfo)
            cam.set_calibration(np.array(cam_info_msg.K).reshape(3,3), np.array(cam_info_msg.D), cam_info_msg.width, cam_info_msg.height)
            rospy.loginfo('  retrieved intrinsics ({})'.format(cam_info_topic))
            # Retrieve camera extrinsic
            while not cam.is_localized():
                cam_frame = '{}_optical_frame'.format(cam.name)
                try:
                    world_to_camo_transf = self.tf_buffer.lookup_transform(target_frame=cam_frame, source_frame='world', time=rospy.Time(0))
                    world_to_camo_t, world_to_camo_q = smocap.utils.t_q_of_transf_msg(world_to_camo_transf.transform)
                    cam.set_location(world_to_camo_t, world_to_camo_q)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.loginfo_throttle(1., " waiting to get camera location")
            rospy.loginfo('  retrieved extrinsics ({})'.format(cam_frame))
            # Retrieve camera encoding
            rospy.loginfo(' retrieving camera: "{}" encoding'.format(cam_name))
            cam_img_topic = '/{}/image_raw'.format(cam_name)
            cam_img_msg = rospy.wait_for_message(cam_img_topic, sensor_msgs.msg.Image)
            cam.set_encoding(cam_img_msg.encoding)
            rospy.loginfo('  retrieved encoding ({})'.format(cam_img_msg.encoding))
        return cam_sys
