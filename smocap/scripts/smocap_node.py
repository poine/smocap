#!/usr/bin/env python
import roslib
#roslib.load_manifest('my_package')
import os, sys , time, numpy as np, rospy, cv2, sensor_msgs.msg, geometry_msgs.msg, cv_bridge
import tf.transformations, tf2_ros
import threading
import pdb

import smocap, smocap_node_publisher, utils, smocap.rospy_utils

def round_pt(p): return (int(p[0]), int(p[1]))




class LossesTrapper:
    '''
    Save images and poses upon marker loss
    '''
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.lost_log_lock = threading.Lock()
        self.lost_log = []
        
    def record(self, img, pose):
        img_path = self.img_dir+'/img_{:03d}.png'.format(len(self.lost_log))
        print 'lost at {} ({})'.format(pose[:3,3], img_path)
        with self.lost_log_lock:
            self.lost_log.append(pose[:3,3])
        cv2.imwrite(img_path, img)

    def stop(self):
        with open(os.path.join(self.img_dir, 'lost_log.npz'), 'wb') as f:
            np.savez(f, self.lost_log)


        

class Timer:
    '''
    Records fps, skipped frames and processing duration for every video stream
    '''
    def __init__(self, nb_cam):
        self.last_frame_time = [None]*nb_cam
        self.processing_duration = [0.]*nb_cam
        self.frame_duration = [0.]*nb_cam
        self.fps = [0.]*nb_cam
        self.fps_lp = 0.95
        self.last_frame_seq = [0]*nb_cam
        self.skipped = [0]*nb_cam
        
    def signal_start(self, cam_idx, stamp, seq):
        if self.last_frame_time[cam_idx] is not None:
            self.frame_duration[cam_idx] = stamp - self.last_frame_time[cam_idx]
            self.fps[cam_idx] = self.fps_lp*self.fps[cam_idx] + (1-self.fps_lp)/self.frame_duration[cam_idx].to_sec()
            self.skipped[cam_idx] += seq - self.last_frame_seq[cam_idx] - 1
        self.last_frame_time[cam_idx] = stamp
        self.last_frame_seq[cam_idx] = seq
        
    def signal_done(self, cam_idx, stamp):
        self.processing_duration[cam_idx] = stamp - self.last_frame_time[cam_idx]




class FrameSearcher(threading.Thread):
    '''
    Encapsulates the threads responsible for searching markers in full frames at a lower framerate
    One for each camera
    '''
    def __init__(self, _cam_idx, _smocap):
        super(FrameSearcher, self).__init__(name='FrameSearcher_{}'.format(_smocap.cameras[_cam_idx].name))
        self.cam_idx = _cam_idx
        self.smocap = _smocap
        self.condition = threading.Condition()
        self._quit = False
        # the processed image
        self.seq = None
        self.img = None
        self.stamp = None
        
    def run(self):
        while not self._quit:
            with self.condition:
                self.condition.wait()
                if self.seq is not None:
                    #print 'working '+ self.name + " " + str(self.seq)
                    self.smocap.detect_markers_in_full_frame(self.img, self.cam_idx, self.stamp)
                    self.seq = None # indicates image has been processed
                elif self._quit:
                    return
            
    def put_image(self, img, seq, stamp):
        if not self.condition.acquire(blocking=False):
            return
        else:
            self.seq = seq
            self.img = np.copy(img)
            self.stamp = stamp
            self.condition.notify()
            self.condition.release()
        
    def stop(self):
        with self.condition:
            self._quit = True
            self.condition.notify()




            
class SMoCapNode:

    def __init__(self):
        self.publish_image = rospy.get_param('~publish_image', True)
        self.publish_est =   rospy.get_param('~publish_est', True)
        camera_names =       rospy.get_param('~cameras', 'camera_1').split(',')
        self.img_encoding =  rospy.get_param('~img_encoding', 'mono8')
        detector_cfg_path =    rospy.get_param('~detector_cfg_path', '/home/poine/work/smocap.git/smocap/params/f111_detector_default.yaml')
        self.trap_losses = rospy.get_param('~trap_losses', False)
        self.trap_losses_img_dir = rospy.get_param('~trap_losses_img_dir', '/home/poine/work/smocap.git/smocap/test/enac_demo_z/losses')
        
        self.run_multi_tracker = rospy.get_param('~run_multi_tracker', False)
        self.run_mono_tracker = rospy.get_param('~run_mono_tracker', True)
        
        rospy.loginfo('   using cameras: "{}"'.format(camera_names))
        rospy.loginfo('   using detector config: "{}"'.format(detector_cfg_path))
        rospy.loginfo('   image encoding: "{}"'.format(self.img_encoding))
        rospy.loginfo('   trapping losses: "{}"'.format(self.trap_losses))
        rospy.loginfo('   run_mono_tracker: "{}"'.format(self.run_mono_tracker))
        rospy.loginfo('   run_multi_tracker: "{}"'.format(self.run_multi_tracker))
        
        # Retrieve camera configuration from ROS
        self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(camera_names)
        self.timer = Timer(self.cam_sys.nb_cams())

        if self.trap_losses:
            self.losses_trapper = LossesTrapper(self.trap_losses_img_dir)
        
        self.smocap = smocap.SMoCap(self.cam_sys.get_cameras(), undistort=True, detector_cfg_path=detector_cfg_path)

        self.smocap.marker.heigth_above_floor = 0.15#0.09
        self.smocap.markers[0].heigth_above_floor = 0.15

        self.publisher = smocap_node_publisher.SmocapNodePublisher(self.smocap)

        
        if self.publish_image:
            self.image_pub = rospy.Publisher("/smocap/image_debug", sensor_msgs.msg.Image, queue_size=1)

        if self.publish_est:
            self.est_marker_pub = rospy.Publisher('/smocap/est_marker', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)
            self.est_body_pub = rospy.Publisher('/smocap/est_body', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)

        # Start frame searchers
        self.frame_searchers = [FrameSearcher(i, self.smocap) for i, c in enumerate(self.cam_sys.get_cameras())]
        for f in self.frame_searchers: f.start()
        
        # Subscribe to video streams
        self.bridges = [cv_bridge.CvBridge() for c in self.cam_sys.get_cameras()]
        for cam_idx, cam in enumerate(self.cam_sys.get_cameras()):
            cam_img_topic = '/{}/image_raw'.format(cam.name)
            rospy.Subscriber(cam_img_topic, sensor_msgs.msg.Image, self.img_callback, cam_idx, queue_size=1)
            rospy.loginfo(' subscribed to ({})'.format(cam_img_topic))

        
    def draw_debug_image(self, img, draw_true_markers=False):
        img_with_keypoints = self.smocap.draw_debug_on_image(img, 0)

        # draw projected markers
        if draw_true_markers:
            for i in range(4):
                loc = round_pt(self.smocap.projected_markers_img[i])
                cv2.circle(img_with_keypoints, loc, 5, (0,255,0), 1)
                cv2.putText(img_with_keypoints, self.smocap.markers_names[i], loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 


        for i in range(len(self.smocap.cameras)):
            txt = 'camera {}: {:5.1f} fps, {} detected, (d:{:.0f} ms t:{:.0f} ms)'.format(i, self.timer.fps[i], len(self.smocap.keypoints),
                                                                               self.smocap.marker.detection_duration*1e3,
                                                                               self.smocap.marker.tracking_duration*1e3)
            h, w, d = img_with_keypoints.shape    
            loc = (10, 40+(50*i))
            cv2.putText(img_with_keypoints, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) 
            
        return img_with_keypoints


    def do_publish_est(self):
        msg = geometry_msgs.msg.PoseWithCovarianceStamped()
        if 1:
            msg.header.frame_id = "world"
            msg.header.stamp = rospy.Time.now()#self.last_frame_time
            if self.run_mono_tracker:
                utils._position_and_orientation_from_T(msg.pose.pose.position, msg.pose.pose.orientation, self.smocap.marker.irm_to_world_T)
            else:
                utils._position_and_orientation_from_T(msg.pose.pose.position, msg.pose.pose.orientation, self.smocap.markers[0].irm_to_world_T)
        if 0:
            msg.header.frame_id = "ueye_enac_ceiling_1_6mm_optical_frame"
            msg.header.stamp = rospy.Time.now()#self.last_frame_time
            utils.position_and_orientation_from_T(msg.pose.pose.position, msg.pose.pose.orientation, self.smocap.marker.irm_to_cam_T)


        #std_xy, std_z, std_rxy, std_rz = 0.05, 0.01, 0.5, 0.05
        std_xy, std_z, std_rxy, std_rz = 0.1, 0.01, 0.5, 0.1
        msg.pose.covariance[0]  = msg.pose.covariance[7] = std_xy**2
        msg.pose.covariance[14] = std_z**2
        msg.pose.covariance[21] = msg.pose.covariance[28] = std_rxy**2
        msg.pose.covariance[35] = std_rz**2
        self.est_marker_pub.publish(msg)
        #utils._position_and_orientation_from_T(msg.pose.pose.position, msg.pose.pose.orientation, self.smocap.marker.body_to_world_T)
        #self.est_body_pub.publish(msg)
        
        
    def img_callback(self, msg, camera_idx):
        ''' 
        Called in each video stream thread.
        '''

        # No need for exclusion, each video stream has its own bridge and full frame detector
        try:
            cv_image = self.bridges[camera_idx].imgmsg_to_cv2(msg, "passthrough")
        except cv_bridge.CvBridgeError as e:
            print(e)
        
        if self.run_multi_tracker or self.smocap.has_unlocalized_marker(): 
            self.frame_searchers[camera_idx].put_image(cv_image, msg.header.seq, msg.header.stamp)

        if self.run_multi_tracker:
            for marker_id, marker in enumerate(self.smocap.markers):
                try:
                    self.smocap.detect_marker_in_roi2(cv_image, camera_idx, msg.header.stamp, marker_id)
                    self.smocap.track_marker2(marker, camera_idx)
                except smocap.MarkerNotInFrustumException:
                    marker.observations[camera_idx].tracked = False
                    print 'not in frustum'
                except smocap.MarkerNotDetectedException:
                    marker.observations[camera_idx].tracked = False
                    print 'not detected'
                except smocap.MarkerNotLocalizedException:
                    marker.observations[camera_idx].tracked = False
                    #print 'not localized'
                else:
                    marker.observations[camera_idx].tracked = True 
                finally:
                    if not any([o.tracked for o in marker.observations]):
                        marker.set_world_pose(None)

        if self.run_mono_tracker:
            try:
                self.timer.signal_start(camera_idx, msg.header.stamp, msg.header.seq)
                self.smocap.detect_marker_in_roi(cv_image, camera_idx)
                self.smocap.identify_marker_in_roi(camera_idx)
                self.smocap.track_marker(camera_idx)
            except smocap.MarkerNotDetectedException:
                if self.trap_losses and self.smocap.marker.observations[camera_idx].tracked: # if we just lost the marker
                    self.losses_trapper.record(cv_image, self.smocap.marker.irm_to_world_T)
                self.smocap.marker.observations[camera_idx].tracked = False
            except smocap.MarkerNotLocalizedException, smocap.MarkerNotInFrustumException:
                self.smocap.marker.observations[camera_idx].tracked = False
            else:
                self.smocap.marker.observations[camera_idx].tracked = True
                if self.publish_est:
                    self.do_publish_est()
            finally:
                if not any([o.tracked for o in self.smocap.marker.observations]):
                    self.smocap.marker.set_world_pose(None)
                self.timer.signal_done(camera_idx, rospy.Time.now())
            
                
    def run(self):
        rate = rospy.Rate(2.)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass
        if self.trap_losses:
            self.losses_trapper.stop()
        for f in self.frame_searchers:
            f.stop()

    def periodic(self):
        self.publisher.publish(self.timer, self.smocap)
                
def main(args):
  rospy.init_node('smocap_node')
  rospy.loginfo('smocap node starting')
  sn = SMoCapNode()
  sn.run()

if __name__ == '__main__':
    main(sys.argv)



