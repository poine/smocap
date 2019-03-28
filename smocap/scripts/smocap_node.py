#!/usr/bin/env python
import roslib
#roslib.load_manifest('my_package')
import math, os, sys , time, numpy as np, rospy, cv2, sensor_msgs.msg, geometry_msgs.msg, cv_bridge
import tf.transformations, tf2_ros
import threading
import pdb

import smocap, smocap_node_publisher
import utils, smocap.rospy_utils, smocap.real_time_utils
import smocap.smocap_multi
import smocap.smocap_single


class NodePublisher:
    def __init__(self, cam_sys):
        self.pose_pub = smocap.rospy_utils.PosePublisher()
        self.fov_pub =  smocap.rospy_utils.FOVPublisher(cam_sys)
        self.stat_pub = smocap.rospy_utils.StatusPublisher()
        self.img_pub = smocap.rospy_utils.ImgPublisher(cam_sys)

    def publish_pose(self, T_m2w):
        self.pose_pub.publish(T_m2w)

    
    def publish_periodic(self, _profiler, _mocap, _imgs):
        self.fov_pub.publish()
        self.stat_pub.publish_txt(self.write_status(_profiler, _mocap))
        self.img_pub.publish( _imgs, _profiler, _mocap)


class MonoNodePublisher(NodePublisher):
    
    def write_status(self, _profiler, _mocap):
        txt = '\nTime: {}\n'.format(rospy.Time.now().to_sec())
        txt += 'Timing:\n'
        for i, (fps, skipped, fs_dur) in enumerate(zip(_profiler.fps, _profiler.skipped, _profiler.ff_duration)):
            txt += (' camera {}: fps {:4.1f} skipped {:d}\n'.format(i, fps, skipped))
            txt += ('   ff: {:.3f}s ({:.1f}fps)\n'.format(fs_dur, 1./fs_dur))
            txt += ('   roi: {}s\n'.format(_profiler.processing_duration[i].to_sec()))
        txt += 'Mono tracker:\n'
        txt += ' marker 0:\n'
        txt += '   localized: {}\n'.format(_mocap.marker.is_localized)
        txt += '   full frame observations: {}\n'.format([o.roi is not None for o in _mocap.marker.ff_observations] )
        t_w_to_m = ' '.join(['{:6.3f}'.format(p) for p in _mocap.marker.irm_to_world_T[:3,3]])
        a_w_to_m = math.atan2(_mocap.marker.irm_to_world_T[1,0], _mocap.marker.irm_to_world_T[0,0])
        txt += '   pose {} m {:5.2f} deg\n'.format(t_w_to_m, utils.deg_of_rad(a_w_to_m))
        return txt
        
class MultiNodePublisher(NodePublisher):

    def __init__(self, cam_sys):
        NodePublisher.__init__(self, cam_sys)
        self.poses_pub =  smocap.rospy_utils.PoseArrayPublisher()

    def publish_poses(self, _mocap):
        Ts_m2w = [m.irm_to_world_T for m in _mocap.markers]
        self.poses_pub.publish(Ts_m2w)
        
    def write_status(self, _profiler, _mocap):
        txt = '\nTime: {}\n'.format(rospy.Time.now().to_sec())
        txt += 'Timing:\n'
        for i, (fps, skipped, fs_dur) in enumerate(zip(_profiler.fps, _profiler.skipped, _profiler.ff_duration)):
            txt += (' camera {}: fps {:4.1f} skipped {:d}\n'.format(i, fps, skipped))
        for mid, m in enumerate(_mocap.markers):
            txt += 'marker {}\n'.format(mid)
            txt += '   localized: {}\n'.format(m.is_localized)
            t_w_to_m = ' '.join(['{:6.3f}'.format(p) for p in m.irm_to_world_T[:3,3]])
            a_w_to_m = math.atan2(m.irm_to_world_T[1,0], m.irm_to_world_T[0,0])
            txt += '   pose: {} m {:5.2f} deg\n'.format(t_w_to_m, utils.deg_of_rad(a_w_to_m))
            txt += '   ff_obs: {}\n'.format([m.has_ff_observation(i) for i in range(len(_mocap.cameras))])
        return txt
    
class SMoCapNode:

    def __init__(self):
        self.publish_image = rospy.get_param('~publish_image', True)
        self.publish_est =   rospy.get_param('~publish_est', True)
        camera_names =       rospy.get_param('~cameras', 'camera_1').split(',')
        detector_cfg_path =  rospy.get_param('~detector_cfg_path', '/home/poine/work/smocap.git/smocap/params/f111_detector_default.yaml')
        self.trap_losses =   rospy.get_param('~trap_losses', False)
        self.trap_losses_img_dir = rospy.get_param('~trap_losses_img_dir', '/home/poine/work/smocap.git/smocap/test/enac_demo_z/losses')
        
        self.run_multi_tracker = rospy.get_param('~run_multi_tracker', False)
        self.run_mono_tracker = rospy.get_param('~run_mono_tracker', True)
        self.height_above_floor = rospy.get_param('~height_above_floor', 0.05)
        
        rospy.loginfo('   using cameras: "{}"'.format(camera_names))
        rospy.loginfo('   using detector config: "{}"'.format(detector_cfg_path))
        rospy.loginfo('   trapping losses: "{}"'.format(self.trap_losses))
        rospy.loginfo('   run_mono_tracker: "{}"'.format(self.run_mono_tracker))
        rospy.loginfo('   run_multi_tracker: "{}"'.format(self.run_multi_tracker))
        rospy.loginfo('   height_above_floor: "{}"'.format(self.height_above_floor))
        
        # Retrieve camera configuration from ROS
        self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(camera_names)
        
        self.profiler = smocap.real_time_utils.Profiler(self.cam_sys.nb_cams())

        if self.trap_losses:
            self.losses_trapper = smocap.real_time_utils.LossesTrapper(self.trap_losses_img_dir)
        
        if self.run_mono_tracker:
            self.smocap = smocap.smocap_single.SMocapMonoMarker(self.cam_sys.get_cameras(), detector_cfg_path=detector_cfg_path, height_above_floor=self.height_above_floor)
        else:
            self.smocap = smocap.smocap_multi.SMoCapMultiMarker(self.cam_sys.get_cameras(), detector_cfg_path=detector_cfg_path, height_above_floor=self.height_above_floor)

        self.publisher = MonoNodePublisher(self.cam_sys) if self.run_mono_tracker else MultiNodePublisher(self.cam_sys)

        # Start frame searchers
        self.frame_searchers = [smocap.real_time_utils.FrameSearcher(i, self.smocap, self.profiler) for i, c in enumerate(self.cam_sys.get_cameras())]
        
        # Subscribe to video streams
        self.cam_listener = smocap.rospy_utils.CamerasListener(cams=camera_names, cbk=self.img_callback_mono if self.run_mono_tracker else self.img_callback_multi)

 
        
        
    def img_callback_mono(self, cv_image, (camera_idx, stamp, seq)):
        ''' 
        Called in main thread.
        '''
        
        #if self.smocap.has_unlocalized_marker(): 
        self.frame_searchers[camera_idx].put_image(cv_image, seq, stamp)
        try:
            self.profiler.signal_start(camera_idx, stamp, seq)
            self.smocap.detect_marker_in_roi(cv_image, camera_idx)
            self.smocap.identify_marker_in_roi(camera_idx)
            self.smocap.track_marker(camera_idx)
        except smocap.MarkerLostException:
            if self.trap_losses: # if we just lost the marker
                self.losses_trapper.record(camera_idx, cv_image, self.smocap.marker.irm_to_world_T)
        except (smocap.MarkerNotDetectedException, smocap.MarkerNotLocalizedException, smocap.MarkerNotInFrustumException):
            pass
        else:
            if self.publish_est:
                if not self.smocap.marker.is_localized :
                    print('error publishing non localized pose')
                else:
                    self.publisher.publish_pose(self.smocap.marker.irm_to_world_T)
        finally:
            self.smocap.localize_marker_in_world(camera_idx)
            self.profiler.signal_done(camera_idx, rospy.Time.now())
            

    def img_callback_multi(self, cv_image, (camera_idx, stamp, seq)):
        # 
        self.profiler.signal_start(camera_idx, stamp, seq)
        # post full image for slow full frame searchers
        self.frame_searchers[camera_idx].put_image(cv_image, seq, stamp)
        
        for m in self.smocap.markers:
            try:
                self.smocap.detect_marker_in_roi(cv_image, camera_idx, stamp, m)
            except (smocap.MarkerNotLocalizedException, smocap.MarkerNotDetectedException, smocap.MarkerNotInFrustumException):
                pass
            else:
                self.smocap.track_marker(m, camera_idx)
                self.publisher.publish_poses(self.smocap) # why is this done before localize in world?
            finally:
                self.smocap.localize_marker_in_world(m, camera_idx)
                if m.is_localized: self.publisher.publish_pose(m.irm_to_world_T)
        self.profiler.signal_done(camera_idx, rospy.Time.now())



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
        self.publisher.publish_periodic(self.profiler, self.smocap, self.cam_listener.get_images_as_rgb())

                
def main(args):
  rospy.init_node('smocap_node')
  rospy.loginfo('smocap node starting')
  sn = SMoCapNode()
  sn.run()

if __name__ == '__main__':
    main(sys.argv)



