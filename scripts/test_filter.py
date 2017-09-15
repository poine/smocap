#!/usr/bin/env python
import roslib
import sys , numpy as np, rospy, cv2, sensor_msgs.msg, geometry_msgs.msg, cv_bridge
import tf.transformations
import pdb

import utils


def print_T(txt, T):
    t, euler = T[:3, 3], tf.transformations.euler_from_matrix(T, 'sxyz')
    print('{}: {} {}'.format(txt, t, euler))

class FilterNode:
    def __init__(self):
        self.tfl = utils.TfListener()
        self.br = tf.TransformBroadcaster()
        self.w_to_irm_T = None
        self.w_to_odom_align_T = None
        self.world_to_odom_T, self.world_to_odom_t, self.world_to_odom_q = None, None, None
        self.camol_to_bl_T = None
        self.w_to_camol_T = None  # constant
        self.foo_pub = rospy.Publisher('/foo/est', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)
        self.bar_pub = rospy.Publisher('/bat/est', geometry_msgs.msg.PoseStamped, queue_size=1)
        rospy.Subscriber('/smocap/est', geometry_msgs.msg.PoseWithCovarianceStamped, self.smocap_cbk)
         
    def run(self):
        rate = rospy.Rate(20.)
        while not rospy.is_shutdown():
            self.periodic()
            rate.sleep()

    def smocap_cbk(self, msg):
        self.last_frame_time = msg.header.stamp
        ''' get robot pose in camera frame / compute pose in world frame if T_camera->world is available'''
        self.camol_to_bl_t = utils.list_of_position(msg.pose.pose.position)
        self.camol_to_bl_q = utils.list_of_orientation(msg.pose.pose.orientation)
        self.camol_to_bl_T = utils.T_of_t_q(self.camol_to_bl_t, self.camol_to_bl_q)
        # self.camol_to_bl_T, ca c'est good 
        #pose44 = np.dot(tf.listener.xyz_to_mat44(msg.pose.pose.position), tf.listener.xyzw_to_mat44(msg.pose.pose.orientation))
        #print self.camol_to_bl_T ,'\n',  pose44, '\n\n'
        
        if self.w_to_camol_T is not None:

            # checked self.w_to_camol_T
            if 1:
                _msg42 = geometry_msgs.msg.TransformStamped()
                _msg42.header.frame_id = 'world'
                test_w_to_camol_T = self.tfl.tfl.asMatrix(target_frame='/camera_optical_frame', hdr=_msg42.header)
                if not np.allclose(self.w_to_camol_T, test_w_to_camol_T):
                    print 'w_to_camol_T failure\n'

            

            # ca, c'est bad
            #self.w_to_irm_T = np.dot(self.camol_to_bl_T, self.w_to_camol_T)# c'est pas good
            # this is: bl pose in world frame = world_to_camol * bl_pose_in_camol_frame   ... wtf!
            self.w_to_irm_T = np.dot(self.w_to_camol_T, self.camol_to_bl_T)# c'est good

            msg1 = geometry_msgs.msg.PoseWithCovarianceStamped()
            utils.position_and_orientation_from_T(msg1.pose.pose.position, msg1.pose.pose.orientation, self.w_to_irm_T)
            msg1.header.frame_id = "/world"
            msg1.header.stamp = rospy.Time.now()
            self.foo_pub.publish(msg1)

        #ca c'est good
        try:
            msg2 = geometry_msgs.msg.PoseStamped()
            msg2.header.frame_id = msg.header.frame_id # /camera_optical_frame
            msg2.pose = msg.pose.pose
            msg3 = self.tfl.tfl.transformPose(target_frame='world', ps=msg2)
            #print msg3
            self.bar_pub.publish(msg3)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print 'in smocap callback: world to camera_optical_frame tf failed'

        world_to_odom_T = self.compute_world_to_odom(self.w_to_irm_T, self.last_frame_time)
        self.filter(world_to_odom_T)
            

    def filter(self, world_to_odom_measure_T):
        l = 0.99
        #measure_t, measure_q = utils.tq_of_T(world_to_odom_measure_T)
        #self.world_to_odom_t = l*self.world_to_odom_t + (1-l)*measure_t
        #print self.world_to_odom_t
        #self.world_to_odom_T = world_to_odom_measure_T
        T_e = np.dot(self.world_to_odom_T, np.linalg.inv(world_to_odom_measure_T))

        
        
    def get_world_to_cam(self):
        ''' get constant world to cam_optical_link '''
        try:
            w_to_camol_t, w_to_camol_q = self.tfl.get(_from='/world', _to='/camera_optical_frame')
            print('w_to_camol {} {}'.format(w_to_camol_t, w_to_camol_q ))
            self.w_to_camol_T = utils.T_of_t_q(w_to_camol_t, w_to_camol_q)
            # ca c'est good
            #print self.w_to_camol_T
            #_msg = geometry_msgs.msg.TransformStamped()
            #_msg.header.frame_id = 'world'
            #print self.tfl.tfl.asMatrix(target_frame='/camera_optical_frame', hdr=_msg.header)

            print("cam_link to camol {}".format(self.tfl.get(_from='/camera_link', _to='/camera_optical_frame')))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "w_to_camol_T tf failure"
        

            
    def compute_world_to_odom(self, world_to_marker_T, stamp):
        try:
            #bl_to_odom_t, bl_to_odom_q = self.tfl.get(_from='/base_link', _to='/odom')
            bl_to_odom_t, bl_to_odom_q = self.tfl.tfl.lookupTransform(target_frame='/odom', source_frame='/base_link', time=stamp)
            bl_to_odom_T =  utils.T_of_t_q(bl_to_odom_t, bl_to_odom_q)
            #print_T('bl_to_odom_T', bl_to_odom_T) # celui la est bon!!!
          
            #print_T('my w_to_irm_T (is really irm_to_w)', world_to_marker_T)
            ### ca c'est odom to world
            world_to_odom_T = np.dot(world_to_marker_T, np.linalg.inv(bl_to_odom_T))
            #print_T('world_to_odom_T', world_to_odom_T)
                
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "bl_to_odom_t tf failure"
            world_to_odom_T = None
            
        return world_to_odom_T
            
    def periodic(self):
        if self.w_to_camol_T is None:
            self.get_world_to_cam()
            
        if self.w_to_odom_align_T is None and self.w_to_irm_T is not None:
            self.w_to_odom_align_T = self.compute_world_to_odom(self.w_to_irm_T, self.last_frame_time)
            if self.w_to_odom_align_T is not None:
                self.world_to_odom_T = self.w_to_odom_align_T
                self.world_to_odom_t, self.world_to_odom_q = utils.tq_of_T(self.world_to_odom_T)
                print('computed w_to_odom_align\n{}'.format(self.w_to_odom_align_T))
            
        if False and self.w_to_odom_align_T is not None:
            if 0:
                world_to_odom_t, world_to_odom_q = [1.85, 0.38, 0], tf.transformations.quaternion_from_euler(0, 0, 0.59) #[0, 0, 0, 1]
            else:
                #print('w_to_odom_align_T {}'.format(self.w_to_odom_align_T))
                world_to_odom_t, world_to_odom_q = utils.tq_of_T(self.w_to_odom_align_T)
            # from: http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20broadcaster%20%28Python%29
            #   br.sendTransform([...], turtlename, "world")
            #   [...] publishes it as a transform from frame "world" to frame "turtleX
            # here we publish the transform from map to odom
            self.br.sendTransform(world_to_odom_t, world_to_odom_q, rospy.Time.now(), 'odom', 'map')
        else:
            if self.world_to_odom_t is not None and self.world_to_odom_q is not None:
                self.br.sendTransform(self.world_to_odom_t, self.world_to_odom_q, rospy.Time.now(), 'odom', 'map')

            
            
        
def main(args):
  rospy.init_node('dumb_filter')#, anonymous=True)
  FilterNode().run()
  

if __name__ == '__main__':
    main(sys.argv)
