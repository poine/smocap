#!/usr/bin/env python
import sys, numpy as np, rospy, roslib, geometry_msgs.msg, nav_msgs.msg, tf, tf2_ros
import pdb
import utils

class MyNode:
    def __init__(self):
        rospy.Subscriber('/smocap/marker_truth', nav_msgs.msg.Odometry, self.truth_callback)
        # tf2
        #self.tfBuffer = tf2_ros.Buffer()
        #self.tfl = tf2_ros.TransformListener(tfBuffer)
        # tf1
        self.tfl = tf.TransformListener()

        
    def truth_callback(self, msg):
        #print msg
        print 'truth ', utils.list_of_position(msg.pose.pose.position)
        
        # tf1
        irm_to_camo_t, irm_to_camo_q = self.tfl.lookupTransform(target_frame='camera_optical_frame', source_frame='irm_link_actual', time=rospy.Time())
        #pdb.set_trace()
        irm_to_camo_T = utils.T_of_t_q(np.array(irm_to_camo_t), np.array(irm_to_camo_q))
        #print irm_to_camo_T
        print np.array(irm_to_camo_t), np.array(irm_to_camo_q)
        # tf2
        #trans = self.tfBuffer.lookup_transform(target_frame='camera_optical_frame', source_frame='irm_link_actual', time=rospy.Time())
        

        
    def run(self):
        rospy.spin()





def main(args):
    rospy.init_node('calcul_erreur')
    MyNode().run()



if __name__ == '__main__':
    main(sys.argv)
