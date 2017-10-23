#!/usr/bin/env python
import sys, rospy, roslib, geometry_msgs.msg, nav_msgs.msg

import utils
#geometry_msgs/PoseWithCovarianceStamped

class MyNode:
    def __init__(self):
        self.truth_position = (0,0,0)
        self.est_position = (0,0,0)
        self.dif_position = [0,0,0]
        self.position_error = 0
        rospy.Subscriber('/smocap/est_world', geometry_msgs.msg.PoseWithCovarianceStamped, self.est_callback)
        rospy.Subscriber('/smocap/marker_truth', nav_msgs.msg.Odometry, self.truth_callback)

    def run(self):
        rospy.spin()


    def est_callback(self, msg):
        self.est_position = utils.list_of_position(msg.pose.pose.position)
        self.difference()
        print 'dif' , self.dif_position

    def truth_callback(self, msg):
        self.truth_position = utils.list_of_position(msg.pose.pose.position)
        self.difference()
        print 'truth', self.truth_position
        print 'est', self.est_position
        print 'dif' , self.dif_position
        print 'position error', self.position_error

    def difference(self):
        self.dif_position[0] = abs(self.truth_position[0] - self.est_position[0])
        self.dif_position[1] = abs(self.truth_position[1] - self.est_position[1])
        self.dif_position[2] = abs(self.truth_position[2] - self.est_position[2])
        self.cal_position_error()

    def cal_position_error(self):
        self.position_error = (self.dif_position[0]**2 + self.dif_position[1]**2 +self.dif_position[2]**2)**0.5


def main(args):
    rospy.init_node('calcul_erreur')
    MyNode().run()



if __name__ == '__main__':
    main(sys.argv)
