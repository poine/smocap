#!/usr/bin/env python
import sys, rospy, roslib, geometry_msgs.msg, nav_msgs.msg, tf2_ros, numpy as np
import numpy as np
import matplotlib.pyplot as plt
import utils
#geometry_msgs/PoseWithCovarianceStamped

class MyNode:
    def __init__(self):
        self.truth_position = (0,0,0)
        self.est_position = (0,0,0)
        self.dif_position = [0,0,0]
        self.position_error = 0
        self.array_position_error = np.zeros((37,28))
        rospy.Subscriber('/smocap/est_world', geometry_msgs.msg.PoseWithCovarianceStamped, self.est_callback)
        rospy.Subscriber('/smocap/marker_truth', nav_msgs.msg.Odometry, self.truth_callback)
        self.br = tf2_ros.TransformBroadcaster()
        self.counter_x = 0
        self.counter_y = 0
        self.counter = 0

        
    def run(self):
        rate = rospy.Rate(3.5)
        try:
            while not rospy.is_shutdown() and self.counter_y != 37:
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass
        print 'bye'

        with open('/tmp/foo.npz', 'wb') as f:
            np.savez(f, self.array_position_error)
            self.plot()
        rospy.spin()


    def periodic(self):
        print("count", self.counter)
        print("x", self.counter_x)
        print("y",self.counter_y)

        self.array_position_error[self.counter_y][self.counter_x] = self.position_error
        
        tf_msg = geometry_msgs.msg.TransformStamped()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "irm_link_desired"
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.transform.rotation.x = 0
        tf_msg.transform.rotation.y = 0
        tf_msg.transform.rotation.z = 0
        tf_msg.transform.rotation.w = 1.

        tf_msg.transform.translation.x = -0.5 + 0.1*(self.counter_x)
        tf_msg.transform.translation.y = -0.9 + 0.1*(self.counter_y) 
        tf_msg.transform.translation.z = 0.15  
        self.br.sendTransform(tf_msg)

        self.counter += 1
        if self.counter%28==0:
            self.counter_y += 1
        else:
            self.counter_x += (-1)**(self.counter//28)

        
    def est_callback(self, msg):
        self.est_position = utils.list_of_position(msg.pose.pose.position)
        self.difference()
        #print 'dif' , self.dif_position

    def truth_callback(self, msg):
        self.truth_position = utils.list_of_position(msg.pose.pose.position)
        self.difference()
        #print 'truth', self.truth_position
        #print 'est', self.est_position
        #print 'dif' , self.dif_position
        #print 'position error', self.position_error

    def difference(self):
        self.dif_position[0] = abs(self.truth_position[0] - self.est_position[0])
        self.dif_position[1] = abs(self.truth_position[1] - self.est_position[1])
        self.dif_position[2] = abs(self.truth_position[2] - self.est_position[2])
        self.cal_position_error() 

    def cal_position_error(self):
        self.position_error = (self.dif_position[0]**2 + self.dif_position[1]**2 +self.dif_position[2]**2)**0.5

    def plot(self):
        x = np.linspace(-.5, 2.2, 28)
        y = np.linspace(-.9, 2.8, 37)
        X, Y = np.meshgrid(x, y)

        fig_size = plt.rcParams["figure.figsize"]
        print(fig_size[0], fig_size[1])

        plt.rcParams["figure.figsize"] = fig_size

        plt.pcolormesh(X, Y, self.array_position_error)
        plt.colorbar()
        plt.savefig('graph_position_error', format='png')
        plt.show()
    
        npzfile = np.load('/tmp/foo.npz')
        print npzfile.files
        print npzfile['arr_0']

def main(args):
    rospy.init_node('calcul_erreur')
    MyNode().run()





# http://www.courspython.com/visualisation-couleur.html

if __name__ == '__main__':
    main(sys.argv)

    
