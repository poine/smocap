#!/usr/bin/env python
import logging, sys, os, math, numpy as np, cv2, gi, threading
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib, GObject
import roslib, rospy, tf,  tf2_ros, geometry_msgs.msg

import utils

class Node:
    '''
    This is a gazebo node that periodically sets world->irm_link_desired tf
    '''
    def __init__(self):
        self.br = tf2_ros.TransformBroadcaster()
        self.tf_msg = geometry_msgs.msg.TransformStamped()
        self.set_orientation((0, 0, 0))
        self.set_position((0, 0, 0))
        self.tf_msg.header.frame_id = "world"
        self.tf_msg.child_frame_id = "irm_link_desired"
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pose_err = (None, None)

    def set_orientation(self, eulers):
        self.eulers = eulers
        r = self.tf_msg.transform.rotation
        r.x, r.y, r.z, r.w = tf.transformations.quaternion_from_euler(*eulers)
        
    def set_position(self, p):
        self.pos = p
        _t = self.tf_msg.transform.translation
        _t.x, _t.y, _t.z = p
        
    def marker_has_arrived(self, tol_t=1e-2, tol_r=5e-2):
        if self.pose_err[0] is None: return False
        angle_err, _dir, _point = tf.transformations.rotation_from_matrix(tf.transformations.quaternion_matrix(self.pose_err[1]))
        return np.linalg.norm(self.pose_err[0]) < tol_t and abs(angle_err) < tol_r
        
    def run(self):
        rate = rospy.Rate(20.)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass

    def periodic(self):
        self.tf_msg.header.stamp = rospy.Time.now()
        self.br.sendTransform(self.tf_msg)
        try:
            d2a = self.tfBuffer.lookup_transform('irm_link_desired', 'irm_link_actual', rospy.Time())
            self.pose_err = utils.t_q_of_transf_msg(d2a.transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.pose_err = (None, None)

            
class GUI:
    def __init__(self):
        self.b = Gtk.Builder()
        gui_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hog_remote_gui.xml')
        self.b.add_from_file(gui_xml_path)
        self.window = self.b.get_object("window")
        self.pos_entries = [self.b.get_object("entry_pos_"+axis) for axis in ['x', 'y', 'z']]
        self.ori_entries = [self.b.get_object("entry_ori_"+axis) for axis in ['r', 'p', 'y']]
        self.window.show_all()
        
    def run(self):
        Gtk.main()
        
    def display_position(self, p):
        for i in range(3):
            self.pos_entries[i].set_text(format_pos(p[i]))

    def display_orientation(self, e):
        for i in range(3):
            self.ori_entries[i].set_text(format_rad_as_deg(e[i]))
            

def deg_of_rad(r): return r*180/math.pi
def rad_of_deg(d): return d/180*math.pi
def format_rad_as_deg(r): return '{:.1f}'.format(deg_of_rad(r)) 
def format_pos(p): return '{:.2f}'.format(p)

class App:
    def __init__(self):
        self.node = Node()
        self.gui = GUI()
        self.register_gui()
        self.timeout_id = GObject.timeout_add(200, self.on_timeout, None)
        
        
    def register_gui(self):
        self.gui.window.connect("delete-event", self.quit)
        for i in range(3):
            self.gui.pos_entries[i].connect("activate", self.gui_pos_callback)
            self.gui.ori_entries[i].connect("activate", self.gui_ori_callback)
        self.gui.display_position(self.node.pos)
        self.gui.display_orientation(self.node.eulers)
            
    def gui_ori_callback(self, entry):
        _e = list(self.node.eulers)
        for i in range(3):
            try:
                _e[i] = rad_of_deg(float(self.gui.ori_entries[i].get_text()))
            except ValueError:
                pass
        print('setting orientation to {}'.format( _e))
        self.node.set_orientation(_e)
        self.gui.display_orientation(_e)

    def gui_pos_callback(self, entry):
        _p = list(self.node.pos)
        for i in range(3):
            try:
                _p[i] = float(self.gui.pos_entries[i].get_text())
            except ValueError:
                pass
        print('setting position to {}'.format( _p))
        self.node.set_position(_p)
        self.gui.display_position(_p)

    def on_timeout(self, user_data):
        if rospy.is_shutdown(): self.quit(None, None)
        return True
    
    def run(self):
        self.ros_thread = threading.Thread(target=self.node.run)
        self.ros_thread.start()
        self.gui.run()

    def quit(self, a, b):
        rospy.signal_shutdown("just because")
        self.ros_thread.join()
        print 'ros thread ended'
        Gtk.main_quit()


def main(args):
  rospy.init_node('hog_remote')
  App().run()
  

if __name__ == '__main__':
    main(sys.argv)
