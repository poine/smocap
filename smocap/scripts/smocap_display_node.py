#!/usr/bin/env python3

# ~/work/smocap/smocap/scripts/smocap_display_node.py _camera:=ueye_enac_z_2

import os, sys
import math, numpy as np
import roslib, rospy, rospkg, rostopic, dynamic_reconfigure.server
import sensor_msgs.msg

import cv2

import common_vision.utils as cv_u
import common_vision.rospy_utils as cv_rpu
import common_vision.bird_eye as cv_be


import smocap.rospy_utils as smocap_rpu

class BeParamTrack:
    x0, y0, dx, dy = -0.1, 0.3, 1.2, 0.8 # bird eye area in local floor plane frame
    max_dist = 3.
    w = 640                           # bird eye image width (pixel coordinates)
    s = dy/w                          # scale
    h = int(dx/s)                     # bird eye image height


class SavyImgPublisher(cv_rpu.SavyPublisher):
    def __init__(self, cam):
        pub_topic, msg_class = '/smocap_display/image2/compressed', sensor_msgs.msg.CompressedImage
        cv_rpu.SavyPublisher.__init__(self, pub_topic, msg_class, 'SmocapDisplayNode')
        self.subscriber = None
        self.compressed_img = None
        self.bird_eye = cv_be.BirdEye(cam, BeParamTrack(), cache_filename='/tmp/be_cfg.npz', force_recompute=False)

    def _is_connected(self): return self.subscriber is not None
        
    def _connect(self):
        topic = '/ueye_enac_z_2/image_raw/compressed'
        print(f'subscribing to {topic}')
        self.subscriber = cv_rpu.SimpleSubscriber(topic, sensor_msgs.msg.CompressedImage, 'SmocapDisplayNode')
        
    def _disconnect(self):
        topic = '/ueye_enac_z_2/image_raw/compressed'
        print(f'disconnected from {topic}')
        self.subscriber.sub.unregister()
        self.subscriber = None

    def _publish(self, model, args):
        try:
            self.compressed_img = np.frombuffer(self.subscriber.get().data, np.uint8)
            self.img_bgr = cv2.imdecode(self.compressed_img, cv2.IMREAD_COLOR)
            self._draw(self.img_bgr, model, args)
        except cv_rpu.NoRXMsgException:
            self.img_bgr = np.zeros((640, 480, 3), dtype=np.int8)
        msg = sensor_msgs.msg.CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', self.img_bgr)[1]).tostring()
        self.publish(msg)

    def _draw(self, img, model, args):
        debug_img = self.bird_eye.undist_unwarp_img(self.img_bgr, model.cam, fill_bg=cv_u.chroma_green)
        ui = cv_be.UnwarpedImage()
        ui.draw_grid(debug_img, self.bird_eye, model.cam, 0.1)
        f, h, c, w = cv_u.get_default_cv_text_params()
        c = (0,256,0)
        x0, y0, txt = 10, 30, f'{model.cam.name}'
        cv2.putText(debug_img, txt, (x0, y0), f, h, c, w)
        if model.smocap_status is not None:
            dy = 40
            txt = f'fps: {model.smocap_status.fps[0]:.1f}'
            x1, y1 = 30, y0+dy
            cv2.putText(debug_img, txt, (x1, y1), f, h, c, w)
            txt, y2 = f'skipped: {model.smocap_status.skipped[0]:.1f}', y0+2*dy
            cv2.putText(debug_img, txt, (x1, y2), f, h, c, w)
            txt, y3 = f'ff: {model.smocap_status.ff_duration[0]:.1f} s', y0+3*dy
            cv2.putText(debug_img, txt, (x1, y3), f, h, c, w)
            txt, y4 = f'proc: {model.smocap_status.roi_duration[0]:.1f} s', y0+4*dy
            cv2.putText(debug_img, txt, (x1, y4), f, h, c, w)
        
        self.img_bgr = debug_img
        
        
class Node(cv_rpu.PeriodicNode):

    def __init__(self, name):
        cv_rpu.PeriodicNode.__init__(self, name)

        cam_name = rospy.get_param('~camera', 'camera')
        ref_frame = rospy.get_param('~ref_frame', 'world')
        rospy.loginfo(' using ref_frame: {}'.format(ref_frame))
        self.cam = cv_rpu.retrieve_cam(cam_name, fetch_extrinsics=True, world=ref_frame)
        self.cam.set_undistortion_param(alpha=1)

        self.publisher = SavyImgPublisher(self.cam)

        self.stat_sub = smocap_rpu.SmocapStatusSubscriber(what=name)

        
    def periodic(self):
        try:
            self.smocap_status = self.stat_sub.get()
        except cv_rpu.NoRXMsgException:
            self.smocap_status = None
        except cv_rpu.RXMsgTimeoutException:
            self.smocap_status = None
        self.publisher.publish1(self, None)


            
def main(args, name='smocap_display_node', freq_hz=5):
  rospy.init_node(name)
  Node(name).run(freq_hz)


if __name__ == '__main__':
    main(sys.argv)
