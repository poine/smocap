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
import smocap.utils as smocap_u
import smocap.cfg.smocap_display_nodeConfig

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
        self.display_mode = 2

    def set_display_mode(self, m): self.display_mode = m
    
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
            self.img_bgr = np.zeros((model.cam.h, model.cam.w, 3), dtype=np.uint8)
        except cv_rpu.RXMsgTimeoutException:
            self.img_bgr = np.zeros((model.cam.h, model.cam.w, 3), dtype=np.uint8)
        msg = sensor_msgs.msg.CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', self.img_bgr)[1]).tobytes()
        self.publish(msg)

    def _draw_cam_stats(self, img, model, x0, y0, dy, tp):
        f, h, c, w = tp
        txt, x1, y1 = f'fps: {model.smocap_status.fps[0]:.1f} seq:{model.smocap_status.header.seq}', 30, y0+dy
        cv2.putText(img, txt, (x1, y1), f, 0.8*h, c, w)
        txt, y2 = f'skipped: {model.smocap_status.skipped[0]:d}', y0+2*dy
        cv2.putText(img, txt, (x1, y2), f, 0.8*h, c, w)
        txt, y3 = f'proc: {model.smocap_status.roi_duration[0]:.2f} s (ff: {model.smocap_status.ff_duration[0]:.2f}) s', y0+3*dy
        cv2.putText(img, txt, (x1, y3), f, 0.8*h, c, w)
        #txt, y4 = f'', y0+4*dy
        #cv2.putText(img, txt, (x1, y4), f, 0.8*h, c, w)
        

        
    def _draw(self, img, model, args):
        if model.smocap_status is not None:
            t_m2w = np.array(smocap_u.list_of_position(model.smocap_status.marker_pose[0].position))
            q_m2w = smocap_u.list_of_orientation(model.smocap_status.marker_pose[0].orientation)
            T_m2w = smocap_u.T_of_t_q(t_m2w, q_m2w)
        if self.display_mode == 1:
            # Background, input image
            debug_img = self.img_bgr
            if model.smocap_status is not None:  # Draw marker
                cv_u.CamDrawer.draw_trihedral(debug_img, model.cam, T_m2w, _len=0.1)
            
            
        elif self.display_mode == 2:
            # Background, bird eye
            debug_img = self.bird_eye.undist_unwarp_img(self.img_bgr, model.cam, fill_bg=cv_u.chroma_blue)
            ui = cv_be.UnwarpedImage()
            ui.draw_grid(debug_img, self.bird_eye, model.cam, 0.1)
            if model.smocap_status is not None:
                ui.draw_trihedral(debug_img, self.bird_eye, model.cam, T_m2w, _len=0.1)
            
        # Text
        f, h, c, w = cv_u.get_default_cv_text_params()
        c = (0,256,0)
        x0, y0, txt = 10, 30, f'{model.cam.name}'
        cv2.putText(debug_img, txt, (x0, y0), f, h, c, w)
        dy = 40
        if model.smocap_status is not None:
            self._draw_cam_stats(img, model, x0, y0, dy, (f, h, c, w))
            x1=30
            txt, y5 = f'marker:', y0+5*dy
            cv2.putText(debug_img, txt, (x0, y5), f, h, c, w)
            txt, y6 = f'localized: {model.smocap_status.marker_localized[0]}', y0+6*dy
            cv2.putText(debug_img, txt, (x1, y6), f, 0.8*h, c, w)
            t_m_to_w = ' '.join(['{:6.3f}'.format(p) for p in t_m2w[:2]])
            a_m_to_w = math.atan2(T_m2w[1,0], T_m2w[0,0]) # quick yaw
            txt, y7 = 'pose {} m {:5.2f} deg'.format(t_m_to_w, np.rad2deg(a_m_to_w)), y0+7*dy
            cv2.putText(debug_img, txt, (x1, y7), f, 0.8*h, c, w)
        else:
            txt, x1, y1 = 'Status not available', 30, y0+dy
            cv2.putText(debug_img, txt, (x1, y1), f, 0.8*h, c, w)  
        self.img_bgr = debug_img


# TODO publish markers
        
        
class Node(cv_rpu.PeriodicNode):

    def __init__(self, name):
        cv_rpu.PeriodicNode.__init__(self, name)

        cam_name = rospy.get_param('~camera', 'camera')
        ref_frame = rospy.get_param('~ref_frame', 'world')
        rospy.loginfo(' using ref_frame: {}'.format(ref_frame))
        self.cam = cv_rpu.retrieve_cam(cam_name, fetch_extrinsics=True, world=ref_frame)
        self.cam.set_undistortion_param(alpha=1)

        self.publisher = SavyImgPublisher(self.cam)

        self.stat_sub = smocap_rpu.SmocapStatusSubscriber(what=name, timeout=0.75)

        self.cfg_srv = dynamic_reconfigure.server.Server(smocap.cfg.smocap_display_nodeConfig, self.cfg_callback)

    def cfg_callback(self, config, level):
        rospy.loginfo("  Reconfigure Request:")
        #try:
        #    self.pipeline.contour_finder.min_area = config['mask_min_area']
        #    self.pipeline.thresholder.set_offset(config['bridge_offset'])
        #except AttributeError: pass
        #self.pipeline.display_mode = config['display_mode']
        self.publisher.set_display_mode(config['display_mode'])
        return config
    
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
