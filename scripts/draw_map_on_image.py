#!/usr/bin/env python
import roslib
import sys , numpy as np, rospy, cv2, sensor_msgs.msg, geometry_msgs.msg, cv_bridge
import tf.transformations, tf2_ros
import pdb

import smocap, utils, guidance



class DrawingNode:

    def __init__(self, camera_name = 'ueye_enac_ceiling_1_6mm',
                       camera_img_fmt = 'mono8',
                       map_path = '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_ethz_2.yaml',
                       path_path = '/home/poine/work/oscar.git/oscar/oscar_control/path_track_ethz_5.npz'):
        self.cam_img_type = camera_img_fmt
        self.camera = smocap.Camera(camera_name)
        self.retrieve_cam_info(camera_name)
        self.retrieve_cam_localization(camera_name)
       
        self.load_map(map_path)
       
        if path_path is not None:
            self.path = guidance.Path(load=path_path)

        self.draw_decorations()
        
        self.image_pub = rospy.Publisher("/smocap/image_map_debug", sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(camera_name+'/image_raw', sensor_msgs.msg.Image, self.img_callback, queue_size=1)

         
    def retrieve_cam_info(self, camera_name):
        cam_info_msg = rospy.wait_for_message(camera_name+'/camera_info', sensor_msgs.msg.CameraInfo)
        self.camera.set_calibration(np.array(cam_info_msg.K).reshape(3,3), np.array(cam_info_msg.D), cam_info_msg.width, cam_info_msg.height)
        rospy.loginfo(' retrieved camera calibration')
    
    def retrieve_cam_localization(self, camera_name):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        while not self.camera.is_localized():
            try:
                world_to_camo_transf = self.tf_buffer.lookup_transform(source_frame='world', target_frame='{}_optical_frame'.format(camera_name), time=rospy.Time(0))
                world_to_camo_t, world_to_camo_q = utils.t_q_of_transf_msg(world_to_camo_transf.transform)
                self.camera.set_location(world_to_camo_t, world_to_camo_q)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.loginfo_throttle(1., " waiting to get camera location")
        rospy.loginfo(' retrieved camera location')

    def load_map(self, map_filename):
        self.map = guidance.Map(map_filename)
        
    
    def img_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, self.cam_img_type)
        except cv_bridge.CvBridgeError as e:
            print(e)

        debug_image = self.draw_debug_image(cv_image)
        
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgra8"))
        except cv_bridge.CvBridgeError as e:
            print(e)

    def world_to_image(self, pt_world):
        return cv2.projectPoints(pt_world, self.camera.world_to_cam_r, self.camera.world_to_cam_t, self.camera.K, self.camera.D)[0].squeeze().astype(int)
        #pt_world_hom = np.array([pt_world[0], pt_world[1], pt_world[2], 1])
        #pt_cam = np.dot(self.camera.world_to_cam_T,  pt_world_hom)
        #pdb.set_trace()

    def draw_line_world(self, image, p0_w, p1_w, np_pt, color, thickness=1):
        pts_w = np.stack([np.linspace(p0_w[i], p1_w[i], np_pt) for i in range(3)], axis=-1)
        pts_img = cv2.projectPoints(pts_w, self.camera.world_to_cam_r, self.camera.world_to_cam_t, self.camera.K, self.camera.D)[0].squeeze().astype(int)
        for i in range(len(pts_img)-1):
            cv2.line(image, tuple(pts_img[i]), tuple(pts_img[i+1]), color, thickness)
        
    def draw_decorations(self, draw_map=True, draw_path=True):
        self.decorations = np.zeros((self.camera.h, self.camera.w, 4), dtype=np.uint8)
        self.decorations[:,:,3] = 200
        # draw grid
        tile_size = 0.3
        for x in range(-8, 10):
            self.draw_line_world(self.decorations, [x*tile_size, -0.1, 0], [x*tile_size, 3.5, 0], 500, (128, 128, 128))
        for y in range(0, 12):
            self.draw_line_world(self.decorations, [-2.6, y*tile_size, 0], [2.5, y*tile_size, 0], 500, (128, 128, 128)) 
        # draw world triedra
        for ep_w, c in zip(([1., 0, 0], [0., 1, 0], [0., 0, 1]), ((0,0,255), (0, 255,0), (255,0,0))):
            self.draw_line_world(self.decorations, [0, 0, 0], ep_w, 500, c)
        # draw map
        if draw_map:
            for u in range(self.map.width):
                for v in range(self.map.height):
                    pixel = self.map.img[v, u]
                    if pixel < 0.1:
                        p_w_x, p_w_y, p_w_z = self.map.world_of_pixel([u, v])
                        im_x, im_y = self.world_to_image(np.array([[p_w_x, p_w_y, p_w_z]]))
                        if im_x >= 0 and im_x < self.camera.w and  im_y >= 0 and im_y < self.camera.h:
                            self.decorations[im_y, im_x] = (128, 128, 255, 0)
        # draw path
        if draw_path:
            for i in range(len(self.path.points)-1):
                d = self.map.origin
                p0 = [self.path.points[i][0], self.path.points[i][1], 0] + d
                p1 = [self.path.points[i+1][0], self.path.points[i+1][1], 0] + d
                self.draw_line_world(self.decorations, p0, p1, 2, c)
                
        
    def draw_debug_image(self, cam_image):
        debug_image = cv2.cvtColor(cam_image, cv2.COLOR_GRAY2RGBA) # make a copy of image, insuring it is a color one
        
        
        return cv2.add(debug_image, self.decorations)



    
        
def main(args):
  rospy.init_node('smocap_draw_map_on_image')
  rospy.loginfo('smocap draw starting')
  params = {
      'camera_name': 'ueye_enac_ceiling_2_6mm',
      'camera_img_fmt': 'mono8',
      'map_path':    '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_ethz_dual.yaml',
      #'map_path':   '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_empty.yaml',
      #'map_path':    '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_ethz_2.yaml',
      #'map_path':    '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_ethz_3.yaml',
      #'map_path':    '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_ethz_4.yaml',
      #'map_path':   '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_oval_01.yaml',
      #'map_path':   '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_circle_01.yaml',
      #'path_path':   '/home/poine/work/oscar.git/oscar/oscar_control/path_track_ethz_4_1.npz'
      #'path_path':   '/home/poine/work/oscar.git/oscar/oscar_control/paths/line_02.npz'
      #'path_path':   '/home/poine/work/oscar.git/oscar/oscar_control/paths/arc_02.npz'
      'path_path':   '/home/poine/work/oscar.git/oscar/oscar_control/paths/oval_01.npz'
  }
  sn = DrawingNode(**params)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

def test():
    map_filename = '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_ethz_2.yaml'
    _map = guidance.Map(map_filename)
    def test(p_m):
        u, v = p_m 
        p_w = _map.world_of_pixel(p_m)
        print p_m, p_w

    test((0, 0))
    test((0, 500))
    test((300, 0))
    test((300, 500))
    
    debug_image = _map.img
    
    cv2.imshow('my window title', debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main(sys.argv)
    #test()
    
