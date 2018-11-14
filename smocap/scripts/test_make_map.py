#!/usr/bin/env python

## trying to automate the ground track to map conversion
#
#  grabcut did not work 
#  https://docs.opencv.org/3.1.0/d8/d83/tutorial_py_grabcut.html#gsc.tab=0
#
#  used floodfill
#  https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
#
# looks like i alreday made something similar: /home/poine/work/julie/julie/julie_worlds/scripts/test_03_make_rosmap.py
#
#

import sys, os, logging, numpy as np, matplotlib.pyplot as plt, pickle
import rospy, tf.transformations, tf2_ros, sensor_msgs.msg
import cv2
import pdb

import two_d_guidance as tdg
import smocap, utils

LOG = logging.getLogger('test_make_map')

def get_smocap_camera_from_ros(camera_name):
    ''' retrieve camera info (extrinsic and intrinsic) from a running smocap '''
    camera = smocap.Camera(camera_name)
    rospy.init_node('SmocapConfigGetter')
    # retrieve camera calibration
    cam_info_msg = rospy.wait_for_message(camera_name+'/camera_info', sensor_msgs.msg.CameraInfo)
    camera.set_calibration(np.array(cam_info_msg.K).reshape(3,3), np.array(cam_info_msg.D), cam_info_msg.width, cam_info_msg.height)
    rospy.loginfo(' retrieved camera calibration for {}'.format(camera_name))
    # retrieve camera localization
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    while not camera.is_localized():
        try:
            world_to_camo_transf = tf_buffer.lookup_transform(source_frame='world', target_frame='{}_optical_frame'.format(camera_name), time=rospy.Time(0))
            world_to_camo_t, world_to_camo_q = utils.t_q_of_transf_msg(world_to_camo_transf.transform)
            camera.set_location(world_to_camo_t, world_to_camo_q)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo_throttle(1., " waiting to get camera location")
    rospy.loginfo(' retrieved camera location')
    return camera


def get_smocap_camera(cam_name, force_ros=False):
    ''' retrieve camera info from ros or a pickle file'''
    LOG.info(' retrieving camera {}'.format(cam_name))
    cam_filename = '/tmp/smocap_{}.pkl'.format(cam_name)
    if force_ros or not os.path.isfile(cam_filename):
        smocap_cam = get_smocap_camera_from_ros(cam_name)
        with open(cam_filename, "wb") as f:
            pickle.dump(smocap_cam, f)
    else:
        with open(cam_filename, "rb") as f:
            smocap_cam = pickle.load(f)
    return smocap_cam


def extract_track(filename, roi_lu, roi_rd, debug=False):
    ''' returns a segmented track from a camera image '''
    LOG.info(' segmenting track from image {}'.format(filename))
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # region of interest in camera image
    x0, x1, y0, y1 = roi_lu[1], roi_rd[1], roi_lu[0], roi_rd[0]
    img_roi = img[x0:x1, y0:y1]
    # blur and threshold
    img_blur = cv2.GaussianBlur(img_roi,(5,5),0)
    if debug: cv2.imshow('blured', img_blur)
    thr, img_thr = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #print('threshold {}'.format(thr))
    if debug: cv2.imshow('thresholded', img_thr)
    # floodfill
    roi_height, roi_width = img_thr.shape[:2]
    mask = np.zeros((roi_height+2, roi_width+2), np.uint8)
    newval, start_point, n_connect, mask_fill_vall = 0, (736, 246), 8, 255 # newval is ignored because we use mask_only
    flags = n_connect + mask_fill_vall*256 + cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(img_thr, mask, start_point, newval, flags=flags);
    if debug: cv2.imshow('mask', mask)
    # 
    height, width = img.shape[:2]
    img_res = np.zeros((height,width,1), np.uint8)
    cv2.bitwise_not(img_res[x0:x1, y0:y1], img_res[x0:x1, y0:y1], mask=mask[1:-1, 1:-1])
    if debug:
        cv2.imshow('res', img_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_res


def make_map(size_px, res, origin, smocap_cam, cam_img):
            
    if 0:
        map_path='/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/expe_z/track_ethz_cam1.yaml'
        _map = tdg.ROSMap(yaml_path=map_path)
        print _map.width, _map.height, _map.origin, _map.img.shape

    map_path='/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/expe_z/track_ethz_cam1_new.yaml'
    _map = tdg.ROSMap(size_px=size_px, resolution=res, origin=origin)
  
    #_map.draw_line_world([0,0], [1,0], (255, 255, 255), 5)
    map_pixels = _map.img.load()
    for px in range(_map.size_px[0]):
            for py in range(_map.size_px[1]):
                p_world = _map.pixel_to_world([px, py], 0)
                p_cam_img = smocap_cam.project(p_world.reshape((1,3))).squeeze().astype(int)
                if cam_img[p_cam_img[1], p_cam_img[0]] == 255:
                    map_pixels[px, py] = 1.
                #pdb.set_trace()

    
    _map.save(map_dir='/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/expe_z/', map_name='track_ethz_cam1_new')

def main(args):

    smocap_cam = get_smocap_camera(cam_name='ueye_enac_z_1')

    cam_image_filename = '/home/poine/work/smocap.git/test/expe_z/track_etzh_cam1.png'
    roi_lu, roi_rd = (262,113), (1630, 1160) # as displayed when clicked
    img_track = extract_track(cam_image_filename, roi_lu, roi_rd, debug=False)
    
    make_map(size_px=(965, 706), res=0.005, origin=[-0.6, 0.0, 0.0], smocap_cam=smocap_cam, cam_img=img_track)
  

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv)
