#!/usr/bin/env python
import logging, math, numpy as np, cv2
import tf

import smocap, utils

import smocap.camera
import smocap.smocap_multi
import pdb

LOG = logging.getLogger('test_on_one_image')

def load_camera(img_enc, cam_intr_path, cam_extr):
    try:
        camera_matrix, dist_coeffs, w, h = smocap.camera.load_intrinsics(cam_intr_path)
    except:
        print('unable to read camera calibration: {}'.format(cam_intr_path))
        return

    cam_names = ['camera_1']
    cam_sys = smocap.camera_system.CameraSystem(cam_names=cam_names)

    cam = cam_sys.get_camera(0)
    cam.set_encoding(img_enc)
    cam.set_calibration(camera_matrix, dist_coeffs, w, h)
    world_to_camo_t, world_to_camo_q = cam_extr
    cam.set_location(world_to_camo_t, world_to_camo_q)
    return cam_sys

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print('unable to read image: {}'.format(img_path))
        return
    else:
        LOG.info(' loaded image {}'.format(img_path)) 
    return img

        
def test_multi(img_path, img_enc, dtctr_cfg_path, cam_intr_path, cam_extr):
    img =  load_image(img_path)
    cam_sys = load_camera(img_enc, cam_intr_path, cam_extr)
   
    _smocap = smocap.smocap_multi.SMoCapMultiMarker(cam_sys.get_cameras(), dtctr_cfg_path, height_above_floor=0.05)
    cam_idx = 0

    _smocap.detect_markers_in_full_frame(img, cam_idx=cam_idx, stamp=None)
    for idx_marker, marker in enumerate(_smocap.markers):
        print('{}: ffobs {}'.format(idx_marker, marker.has_ff_observation(cam_idx)))
    
    for m in _smocap.markers:
        try:
            _smocap.detect_marker_in_roi(img, cam_idx, None, m)
        except smocap.MarkerNotLocalizedException:
            print 'smocap.MarkerNotLocalizedException'
        except smocap.MarkerNotDetectedException:
            print 'smocap.MarkerNotDetectedException'
        except smocap.MarkerNotInFrustumException:
            print 'smocap.MarkerNotInFrustumException'
        else:
            print 'detect_marker_in_roi suceeded'
            _smocap.track_marker(m, cam_idx)
            print(' irm_to_cam_T {}'.format(m.observations[cam_idx].irm_to_cam_T))
        finally:
            _smocap.localize_marker_in_world(m, cam_idx, verbose=True)
            if m.is_localized:
                print m.irm_to_world_T
                _angle, _dir, _point = tf.transformations.rotation_from_matrix(m.irm_to_world_T)
                print('irm_to_world_T {} deg {}'.format(np.rad2deg(_angle), m.irm_to_world_T[:3, 3]))
                
    debug_img = np.array(img)#cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    _smocap.draw(debug_img, cam_sys.get_camera(0))
    
    cv2.imshow('image', debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
def test_single(img_path, img_enc, dtctr_cfg_path, cam_intr_path, cam_extr):
    img =  load_image(img_path)
 
    cam_sys = load_camera(img_enc, cam_intr_path, cam_extr)

    _smocap = smocap.SMocapMonoMarker(cam_sys.get_cameras(), dtctr_cfg_path, height_above_floor=0.05)

    cam_idx = 0
    _smocap.detect_markers_in_full_frame(img, cam_idx=cam_idx, stamp=None)
    print('full frame observation: {}'.format( _smocap.marker.has_ff_observation(cam_idx)))

    _smocap.detect_marker_in_roi(img, cam_idx)
    _smocap.identify_marker_in_roi(cam_idx)
    _smocap.track_marker(cam_idx, verbose=True)
    _smocap.localize_marker_in_world(cam_idx)
    _smocap.track_marker(cam_idx, verbose=True)
    
   
  
    debug_img = np.array(img)#cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    _smocap.draw(debug_img, cam_sys.get_camera(0))
    

    cv2.imshow('image', debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    if 0:
        args = {
            'img_path':  '/home/poine/work/smocap/smocap/test/2_markers_diff_01.png',
            'img_enc':   'mono8',
            'dtctr_cfg_path': '/home/poine/work/smocap/smocap/params/enac_demo_z/expe_z_detector_default.yaml',
            'cam_intr_path': '/home/poine/work/smocap/smocap/params/enac_demo_z/ueye_enac_z_1.yaml',
            'cam_extr': [[1.34435324,  1.77016933,  2.82543134], [-0.70610063,  0.70809591,  0.00423334,  0.00204123]]
        }
        
    if 0:
        args = {
            'img_path':  '/home/poine/work/smocap/smocap/test/debug_2019_03/rosmip2_01.png',
            'img_enc':   'mono8',
            'dtctr_cfg_path': '/home/poine/work/smocap/smocap/params/enac_demo_z/expe_z_detector_default.yaml',
            'cam_intr_path': '/home/poine/work/smocap/smocap/params/enac_demo_z/ueye_enac_z_1.yaml',
            'cam_extr': [[1.34435324,  1.77016933,  2.82543134], [-0.70610063,  0.70809591,  0.00423334,  0.00204123]]
        }

    if 0:
        args = {
            'img_path':  '/home/poine/work/smocap/smocap/test/debug_2019_03/roscar_z_01.png ',
            'img_enc':   'mono8',
            'dtctr_cfg_path': '/home/poine/work/overlay_ws/src/smocap/smocap/params/enac_demo_z/expe_z_detector_default.yaml',
            'cam_intr_path': '/home/poine/work/smocap/smocap/params/enac_demo_z/ueye_enac_z_1.yaml',
            'cam_extr': [[1.34435324,  1.77016933,  2.82543134], [-0.70610063,  0.70809591,  0.00423334,  0.00204123]]
        }
        
    if 0:  # FIX camera intr and extr...
        args = {
            'img_path':  '/home/poine/work/smocap/smocap/test/debug_2019_03/rosmip_gazebo_z_03_r30_t11.png',
            'img_enc':   'bgr8',
            'dtctr_cfg_path': '/home/poine/work/smocap/smocap/params/gazebo_detector_cfg.yaml',
            'cam_intr_path':  '/home/poine/work/smocap/smocap/params/enac_demo_z/ueye_enac_z_1.yaml',
            'cam_extr': [[1.34435324,  1.77016933,  2.82543134], [-0.70610063,  0.70809591,  0.00423334,  0.00204123]]
        }
    
    if 0:  #
        args = {
            'img_path':  '/home/poine/work/smocap/smocap/test/image_15.png',
            'img_enc':   'mono8',
            'dtctr_cfg_path': '/home/poine/work/overlay_ws/src/smocap/smocap/params/enac_demo_z/expe_z_detector_default.yaml',
            'cam_intr_path': '/home/poine/work/smocap/smocap/params/enac_demo_z/ueye_enac_z_1.yaml',
            'cam_extr': [[1.34435324,  1.77016933,  2.82543134], [-0.70610063,  0.70809591,  0.00423334,  0.00204123]]
        }
    if 1:  #
        args = {
            'img_path':  '/home/poine/work/smocap/smocap/test/debug_2019_03/rosmip_01.jpg',
            'img_enc':   'mono8',
            'dtctr_cfg_path': '/home/poine/work/overlay_ws/src/smocap/smocap/params/ricou/blob_detector_cfg.yaml',
            'cam_intr_path': '/home/poine/work/smocap/smocap/params/ricou/ueye_enac_112.yaml',
            'cam_extr': [[1.34435324,  1.77016933,  2.82543134], [-0.70610063,  0.70809591,  0.00423334,  0.00204123]]
        }
        
    test_single(**args)
    #test_multi(**args)
