#!/usr/bin/env python
import logging, numpy as np, cv2
import smocap, utils

LOG = logging.getLogger('test_on_one_image')


def test_identify(keypoints, keypoints_img):
    print keypoints_img
    

def test(image_path, cam_calib_path, min_area):
    img = cv2.imread(image_path)
    if img is None:
        print('unable to read image: {}'.format(image_path))
        return
    else:
        LOG.info(' loaded image {}'.format(image_path)) 

    try:
        camera_matrix, dist_coeffs, w, h = utils.load_camera_model(cam_calib_path)
    except:
        print('unable to read camera calibration: {}'.format(cam_calib_path))
        return
        
    cam = smocap.Camera("noname", encoding='bgr8')
    cam.set_calibration(camera_matrix, dist_coeffs, w, h)
    cam.set_location([1, 1, 2.9], [0, 0, 0, 1.])
    
    _smocap = smocap.SMoCap([cam], undistort=False, min_area=min_area)
   
    _smocap.detect_keypoints(img, cam_idx=0)
    #_smocap.detect_markers_in_full_frame(img, cam_idx=0)
    print('detected {} keypoints'.format(len(_smocap.detected_kp_img)))
    test_identify(_smocap.keypoints, _smocap.detected_kp_img)
    

    _smocap.identify_keypoints()
    
    #_smocap.tracking_method = _smocap.track_dumb
    #_smocap.track()
    #t, q = utils.tq_of_T(_smocap.marker.world_to_irm_T)
    #print('world_to_marker: t {} q {} ( {:.1f} ms)'.format(t, q, _smocap.marker.tracking_duration*1e3))

    #_smocap.tracking_method = _smocap.track_pnp
    #_smocap.marker.cam_to_irm_T = np.eye(4)
    #_smocap.track()
    #t, q = utils.tq_of_T(_smocap.marker.world_to_irm_T)
    #print('world_to_marker: t {} q {} ( {:.1f} ms)'.format(t, q, _smocap.marker.tracking_duration*1e3))
    
    _smocap.tracking_method = _smocap.track_in_plane
    _smocap.marker.cam_to_irm_T = np.eye(4)
    _smocap.track()
    t, q = utils.tq_of_T(_smocap.marker.world_to_irm_T)
    print('world_to_marker: t {} q {} ( {:.1f} ms)'.format(t, q, _smocap.marker.tracking_duration*1e3))

    debug_img = _smocap.draw_debug_on_image(img, camera_idx=0)
    cv2.imshow('image', debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #test('../test/gazebo_samples_no_distortion/image_00.png', '../test/gazebo_samples_no_distortion/camera_info.yaml', min_area=2)

    test('../test/image_13.png', '../test/gazebo_samples_no_distortion/camera_info.yaml', min_area=2)
    #test('../test/ueye_ceiling_1_6mm_02.png', '../test/camera_ueye_enac_ceiling_1_6mm.yaml', min_area=12, detect_rgb=False)

