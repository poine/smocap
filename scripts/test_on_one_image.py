#!/usr/bin/env python
import logging, numpy as np, cv2
import smocap, utils



def test(image_path, cam_calib_path, min_area, detect_rgb):
    img = cv2.imread(image_path)
    if img is None:
        print('unable to read image: {}'.format(image_path))
        return

    try:
        camera_matrix, dist_coeffs, w, h = utils.load_camera_model(cam_calib_path)
    except:
        print('unable to read camera calibration: {}'.format(cam_calib_path))
        return
        
    cam = smocap.Camera("noname")
    cam.set_calibration(camera_matrix, dist_coeffs, w, h)
    cam.set_location([1, 1, 2.9], [0, 0, 0, 1.])
    
    smocap_ = smocap.SMoCap([cam], detect_rgb=detect_rgb, undistort=False, min_area=min_area)
   
    smocap_.detect_keypoints(img, cam_idx=0)
    print('detected {} keypoints'.format(len(smocap_.detected_kp_img)))
    smocap_.identify_keypoints()
    
    smocap_.tracking_method = smocap_.track_dumb
    smocap_.track()
    t, q = utils.tq_of_T(smocap_.marker.cam_to_irm_T)
    print('cam_to_marker: t {} q {} ( {:.1f} ms)'.format(t, q, smocap_.marker.tracking_duration*1e3))

    smocap_.tracking_method = smocap_.track_pnp
    smocap_.marker.cam_to_irm_T = np.eye(4)
    smocap_.track()
    t, q = utils.tq_of_T(smocap_.marker.cam_to_irm_T)
    print('cam_to_marker: t {} q {} ( {:.1f} ms)'.format(t, q, smocap_.marker.tracking_duration*1e3))
    
    smocap_.tracking_method = smocap_.track_in_plane
    smocap_.marker.cam_to_irm_T = np.eye(4)
    smocap_.track()
    t, q = utils.tq_of_T(smocap_.marker.cam_to_irm_T)
    print('cam_to_marker: t {} q {} ( {:.1f} ms)'.format(t, q, smocap_.marker.tracking_duration*1e3))




    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #test('../test/gazebo_samples_no_distortion/image_00.png', '../test/gazebo_samples_no_distortion/camera_info.yaml', min_area=2, detect_rgb=True)

    test1 = {}
    test('../test/image_11.png', '../test/gazebo_samples_no_distortion/camera_info.yaml', min_area=2, detect_rgb=True)
    #test('../test/ueye_ceiling_1_6mm_02.png', '../test/camera_ueye_enac_ceiling_1_6mm.yaml', min_area=12, detect_rgb=False)

