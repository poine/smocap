#!/usr/bin/env python
import numpy as np, cv2
import smocap, utils


def test(image_path, cam_calib_path, min_area, detect_rgb):
    img = cv2.imread(image_path)
    camera_matrix, dist_coeffs, w, h = utils.load_camera_model(cam_calib_path)

    smocap_ = smocap.SMoCap(detect_rgb=detect_rgb, min_area=min_area, undistort=False)
    smocap_.set_camera_calibration(camera_matrix, dist_coeffs, w, h)
    smocap_.set_world_to_cam([1, 1, 2.9], [0, 0, 0, 1.])

    smocap_.detect_keypoints(img)
    print('detected {}'.format(smocap_.keypoints_detected()))
    smocap_.identify_keypoints()
    
    smocap_.track()
    print smocap_.cam_to_irm_T # camera to marker transform
    #smocap_.track_pnp()
    #print smocap_.cam_to_irm_T # camera to marker transform
    


if __name__ == '__main__':
    #test('../test/image_4.png', '../test/camera_gazebo.yaml', min_area=2, detect_rgb=True)
    test('../test/ueye_ceiling_1_6mm_02.png', '../test/camera_ueye_enac_ceiling_1_6mm.yaml', min_area=12, detect_rgb=False)

