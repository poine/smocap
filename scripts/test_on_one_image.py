#!/usr/bin/env python
import numpy as np, cv2
import smocap, utils


if __name__ == '__main__':

    img = cv2.imread('../test/image_4.png')
    camera_matrix, dist_coeffs, w, h = utils.load_camera_model('../test/camera_gazebo.yaml')

    smocap = smocap.SMoCap(detect_rgb=True, min_area=2)
    smocap.set_camera_calibration(camera_matrix, dist_coeffs, w, h)
    smocap.set_world_to_cam([1, 1, 2.9], [0, 0, 0, 1.])
    smocap.detect_keypoints(img)
    smocap.identify_keypoints()
    smocap.track()
    print smocap.cam_to_irm_T # camera to marker transform
    smocap.track_pnp()
    print smocap.cam_to_irm_T # camera to marker transform
