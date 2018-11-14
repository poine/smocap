#!/usr/bin/env python

import numpy as np, cv2
import utils

image_path = '/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_ethz_3_backgound_distorted.png'
cam_calib_path = '/home/poine/work/smocap.git/test/camera_ueye_enac_ceiling_1_6mm.yaml'

def test():
    img = cv2.imread(image_path)
    if img is None:
        print('unable to read image: {}'.format(image_path))
        return

    try:
        camera_matrix, dist_coeffs, w, h = utils.load_camera_model(cam_calib_path)
    except:
        print('unable to read camera calibration: {}'.format(cam_calib_path))
        return
    
    new_camera_matrix, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coeffs, (w,h), 1, (w,h))

    img2 = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    cv2.imwrite('/home/poine/work/rosmip.git/rosmip/rosmip_worlds/maps/track_ethz_3_backgound_undistorted.png', img2)

    cv2.imshow('distorted image', img)
    cv2.imshow('undistorted image', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



test()
