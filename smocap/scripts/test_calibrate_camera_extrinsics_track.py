#!/usr/bin/env python3

import os, sys, logging
import math, numpy as np

import cv2
LOG = logging.getLogger('calibrate_intrisinc')

import common_vision.utils as cv_u
import common_vision.plot_utils as cv_pu
import common_vision.camera as cv_c


# I want to try to obtain extrinsics by performing a PnP on the track


def main(args):

    bgr_img = cv2.imread('/home/poine/work/smocap/smocap/test/ricou/table_1.png')
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    print(gray_img.shape, gray_img.dtype)
    
    intr_cam_calib_path = '/home/poine/.ros/camera_info/ueye_enac_z_2.yaml'
    cam = cv_c.load_cam_from_files(intr_cam_calib_path, extr_path=None)

    #img2 = cam.undistort_img_map(gray_img)
    #img2 = cv2.undistort(img, camera_matrix, dist_coeffs, None, undist_camera_matrix)

    blurred_img = cv2.GaussianBlur(gray_img, (9, 9), 0)
    ret, threshold = cv2.threshold(blurred_img, 180, 255, cv2.THRESH_BINARY)
    
    cnts, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    cnt_max = max(cnts, key=cv2.contourArea)

    tl, br = np.min(cnt_max, axis=0).squeeze(), np.max(cnt_max, axis=0).squeeze()
    print(tl, br)

    cropped_img = blurred_img[tl[1]:br[1]+1, tl[0]:br[0]+1]
    inv_img = 255-cropped_img
    ret, threshold2 = cv2.threshold(inv_img, 140, 255, cv2.THRESH_BINARY)


    cnt_color, cntmax_color, thickness = (255,0,0), (0, 255,0), 3
    cv2.circle(bgr_img, tuple(tl), 1, (0,0,255), -1)
    cv2.circle(bgr_img, tuple(br), 1, (0,0,255), -1)
    cv2.rectangle(bgr_img, tuple(tl), tuple(br), (0,0,255), 1)
    

    
    cv2.drawContours(bgr_img, cnts, -1, cnt_color, 2)
    cv2.drawContours(bgr_img, cnt_max, -1, cntmax_color, 3)

    
    
    #cv2.imshow('input image', bgr_img)
    #cv2.imshow('threshold', threshold)
    #cv2.imshow('undistorted image', img2)
    cv2.imshow('inv image', threshold2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
