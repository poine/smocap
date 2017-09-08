#!/usr/bin/env python

import logging, os
import cv2, numpy as np

#http://www.learnopencv.com/blob-detection-using-opencv-python-c/

def detect(img):
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=3, linewidth=300)

    im = cv2.imread("../test/image_1.png")
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    lower_red_hue_range = np.array([0,  100,100]), np.array([10,255,255]) 
    upper_red_hue_range = np.array([160,100,100]), np.array([179,255,255]) 
    mask1 = cv2.inRange(hsv, *lower_red_hue_range)
    mask2 = cv2.inRange(hsv, *upper_red_hue_range)
    #mask = cv2.bitwise_or(mask1)
    masked = cv2.bitwise_and(im, im, mask= mask1)

    #red_min, red_max = [17, 15, 100], [50, 56, 200]
    #mask = cv2.inRange(im, np.array(red_min, dtype = "uint8"), np.array(red_max, dtype = "uint8"))
    #masked = cv2.bitwise_and(im, im, mask = mask)
    
    params = cv2.SimpleBlobDetector_Params()
    if 1:
        # Change thresholds
        params.minThreshold = 2;
        params.maxThreshold = 256;
        
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(255-mask1)
    #print keypoints
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    cv2.imshow("im", im)
    cv2.imshow("mask1", mask1)
    #cv2.imshow("mask2", mask2)
    #cv2.imshow("masked", masked)
    cv2.imshow("Keypoints", im_with_keypoints)

    cv2.waitKey(0)
