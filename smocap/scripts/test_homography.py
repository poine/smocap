#!/usr/bin/env python

import logging, os
import cv2, numpy as np



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=3, linewidth=300)


    im_in = cv2.imread("../test/track_ethz_lines.png")

    src_corners = np.array([[1179, 63], [2245, 187], [2457, 1507], [108, 705]])

    w, h = 1000, 1100
    dest_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    H, status = cv2.findHomography(src_corners, dest_corners)

    print H

    im_out = cv2.warpPerspective(im_in, H, (w, h))

    #cv2.imshow("im_in", im_in)
    cv2.imshow("im_out", im_out)
    cv2.waitKey(0)
    
