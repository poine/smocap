#!/usr/bin/env python
'''
   I am trying to figure how to compute the homography of the infrared marker
   it doesn't work. It seems to be numerical... maybe handedness...
'''
import logging, sys, numpy as np, cv2, pdb

import smocap, utils

def to_homo(p): return np.array([p[0], p[1], 1.])
def from_homo(p): return  np.array([p[0]/p[2], p[1]/p[2]])
def round_pt(p): return (int(p[0]), int(p[1]))

def compute_dest(_p_in, _H, img_out):
    if 0:
        p_in_h = to_homo(_p_in)
        p_out_h = np.dot(_H, p_in_h)
        p_out = from_homo(p_out_h)
        print('transform {} -> {} ({})'.format(_p_in, p_out, p_out_h))
        cv2.circle(img_out, round_pt(p_out), 3, (0,255,0), 1)
    else:
        p_out = cv2.perspectiveTransform(np.array([[_p_in]]), _H).squeeze()
        print('transform {} -> {}'.format(_p_in, p_out))
        cv2.circle(img_out, round_pt(p_out), 3, (0,255,0), 1)
        #pdb.set_trace()
        
    return p_out


def test_homography(_smocap, img_out):
    src_points = _smocap.markers_body[:,:2]#*1e4
    #src_points[:,0] = -src_points[:,0]
    #src_points = np.array([[0, 0], [0, -1], [0, 1], [1, 0]]) # c, l, r, f
    print('src points\n{}'.format(src_points))
    dst_points = _smocap.detected_kp_img[_smocap.kp_of_marker]
    print('dst points\n{}'.format(dst_points))
    H, status = cv2.findHomography(src_points, dst_points)
    print('H\n{}'.format(H))
    #pdb.set_trace()
    #compute_dest(np.array([0, 0.1]), H, img_out)
    compute_dest(np.array([0.2, 0]), H, img_out)
    if 0:
        for i in np.arange(-0.1500, 0.1500, 0.0250):
            #compute_dest(np.array([0. , i], np.float64), H, img_out) # ok
            compute_dest(np.array([i , 0.], np.float64), H, img_out)  # ko

def test_foo(_smocap, img_out):
    invK = np.linalg.inv(_smocap.K)
    mid = _smocap.marker_c
    kid = _smocap.kp_of_marker[mid]
    mc_img = _smocap.detected_kp_img[kid]
    #cv2.circle(img_out, smocap.round_pt(mc_img), 3, (0,255,0), 1)
    mc_img_h = to_homo(mc_img)
    mc_cam = np.dot(invK, mc_img_h)
    pdb.set_trace()

    
            

def main(args):
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=3, linewidth=300)

    img = cv2.imread("../test/image_2.png")

    _smocap = smocap.SMoCap()
    _smocap.detect_keypoints(img)
    print('detected keypoints\n{}'.format(_smocap.detected_kp_img))
    _smocap.identify_keypoints()

    img_out = _smocap.draw_debug_on_image(img)

    if 0: # test marker identification
        mid = _smocap.marker_r
        kid = _smocap.kp_of_marker[mid]
        cv2.circle(img_out, smocap.round_pt(_smocap.detected_kp_img[kid]), 3, (0,255,0), 1)

    #test_homography(_smocap, img_out)

    test_foo(_smocap, img_out)
    
    cv2.imshow("image", img_out)
    cv2.waitKey(0)    


if __name__ == '__main__':
    main(sys.argv)
