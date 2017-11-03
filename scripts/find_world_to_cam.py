#!/usr/bin/env python
import roslib
import sys , numpy as np, rospy, cv2, sensor_msgs.msg, geometry_msgs.msg, cv_bridge
import tf.transformations, tf2_ros
import pdb

import smocap, utils

'''

   Find the world to cam orienetation using a set of hardcoded point correspondances

'''

def draw(cam_image, keypoints_img, rep_keypoints_img):
    debug_image = cam_image#cv2.cvtColor(cam_image, cv2.COLOR_GRAY2RGB)
    for i, p in enumerate(keypoints_img.astype(int)):
        cv2.circle(debug_image, tuple(p), 1, (0,0,255), -1)
        cv2.putText(debug_image, '{}'.format(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    for i, p in enumerate(rep_keypoints_img.astype(int)):
        cv2.circle(debug_image, tuple(p), 1, (0,255,0), -1)

    return debug_image

def main(args):
    image_path = '../test/ueye_ceiling_1_6mm_calib.png'
    img = cv2.imread(image_path)
    if img is None:
        print('unable to read image: {}'.format(image_path))
        return

    camera_path = '/home/poine/work/smocap.git/test/camera_ueye_enac_ceiling_1_6mm.yaml'
    camera_matrix, dist_coeffs, w, h = utils.load_camera_model(camera_path)

    
    pts_img = np.array([(1541., 1118),
                        (1558,   601),
                        (1546,    80),
                        (768,   1146),
                        (765,    609),
                        (764.,    58),
                        (121,   1121),
                        (100,    607),
                        (110,     82)
                        ])

    pts_world = 0.3*np.array([(0.,  0, 0),
                              (4.,  0, 0),
                              (8.,  0, 0),
                              (0.,  6, 0),
                              (4.,  6, 0),
                              (8.,  6, 0),
                              (0., 11, 0),
                              (4., 11, 0),
                              (8., 11, 0)
                              ])

    (success, rotation_vector, translation_vector) = cv2.solvePnP(pts_world, pts_img.reshape(len(pts_img), 1, 2), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    print success, rotation_vector, translation_vector

    rep_pts_img =  cv2.projectPoints(pts_world, rotation_vector, translation_vector, camera_matrix, dist_coeffs)[0].squeeze()
    
    rep_err = np.mean(np.linalg.norm(pts_img - rep_pts_img, axis=1))
    print 'reprojection error {} px'.format(rep_err)

    world_to_cam_T = utils.T_of_t_r(translation_vector.squeeze(), rotation_vector)
    #print world_to_cam_T
    world_to_cam_t, world_to_cam_q = utils.tq_of_T(world_to_cam_T)
    print(' world_to_cam_t {} world_to_cam_q {}'.format(world_to_cam_t, world_to_cam_q))

    cam_to_world_T = np.linalg.inv(world_to_cam_T)
    cam_to_world_t, cam_to_world_q = utils.tq_of_T(cam_to_world_T)
    print(' cam_to_world_t {} cam_to_world_q {}'.format(cam_to_world_t, cam_to_world_q))

    
    debug_image = draw(img, pts_img, rep_pts_img)
    cv2.imshow('my window title', debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
