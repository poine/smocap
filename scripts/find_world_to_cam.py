#!/usr/bin/env python

import sys , numpy as np, rospy, cv2, yaml
import pdb

import smocap, utils

'''

   Find the world to cam orientation using a set of hardcoded point correspondances

'''

def draw(cam_image, pts_id, keypoints_img, rep_keypoints_img, pts_world):
    debug_image = cam_image#cv2.cvtColor(cam_image, cv2.COLOR_GRAY2RGB)
    for i, p in enumerate(keypoints_img.astype(int)):
        cv2.circle(debug_image, tuple(p), 1, (0,0,255), -1)
        cv2.putText(debug_image, '{}'.format(pts_id[i][-2:]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(debug_image, '{}'.format(pts_world[i][:2]), tuple(p+[0, 25]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    for i, p in enumerate(rep_keypoints_img.astype(int)):
        cv2.circle(debug_image, tuple(p), 1, (0,255,0), -1)

    return debug_image

def load_points(path):
    with open(path, 'r') as f:
        _dict = yaml.load(f)
    ids, pts_img, pts_world = [], [], []
    for point in _dict:
        ids.append(point)
        pts_img.append([float(c) for c in _dict[point]['img'].split(',')])
        pts_world.append([float(c) for c in _dict[point]['world'].split(',')])
    return ids, np.array(pts_img), np.array(pts_world)



def main(args):
    if 1:
        image_path =  '../test/F111/f111_cam_1_floor_01.png'
        camera_path = '../test/F111/ueye_enac_ceiling_1_6mm.yaml'
        points_path = '../test/F111/ueye_enac_ceiling_1_extrinsic_points.yaml'
    if 0:
        image_path =  '../test/F111/f111_cam_2_floor_01.png'
        camera_path = '../test/F111/ueye_enac_ceiling_2_6mm.yaml'
        points_path = '../test/F111/ueye_enac_ceiling_2_extrinsic_points.yaml'
    if 0:
        image_path =  '../test/enac_bench/floor.png'
        camera_path = '../params/enac_demo_bench/ueye_enac_ceiling_3.yaml'
        points_path = '../test/enac_bench/floor_extrinsic_points.yaml'

        
        
    img = cv2.imread(image_path)
    if img is None:
        print('unable to read image: {}'.format(image_path))
        return

    camera_matrix, dist_coeffs, w, h = utils.load_camera_model(camera_path)
    
    pts_id, pts_img,  pts_world = load_points(points_path)

    (success, rotation_vector, translation_vector) = cv2.solvePnP(pts_world, pts_img.reshape(len(pts_img), 1, 2), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    print success, rotation_vector, translation_vector

    rep_pts_img =  cv2.projectPoints(pts_world, rotation_vector, translation_vector, camera_matrix, dist_coeffs)[0].squeeze()
    
    rep_err = np.mean(np.linalg.norm(pts_img - rep_pts_img, axis=1))
    print 'reprojection error {} px'.format(rep_err)

    world_to_cam_T = smocap.utils.T_of_t_r(translation_vector.squeeze(), rotation_vector)
    world_to_cam_t, world_to_cam_q = smocap.utils.tq_of_T(world_to_cam_T)
    print(' world_to_cam_t {} world_to_cam_q {}'.format(world_to_cam_t, world_to_cam_q))

    cam_to_world_T = np.linalg.inv(world_to_cam_T)
    cam_to_world_t, cam_to_world_q = smocap.utils.tq_of_T(cam_to_world_T)
    print(' cam_to_world_t {} cam_to_world_q {}'.format(cam_to_world_t, cam_to_world_q))

    
    debug_image = draw(img, pts_id, pts_img, rep_pts_img, pts_world)
    cv2.imshow('my window title', debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
