#!/usr/bin/env python
import sys, numpy as np, cv2, matplotlib, matplotlib.pyplot as plt

import find_head_pose, utils
import pdb

''' here I separate computation and display in order to reuse fonctions from find_head_pose'''

def plot_scene(images):
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    utils.draw_camera(ax, np.eye(4), id='camera')
    for i, image in enumerate(images):
        T = image['T']
        utils.draw_thriedra(ax, T, id='Frame {}'.format(i+1))
    utils.set_3D_axes_equal()


def plot_images(images):
    plt.gcf().subplots_adjust(left=0, right=1., bottom=0.04, top=0.96,
                              hspace=0.3, wspace=0.0)

    ncol = 3
    nrow = len(images)/ncol
    for i, image in enumerate(images):
        ax = plt.gcf().add_subplot(nrow, ncol, i+1)
        plt.imshow(image['pixels'])
        for p1, p2  in zip(image['points'], image['reproj_points']):
            ax.add_patch(matplotlib.patches.Circle(p1,10, color='b'))
            ax.add_patch(matplotlib.patches.Circle(p2,10, color='r', alpha=0.5))
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        plt.title('Frame {}'.format(i+1))
        ax.text(0.5,-0.1, 'rep. err. {:.1f} px'.format(image['reproj_err']), size=12, ha="center", transform=ax.transAxes)
        


def get_chessboard_transform(filename, camera_matrix, dist_coeffs):
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   


    cb_geom, cb_size = (8, 6), 0.1
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS|cv2.CALIB_CB_SYMMETRIC_GRID
    ret, img_points = cv2.findChessboardCorners(img_gray, cb_geom, flags)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners = cv2.cornerSubPix(img_gray, img_points, (11,11), (-1,-1), criteria)
    
    print('found {} corners'.format(len(img_points)))

    world_points = np.zeros((cb_geom[0]*cb_geom[1], 3), np.float32)
    world_points[:,:2] = cb_size*np.mgrid[0:cb_geom[0],0:cb_geom[1]].T.reshape(-1,2)
    (success, rotation_vector, translation_vector) = cv2.solvePnP(world_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    print success, rotation_vector, translation_vector

    p_rep, _unused =  cv2.projectPoints(world_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rep_err = np.mean(np.linalg.norm(img_points - p_rep, axis=2))
    print 'reprojection error {} px'.format(rep_err)

    return img, img_points, p_rep, rep_err, rotation_vector, translation_vector



camera_matrix, dist_coeffs = find_head_pose.load_camera_model('/home/poine/.ros/camera_info/ueye_enac_ceiling_1_6mm.yaml')
#filenames = ['image_01.png']
filenames = ['image_{:02d}.png'.format(i) for i in range(1,16)]
print filenames
images = []
for i, filename in enumerate(filenames):
    img, im_points, re_points, re_err, r_vec, t_vec = get_chessboard_transform('../test/chessboard_on_floor_6mm_lens/'+filename, camera_matrix, dist_coeffs)
    images.append({'filename': filename,
                   'pixels':img,
                   'points': im_points.squeeze(),
                   'reproj_points': re_points.squeeze(),
                   'reproj_err': re_err,
                   'T':utils.T_of_t_r(t_vec.squeeze(), r_vec),
                   'translation': t_vec,
                   'rotation': r_vec})

plot_images(images)
plot_scene(images)
plt.show()
