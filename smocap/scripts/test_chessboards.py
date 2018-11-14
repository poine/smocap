#!/usr/bin/env python
import logging, os, sys, numpy as np, cv2, matplotlib, matplotlib.pyplot as plt, pickle

import utils
import pdb

'''
   I am trying to calibrate the world to camera pose
'''


def image_coord(pt_w, T_w_to_cam, camera_matrix, dist_coeffs):
    pt_cam = np.dot(T_w_to_cam, pt_w)
    pt_img = cv2.projectPoints(world_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)



def plot_scene(images, normal_floor_cam, z_floor_cam):
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    utils.draw_camera(ax, np.eye(4), id='camera') # draw camera
    for i, image in enumerate(images): # draw chessboards
        T = image['T'] # T_cb_to_cam
        utils.draw_thriedra(ax, T, id='Frame {}'.format(i+1), alpha=0.2)
    # draw floor plane
    def z_floor(x, y): return z_floor_cam - x*normal_floor_cam[0]/normal_floor_cam[2] - y*normal_floor_cam[1]/normal_floor_cam[2]
    for x in range(-1,2):
        xys = [[x, -2], [x, 2]]
        p = np.array([[xy[0], xy[1], z_floor(*xy)] for xy in xys])
        plt.plot(p[:,0], p[:,1], p[:,2], '-k')
    for y in range(-2,3):
        xys = [[-1.5, y], [1.5, y]]
        p = np.array([[xy[0], xy[1], z_floor(*xy)] for xy in xys])
        plt.plot(p[:,0], p[:,1], p[:,2], '-k')


def plot_images(images):
    plt.gcf().subplots_adjust(left=0, right=1., bottom=0.04, top=0.96,
                              hspace=0.3, wspace=0.0)
    ncol = 3
    nrow = len(images)/ncol
    for i, image in enumerate(images):
        ax = plt.gcf().add_subplot(nrow, ncol, i+1)
        plt.imshow(image['pixels'])
        for p1, p2  in zip(image['points'], image['reproj_points']):
            ax.add_patch(matplotlib.patches.Circle(p1,5, color='b'))
            ax.add_patch(matplotlib.patches.Circle(p2,5, color='r', alpha=0.5))
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        plt.title('Frame {}'.format(i+1))
        ax.text(0.5,-0.1, 'rep. err. {:.1f} px'.format(image['reproj_err']), size=12, ha="center", transform=ax.transAxes)
        


def compute_world_to_cam(chessboards):
    for chessboard in chessboards:
        #print chessboard['T'] # chessboard to camera
        #print('{}: norm {}'.format(os.path.basename(chessboard['filename']), chessboard['T'][:3,2]))
        pass
    normal_floor_cam = np.mean(np.array([chessboard['T'][:3,2] for chessboard in chessboards]), axis=0)
    print('normal_floor_cam: {}'.format(normal_floor_cam))
    zs_cam = []
    for chessboard in chessboards:
        T = chessboard['T'] # _cb_to_cam
        A = np.array([[T[0,0], T[0,1], 0],[T[1,0], T[1,1], 0],[ T[2,0], T[2,1], -1]])
        Y = np.array([[-T[0,3]],[-T[1,3]],[-T[2,3]]])
        #print A, Y
        x_cb, y_cb, z_cam = np.dot(np.linalg.inv(A), Y)
        #print  x_cb, y_cb, z_cam
        zs_cam.append(z_cam)
    z_floor_cam = np.mean(np.array(zs_cam))
    print('dist to floor: {}'.format( z_floor_cam))
    print('Floor plane equation in cam frame: {:.4f}*x+ {:.4f}*y + {:.4f}*z + {:.4f} = 0'.format(normal_floor_cam[0], normal_floor_cam[1], normal_floor_cam[2], z_floor_cam*normal_floor_cam[2]))
    



    return normal_floor_cam, z_floor_cam

    
        
def get_chessboard_transform(filename, camera_matrix, dist_coeffs):
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cb_geom, cb_size = (8, 6), 0.1
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS|cv2.CALIB_CB_SYMMETRIC_GRID
    ret, img_points = cv2.findChessboardCorners(img_gray, cb_geom, flags)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners = cv2.cornerSubPix(img_gray, img_points, (11,11), (-1,-1), criteria)
    
    print('{}: found {} corners'.format(os.path.basename(filename), len(img_points)))

    world_points = np.zeros((cb_geom[0]*cb_geom[1], 3), np.float32)
    world_points[:,:2] = cb_size*np.mgrid[0:cb_geom[0],0:cb_geom[1]].T.reshape(-1,2)
    (success, rotation_vector, translation_vector) = cv2.solvePnP(world_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    #print success, rotation_vector.squeeze(), translation_vector.squeeze()

    p_rep, _unused =  cv2.projectPoints(world_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rep_err = np.mean(np.linalg.norm(img_points - p_rep, axis=2))
    print ' rot {} trans {} reprojection error {} px'.format(rotation_vector.squeeze(), translation_vector.squeeze(), rep_err)

    return img, img_points, p_rep, rep_err, rotation_vector, translation_vector





def compute_chessboard_poses():
    camera_matrix, dist_coeffs, w, h = utils.load_camera_model('../test/camera_ueye_enac_ceiling_1_6mm.yaml')
    #filenames = ['image_01.png']
    filenames = ['image_{:02d}.png'.format(i) for i in range(1,16)]
    #print filenames
    chessboards = []
    for i, filename in enumerate(filenames):
        img, im_points, re_points, re_err, r_vec, t_vec = get_chessboard_transform('../test/chessboard_on_floor_6mm_lens/'+filename, camera_matrix, dist_coeffs)
        chessboards.append({'filename': filename,
                       'pixels':img,
                       'points': im_points.squeeze(),
                       'reproj_points': re_points.squeeze(),
                       'reproj_err': re_err,
                       'T':utils.T_of_t_r(t_vec.squeeze(), r_vec),
                       'translation': t_vec,
                       'rotation': r_vec})
    return chessboards
        
def main(_compute_chessboards_poses):
    res_filename = '/tmp/chessboards.pkl'
    if _compute_chessboards_poses:
        chessboards = compute_chessboard_poses()
        with open(res_filename, "wb") as f:
            pickle.dump(chessboards, f)
    else:
        with open(res_filename, "rb") as f:
            chessboards = pickle.load(f)

    
    normal_floor_cam, z_floor_cam = compute_world_to_cam(chessboards) 
    plot_images(chessboards)
    plot_scene(chessboards, normal_floor_cam, z_floor_cam)
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(_compute_chessboards_poses=False)
