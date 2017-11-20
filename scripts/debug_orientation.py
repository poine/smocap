#!/usr/bin/env python
import logging, math, numpy as np, cv2, matplotlib.pyplot as plt, yaml, tf, scipy.optimize
import rospy, tf2_ros, sensor_msgs.msg

import smocap, utils

#
# I am trying to debug the invertion in old keypoints identification
#


# this is stolen from smocap_node
def retrieve_cameras(camera_names, img_encoding):
    ''' retrieve camera intrinsics (calibration) and extrinsics (pose) '''
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    cams = []
    for camera_name in camera_names.split(','):
        cam = smocap.Camera(camera_name, img_encoding)
        rospy.loginfo(' adding camera: "{}"'.format(camera_name))
        cam_info_topic = '/{}/camera_info'.format(camera_name)
        cam_info_msg = rospy.wait_for_message(cam_info_topic, sensor_msgs.msg.CameraInfo)
        cam.set_calibration(np.array(cam_info_msg.K).reshape(3,3), np.array(cam_info_msg.D), cam_info_msg.width, cam_info_msg.height)
        rospy.loginfo('  retrieved calibration ({})'.format(cam_info_topic))

        while not cam.is_localized():
            cam_frame = '{}_optical_frame'.format(camera_name)
            try:
                world_to_camo_transf = tf_buffer.lookup_transform(target_frame=cam_frame, source_frame='world', time=rospy.Time(0))
                world_to_camo_t, world_to_camo_q = utils.t_q_of_transf_msg(world_to_camo_transf.transform)
                cam.set_location(world_to_camo_t, world_to_camo_q)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.loginfo_throttle(1., " waiting to get camera location")
        rospy.loginfo('  retrieved pose ({})'.format(cam_frame))
        cams.append(cam)
        return cams

# this is stolen from find_world_to_cam
def load_points(path):
    with open(path, 'r') as f:
        _dict = yaml.load(f)
    ids, pts_img, pts_world = [], [], []
    for point in _dict:
        ids.append(point)
        pts_img.append([float(c) for c in _dict[point]['img'].split(',')])
        pts_world.append([float(c) for c in _dict[point]['world'].split(',')])
    return ids, np.array(pts_img), np.array(pts_world)
    

# this is stolen from smocap
def track_marker2(cam, pts_marker, pts_img, heigth_above_floor=0., verbose=0):
    ''' This is a tracker using bundle adjustment.
    x, y, theta are the position and rotation angle of the marker in world frame
    Position and orientation are constrained to the floor plane '''
        
    def irm_to_cam_T_of_params(params):
        x, y, theta = params
        irm_to_world_r, irm_to_world_t = np.array([0., 0, theta]), [x, y, heigth_above_floor]
        irm_to_world_T = smocap.utils.T_of_t_r(irm_to_world_t, irm_to_world_r)
        return np.dot(cam.world_to_cam_T, irm_to_world_T) 
            
    def residual(params):
        irm_to_cam_T = irm_to_cam_T_of_params(params)
        irm_to_cam_t, irm_to_cam_r = smocap.utils.tr_of_T(irm_to_cam_T)
        projected_kps = cv2.projectPoints(pts_marker, irm_to_cam_r, irm_to_cam_t, cam.K, cam.D)[0].squeeze()
        return (projected_kps - pts_img).ravel()

    def params_of_irm_to_world_T(irm_to_world_T):
        ''' return x,y,theta from irm_to_world transform '''
        _angle, _dir, _point = tf.transformations.rotation_from_matrix(irm_to_world_T)
        return (irm_to_world_T[0,3], irm_to_world_T[1,3], _angle)

    p0 =  params_of_irm_to_world_T(np.eye(4)) 
    res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf')
    irm_to_cam_T = irm_to_cam_T_of_params(res.x)
    #print('in track_marker2 irm_to_cam_T\n{}'.format(irm_to_cam_T))
    cam_to_irm_T = np.linalg.inv(irm_to_cam_T)
    world_to_irm_T = np.dot(cam_to_irm_T, cam.world_to_cam_T)
    return world_to_irm_T
    #marker.set_world_pose(world_to_irm_T)
    

def plot(pts_world):
    plt.scatter(pts_world[:,0], pts_world[:,1])
    ax = plt.gca()
    ax.set_aspect('equal')



def test_0(_cam):
    ''' Here i check that the world to camera orientation i computed and that I am broacasting in ROS is correct '''
    #for i in range(5): cam.D[i]= 0.
    if 1:
        pts_id, pts_img,  pts_world = load_points('../test/F111/ueye_enac_ceiling_1_extrinsic_points.yaml')
        #rep_pts_img =  cv2.projectPoints(pts_world, cam.world_to_cam_r, cam.world_to_cam_t, cam.K, cam.D)[0].squeeze()
        #print rep_pts_img
        #rep_err = np.mean(np.linalg.norm(pts_img - rep_pts_img, axis=1))
        #print 'reprojection error {} px'.format(rep_err)
        rep_pts_img = _cam.project(pts_world).squeeze()
        #print rep_pts_img
        rep_err = np.mean(np.linalg.norm(pts_img - rep_pts_img, axis=1))
        print 'reprojection error {} px'.format(rep_err)
    
def test_1(_cam, x=1., y=1., heigth_above_floor=0.09, theta=0.2):
    ''' Here, i test track marker 2 on a synthetic marker'''
    db = smocap.shapes.Database()
    s_1 = db.shapes[0]
    print('pts irm\n{}'.format(s_1.pts))

 
    irm_to_world_r, irm_to_world_t = np.array([0., 0, theta]), [x, y, heigth_above_floor]
    irm_to_world_T = smocap.utils.T_of_t_r(irm_to_world_t, irm_to_world_r)
    print('irm_to_world\n{}'.format(irm_to_world_T))

    world_to_irm_T = np.linalg.inv(irm_to_world_T)
    print('world to irm\n{}'.format(world_to_irm_T))

    
    pts_marker = s_1._pts#np.array([[p[0], p[1], 0] for p in s_1._pts])
    pts_world = np.array([smocap.utils.transform(irm_to_world_T, p) for p in pts_marker])
    print('pts world\n{}'.format(pts_world))
    pts_img = _cam.project(pts_world).squeeze()
    print('pts img\n{}'.format(pts_img))
    retrieved_world_to_irm_T = track_marker2(_cam, pts_marker, pts_img, heigth_above_floor=heigth_above_floor, verbose=0)
    print('retrieved world to irm\n{}'.format(retrieved_world_to_irm_T))
    retrieved_irm_to_world_T = np.linalg.inv(retrieved_world_to_irm_T)
    print('retrieved_irm_to_world\n{}'.format(retrieved_irm_to_world_T))
    a_w_to_m = math.atan2(irm_to_world_T[1,0], irm_to_world_T[0,0])
    print('retrieved theta {}\n'.format(a_w_to_m))
    #angle, _dir, _point = tf.transformations.rotation_from_matrix(irm_to_world_T)
    #print('retrieved theta {} {}\n'.format(angle, _dir))
    print('success' if np.allclose(world_to_irm_T, retrieved_world_to_irm_T) else 'failure')
    plot(pts_world)




def plot_foo(pts_img_sorted, pts_irm_sorted, window_title):
    ax = plt.subplot(1,2,1)
    plt.scatter(pts_img_sorted[:,0], pts_img_sorted[:,1])
    for i, p in enumerate(pts_img_sorted):
        ax.text(p[0], p[1], '{}'.format(i))
    ax.set_aspect('equal')
    plt.title('image')
    ax = plt.subplot(1,2,2)
    plt.scatter(pts_irm_sorted[:,0], pts_irm_sorted[:,1])
    for i, p in enumerate(pts_irm_sorted):
        ax.text(p[0], p[1], '{}'.format(i))
    ax.set_aspect('equal')
    plt.title('ref shape')
    plt.gcf().canvas.set_window_title(window_title)
    
def test_2(cams, path='../test/image_16.png'):
    ''' Here, i compare the old and the new tracking code'''
    img = cv2.imread(path)

    _smocap = smocap.SMoCap(cams, undistort=False, detector_cfg_path='/home/poine/work/smocap.git/params/f111_detector_default.yaml')
    cam_idx = 0

    _smocap.detect_markers_in_full_frame(img, cam_idx, None)
    ff_obs = _smocap.marker.ff_observations[cam_idx]

    print("#######\nold code")
    _smocap.marker.set_height_ablove_floor(0.09)
    _smocap.detect_marker_in_roi(img, cam_idx)
    obs = _smocap.marker.observations[cam_idx]
    print('observation\n{}'.format(obs.kps_img))
    _smocap.identify_marker_in_roi(cam_idx)
    print('sorted observation\n{}'.format(obs.keypoints_img_sorted))
    _smocap.track_marker(cam_idx, verbose=0)
    print('retrieved_irm_to_world\n{} (rep err {})'.format(_smocap.marker.irm_to_world_T, _smocap.marker.rep_err))

    plot_foo(obs.keypoints_img_sorted, _smocap.marker.ref_shape._pts, 'old code')
    plt.figure()

    
    print("#######\nnew code")
    m_id = 0
    _m = _smocap.markers[m_id]
    _m.set_height_ablove_floor(0.09)
    _smocap.detect_marker_in_roi2(img, cam_idx, None, m_id)
    obs = _m.observations[cam_idx]
    print('observation\n{}'.format(obs.kps_img))
    print('sorted observation (img)\n{}'.format(obs.keypoints_img_sorted))
    print('sorted observation (irm)\n{}'.format(_m.ref_shape._pts_sorted))
    #tmp = obs.keypoints_img_sorted[0]
    #obs.keypoints_img_sorted[0] =  obs.keypoints_img_sorted[2]
    #obs.keypoints_img_sorted[2] = tmp
    _smocap.track_marker2(_m, cam_idx, verbose=1)
    print('retrieved_irm_to_world\n{} (rep err {})'.format(_m.irm_to_world_T, _m.rep_err))


    plot_foo(obs.keypoints_img_sorted, _m.ref_shape._pts_sorted, 'new_code')
    plt.show()
    
    cv2.imshow('my window title', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=300, suppress=True)
    rospy.init_node('debug_orientation')

    # retrieve camera intrinsics and extrinsics from ROS
    cams = retrieve_cameras('ueye_enac_ceiling_1_6mm', img_encoding='mono8')
    cam = cams[0]
  
    
    #test_0(cam)
    #test_1(cam)
    test_2(cams)

    
    plt.show()
