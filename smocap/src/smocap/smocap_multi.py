import logging, threading, numpy as np
import tf.transformations
import scipy.optimize
import cv2

LOG = logging.getLogger('smocap_multi')

import smocap
import smocap.utils
import smocap.shapes
import smocap.detector

import pdb

#
#  This is a multi marker version of the above tracker
#  Sorry for the code duplication
#
            
class SMoCapMultiMarker:

    def __init__(self, cameras, detector_cfg_path, height_above_floor):
        LOG.info(' SMoCapMultiMarker: detector cfg path {}'.format( detector_cfg_path ))
        self.shape_database = smocap.shapes.Database()
        self.detectors = [smocap.detector.Detector(cam.img_encoding, detector_cfg_path) for cam in cameras]
        self.ff_detectors = [smocap.detector.Detector(cam.img_encoding, detector_cfg_path) for cam in cameras]
        self.cameras = cameras

        nb_cams = len(cameras) 
        self.markers = [smocap.Marker(nb_cams, rs, height_above_floor) for rs in self.shape_database.shapes]
        self.optimize_lock = threading.Lock() # fuck....

    def detect_markers_in_full_frame(self, img, cam_idx, stamp):
        '''
        Called by frame searcher thread
        '''
        keypoints, pts_img = self.ff_detectors[cam_idx].detect_ff(img)
        if len(pts_img) > 0:
            clusters_id = self.ff_detectors[cam_idx].cluster_keypoints(pts_img)
            clusters_nb = np.max(clusters_id) + 1
            clusters_pts_img = [pts_img[clusters_id == i] for i in range(clusters_nb)]
            clusters_keypoints = [np.array(keypoints)[clusters_id == i] for i in range(clusters_nb)]
            #print('in detect_markers_in_full_frame cam {} ( {} keypoints -> {} clusters )'.format(cam_idx, len(pts_img.squeeze()), clusters_nb))
            shapes = [smocap.shapes.Shape(c) for c in clusters_pts_img]
            identified_shapes = [self.shape_database.find(s) for s in shapes]
            found_shapes_ids = [ids[0] for ids in identified_shapes]
            
            if 0:
                for id_cluster,( cluster, shape, (id_shape, ref_shape)) in enumerate(zip(clusters, shapes, identified_shapes)):
                    if id_shape >= 0:
                        print('found shape #{}  in cluster #{} at {} {:.2f}'.format(id_shape, id_cluster, shape.Xc, shape.theta))
                    else:
                        print('cluster #{} not matched to any known shape'.format(id_cluster))
        else:
            found_shapes_ids = []
            
        #
        for idx_marker, marker in enumerate(self.markers):
            try:
                idx_shape = found_shapes_ids.index(idx_marker)
                roi = self.cameras[cam_idx].compute_roi(shapes[idx_shape].pts, margin=70)
                marker.set_ff_observation(cam_idx, roi, shapes[idx_shape].Xc, shapes[idx_shape].theta, clusters_keypoints[idx_shape])
            except ValueError: # idx_marker was not in found_shapes_ids
                marker.set_ff_observation(cam_idx, None, None, None, None)                



    #
    #
    # 
    def detect_marker_in_roi(self, img, cam_idx, stamp, marker):
        obs = marker.observations[cam_idx]
        # Compute Region of Interest (roi)
        if marker.is_localized: # use previous position (or predicted...) to find roi
            in_frustum, roi = marker.is_in_frustum(self.cameras[cam_idx])
            if not in_frustum:
                raise smocap.MarkerNotInFrustumException
            marker.set_roi(cam_idx, roi)
        elif marker.has_ff_observation(cam_idx): # use full frame detection - might be old...
            marker.set_roi(cam_idx, marker.ff_observations[cam_idx].roi)
        else: # 
            obs.is_valid = False
            raise smocap.MarkerNotLocalizedException    

        #print marker_id, obs.roi
        # Detect Blobs in region of interest
        obs.keypoints, obs.kps_img = self.detectors[cam_idx].detect(img, obs.roi)
        if len(obs.kps_img) != len(marker.ref_shape.pts):
            obs.is_valid = False
            raise smocap.MarkerNotDetectedException
        #print obs.kps_img
        #print('in detect_marker_in_roi2 {}'.format(obs.kps_img))
        shape = smocap.shapes.Shape(obs.kps_img)
        shape_id, shape_ref = self.shape_database.find(shape)
        if shape_id == -1 or shape_ref != marker.ref_shape:
            if shape_ref != marker.ref_shape: print 'detected wrong shape'
            obs.is_valid = False
            raise smocap.MarkerNotDetectedException
        shape.sort_points(debug=False, sort_cw=True)
        #print('in detect_marker_in_roi2 found shape {}'.format(shape_id))
        #print shape.pts[shape.angle_sort_idx]
        #print shape_ref.pts[shape.angle_sort_idx]
        obs.shape = shape # store that for now to allow debug
        obs.keypoints_img_sorted = shape.pts_sorted
        
    def track_marker(self, marker, cam_idx, verbose=0, add_bug=False):
        ''' This is a tracker using bundle adjustment.
            x, y, theta are the position and rotation angle of the marker in world frame
            Position and orientation are constrained to the floor plane '''
        cam = self.cameras[cam_idx]
        observation = marker.observations[cam_idx]

        pts_marker = np.array(marker.ref_shape._pts_sorted)
        #!!!!! WTF!!!!!
        if add_bug:
            tmp =  np.array(pts_marker[2])
            pts_marker[2] = np.array(pts_marker[3])
            pts_marker[3] = tmp
        #pdb.set_trace()
        #print('in track_marker2 kps_img\n{}'.format(observation.keypoints_img_sorted))
        #print('in track_marker2 kps_marker\n{}'.format(pts_marker))
        
        def irm_to_cam_T_of_params(params):
            x, y, theta = params
            irm_to_world_r, irm_to_world_t = np.array([0., 0, theta]), [x, y, marker.heigth_above_floor]
            irm_to_world_T = smocap.utils.T_of_t_r(irm_to_world_t, irm_to_world_r)
            return np.dot(cam.world_to_cam_T, irm_to_world_T) 
            
        def residual(params):
            irm_to_cam_T = irm_to_cam_T_of_params(params)
            irm_to_cam_t, irm_to_cam_r = smocap.utils.tr_of_T(irm_to_cam_T)
            projected_kps = cv2.projectPoints(pts_marker, irm_to_cam_r, irm_to_cam_t, cam.K, cam.D)[0].squeeze()
            return (projected_kps - observation.keypoints_img_sorted).ravel()

        def params_of_irm_to_world_T(irm_to_world_T):
            ''' return x,y,theta from irm_to_world transform '''
            _angle, _dir, _point = tf.transformations.rotation_from_matrix(irm_to_world_T)
            return (irm_to_world_T[0,3], irm_to_world_T[1,3], _angle)

        p0 =  params_of_irm_to_world_T(marker.irm_to_world_T if marker.is_localized else np.eye(4)) 
        self.optimize_lock.acquire()
        res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf')
        #res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='lm')
        self.optimize_lock.release()
        if res.success:
            observation.rep_err = res.cost
            observation.irm_to_cam_T = irm_to_cam_T_of_params(res.x)
            observation.is_valid = True
        else:
            observation.is_valid = False
            raise smocap.MarkerNotLocalizedException
        

    def localize_marker_in_world(self, marker, cam_idx, verbose=False):
        observation, cam = marker.observations[cam_idx], self.cameras[cam_idx]

        valid_obs = []
        for o in marker.observations:
            if o.is_valid: valid_obs.append(o)
        if len(valid_obs) == 0:
            marker.set_world_pose(None)
            return
        
        rep_errs = [o.rep_err for o in valid_obs]
        best_obs = np.argmin(rep_errs)
        
        if observation == valid_obs[best_obs]:
            cam_to_irm_T = np.linalg.inv(observation.irm_to_cam_T)
            world_to_irm_T = np.dot(cam_to_irm_T, cam.world_to_cam_T)
            marker.set_world_pose(world_to_irm_T)
            if verbose:
                # FIXME
                marker_id = 0
                _angle, _dir, _point = tf.transformations.rotation_from_matrix(marker.irm_to_world_T)
                fmt = 'marker {}: irm_to_world_T from cam {}: r {:.1f} deg t: {} (rep err {})'
                print(fmt.format(marker_id, cam_idx, np.rad2deg(_angle), marker.irm_to_world_T[:3, 3], rep_errs))
                #pdb.set_trace()


  

    def draw(self, img, cam):
        colors = [(0, 0, 255), (0, 128, 128)]
        for mid, m in enumerate(self.markers):
            if m.has_ff_observation(cam._id):
                o = m.get_ff_observation(cam._id)
                #pdb.set_trace()
                cv2.drawKeypoints(img, o.keypoints, img, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                if o.roi is None: return
                try:
                    pt1 = (o.roi[1].start, o.roi[0].start)
                    pt2 = (o.roi[1].stop, o.roi[0].stop) 
                    cv2.rectangle(img, pt1, pt2, colors[mid], 2)
                    cv2.putText(img, '{}:roi'.format(mid), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1., colors[mid], 2)

                    pt1 = o.centroid_img
                    pt2 = pt1 + 20*np.array([np.cos(o.orientation), np.sin(o.orientation)])
                    cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 255, 0), thickness=1)
                except AttributeError:
                    #pdb.set_trace()
                    print ("in smocap multi draw, bug")

