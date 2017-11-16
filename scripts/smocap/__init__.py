#!/usr/bin/env python

import logging, os, timeit, yaml
import math, numpy as np, cv2, tf.transformations
import scipy.optimize
import scipy.spatial.distance
import sklearn.cluster

import utils
import pdb

import smocap.shapes

LOG = logging.getLogger('smocap')

def round_pt(p): return (int(p[0]), int(p[1]))

class MarkerNotDetectedException(Exception):
    pass

class MarkerNotInFrustumException(Exception):
    pass

class MarkerNotLocalizedException(Exception):
    pass
        
class Detector:

    param_names = [
        'blobColor',
        'filterByArea',
        'filterByCircularity',
        'filterByColor',
        'filterByConvexity',
        'filterByInertia',
        'maxArea',
        'maxCircularity',
        'maxConvexity',
        'maxInertiaRatio',
        'maxThreshold',
        'minArea',
        'minCircularity',
        'minConvexity',
        'minDistBetweenBlobs',
        'minInertiaRatio',
        'minRepeatability',
        'minThreshold',
        'thresholdStep' ]
    
    
    def __init__(self, img_encoding, cfg_path=None):
        self.img_encoding = img_encoding
        if img_encoding in ['rgb8', 'bgr8']:
            self.lower_red_hue_range = np.array([0,  100,100]), np.array([10,255,255]) 
            self.upper_red_hue_range = np.array([160,100,100]), np.array([179,255,255])
        self.params = cv2.SimpleBlobDetector_Params()
        if cfg_path is not None:
            self.load_cfg(cfg_path)
        else:
            self.detector = cv2.SimpleBlobDetector_create(self.params)

    def detect(self, img, roi):
        if self.img_encoding == 'rgb8':
            hsv = cv2.cvtColor(img[roi], cv2.COLOR_RGB2HSV)
            self.sfc = cv2.inRange(hsv, *self.lower_red_hue_range)
            #mask2 = cv2.inRange(hsv, *self.upper_red_hue_range)
        elif self.img_encoding == 'bgr8':
            hsv = cv2.cvtColor(img[roi], cv2.COLOR_BGR2HSV)
            self.sfc = cv2.inRange(hsv, *self.lower_red_hue_range)
            #mask2 = cv2.inRange(hsv, *self.upper_red_hue_range)
        elif self.img_encoding == 'mono8':
            self.sfc = img[roi]
        keypoints = self.detector.detect(self.sfc)

        img_coords = np.array([kp.pt for kp in keypoints])
        if len(img_coords) > 0:
            img_coords += [roi[1].start, roi[0].start] 
        return keypoints, img_coords


    def detect_ff(self, img):
        h, w = img.shape[:2]
        roi = slice(0, h), slice(0, w)
        return self.detect(img, roi)


    def cluster_keypoints(self, kps_img, max_dist=50.):
        Y1 = scipy.spatial.distance.pdist(kps_img)
        Y2 = scipy.spatial.distance.squareform(Y1)
        db = sklearn.cluster.DBSCAN(eps=max_dist, min_samples=1, metric='precomputed')
        y_db = db.fit_predict(Y2)
        return y_db

        
    def load_cfg(self, path):
        with open(path, 'r') as stream:   
            d = yaml.load(stream)
        for k in d:
            setattr(self.params, k, d[k])
        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def save_cfg(self, path):
        d = {}
        for p in Detector.param_names:
            d[p] =  getattr(self.params, p)
        with open(path, 'w') as stream:
            yaml.dump(d, stream, default_flow_style=False)
    
    def update_param(self, name, value):
        setattr(self.params, name, value)
        self.detector = cv2.SimpleBlobDetector_create(self.params)

            

class Observation:
    def __init__(self):
        self.roi = None                # region of interest in wich to look for the marker
        self.centroid_img = None       # coordinates of marker centroid in image frame
        self.tracked = False
        self.kps_img = None
        
    def set_roi(self, roi):
        self.roi = roi

        
class Marker:
    # keypoints index
    kp_id_c, kp_id_l, kp_id_r, kp_id_f = range(4)
    # keypoints names
    kps_names = ['c', 'l', 'r', 'f']
    
    def __init__(self, nb_cams):

        # Constants:
        #   coordinates of keypoints in marker frame
        self.kps_m = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
        #   constrain marker to floor plan 
        self.heigth_above_floor = 0
        self.irm_to_body_T = np.eye(4)
        
        # Marker and body poses
        self.set_world_pose(None)
        # timing statistics
        self.detection_duration = 0. # duration of blob detection
        self.tracking_duration = 0.

        self.ff_observations = [Observation() for i in range(nb_cams)]
        self.observations = [Observation() for i in range(nb_cams)]
        

    def set_height_ablove_floor(self, _h):
        self.heigth_above_floor = _h
        self.irm_to_body_T[2,3] = _h # FIXME!!!! WTF, this is backwards too!!!
        
    def set_roi(self, cam_idx, roi):
        self.observations[cam_idx].set_roi(roi)
        
    def  set_ff_observation(self, cam_idx, roi):
        #print 'setting observation {} {}'.format(cam_idx, roi)
        self.ff_observations[cam_idx].set_roi(roi)

    def has_ff_observation(self, cam_idx): return self.ff_observations[cam_idx].roi is not None

    def is_in_frustum(self, cam):
        pts_img = cam.project(self.pts_world)
        in_frustum = np.all(pts_img > [0, 0]) and np.all(pts_img < [cam.w, cam.h])
        roi = cam.compute_roi(pts_img) if in_frustum else None
        return in_frustum , roi
    
    def tracking_succeeded(self): return True#self.cam_to_irm_T is not None

    def set_world_pose(self, world_to_irm_T):
        if world_to_irm_T is not None:
            self.is_localized = True
            self.world_to_irm_T = world_to_irm_T
            self.irm_to_world_T = np.linalg.inv(world_to_irm_T)
            self.world_to_body_T = np.dot(self.irm_to_body_T, self.world_to_irm_T)
            self.body_to_world_T = np.linalg.inv(self.world_to_body_T)
            self.pts_world = np.array([utils.transform(self.irm_to_world_T, pt_m) for pt_m in self.kps_m])
        else:
            self.is_localized = False
            self.world_to_irm_T = np.eye(4)

class Camera:
    def __init__(self, name, encoding='mono8'):
        self.name = name
        # camera matrix, distortion coefficients, inverted camera matrix
        self.K, self.D, self.invK = None, None, None
        # world to camera transform
        self.world_to_cam_T, self.world_to_cam_t, self.world_to_cam_q, self.world_to_cam_r = None, None, None, None
        self.cam_to_world_T = None
        # image encoding
        self.img_encoding = encoding
        
    def set_calibration(self, K, D, w, h):
        self.K, self.D, self.w, self.h = K, D, w, h
        self.invK = np.linalg.inv(self.K)

    def set_location(self,  world_to_camo_t, world_to_camo_q):
        self.world_to_cam_t, self.world_to_cam_q = world_to_camo_t, world_to_camo_q 
        self.world_to_cam_T = utils.T_of_t_q(world_to_camo_t, world_to_camo_q)
        self.world_to_cam_r, _unused = cv2.Rodrigues(self.world_to_cam_T[:3,:3])
        self.cam_to_world_T = np.linalg.inv(self.world_to_cam_T)
        # compute floor plan normal and distance
        # FIXME is world_to_cam really cam_to_world???
        # yes!!!
        self.fp_n = self.world_to_cam_T[:3,2]                      # image of [0 0 1]_world in cam frame
        self.fp_d = -np.dot(self.fp_n , self.world_to_cam_T[:3,3]) # 

        
    def has_calibration(self): return self.K is not None
    def is_localized(self): return self.world_to_cam_t is not None
        
    def project(self, points_world):
        return cv2.projectPoints(points_world, self.world_to_cam_r, self.world_to_cam_t, self.K, self.D)[0]

    def compute_roi(self, pts_img, margin=70):
        ''' Compute RegionOfInterest for a set_of_points in this camera '''
        x_lu, y_lu = np.min(pts_img, axis=0).squeeze().astype(int)
        x_rd, y_rd = np.max(pts_img, axis=0).squeeze().astype(int)
        roi = slice(max(0, y_lu-margin), min(self.w, y_rd+margin)), slice(max(0, x_lu-margin), min(self.w, x_rd+margin))
        return roi
    
    
class SMoCap:

    def __init__(self, cameras, undistort, detector_cfg_path):
        LOG.info(' undistort {}'.format( undistort ))
        LOG.info(' detector cfg path {}'.format( detector_cfg_path ))

        self.undistort = undistort
        self.shape_database = shapes.Database()

        self.detectors = [Detector(cam.img_encoding, detector_cfg_path) for cam in cameras]
        self.ff_detectors = [Detector(cam.img_encoding, detector_cfg_path) for cam in cameras]
        self.cameras = cameras

        nb_cams = len(cameras) 
        self.markers = [Marker(nb_cams) for i in range(len(self.shape_database.shapes))]
        self.marker = Marker(nb_cams)
        self._keypoints_detected = False

         

    def detect_markers_in_full_frame(self, img, cam_idx):
        '''
        Called by frame searcher thread
        '''
        keypoints, pts_img = self.ff_detectors[cam_idx].detect_ff(img)
        if len(pts_img) > 0:
            clusters_id = self.ff_detectors[cam_idx].cluster_keypoints(pts_img)
            clusters_nb = np.max(clusters_id) + 1
            clusters = [pts_img[clusters_id == i] for i in range(clusters_nb)]
            #print('in detect_markers_in_full_frame cam {} ( {} keypoints -> {} clusters )'.format(cam_idx, len(pts_img.squeeze()), clusters_nb))
            shapes = [smocap.shapes.Shape(c) for c in clusters]
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

        # old single marker code
        if len(keypoints) == 4: # marker is found, to be adapted...
            self.marker.set_ff_observation(cam_idx, self.cameras[cam_idx].compute_roi(pts_img))
        else:
            self.marker.set_ff_observation(cam_idx, None)
        #
        
        for idx_marker, marker in enumerate(self.markers):
            try:
                idx_shape = found_shapes_ids.index(idx_marker)
                marker.set_ff_observation(cam_idx, self.cameras[cam_idx].compute_roi(shapes[idx_shape].pts))
            except ValueError:
                marker.set_ff_observation(cam_idx, None)                


            

    def has_unlocalized_markers(self):
        return not self.marker.is_localized

    def detect_marker_in_roi(self, img, cam_idx):
        ''' called by video stream threads '''
        if self.marker.is_localized:
            #print 'marker is localized', cam_idx
            in_frustum, roi = self.marker.is_in_frustum(self.cameras[cam_idx])
            if not in_frustum:
                raise MarkerNotInFrustumException
            self.marker.set_roi(cam_idx, roi)
        elif self.marker.has_ff_observation(cam_idx):
            #print 'marker has ff obs', cam_idx
            self.marker.set_roi(cam_idx, self.marker.ff_observations[cam_idx].roi)
        else:
            raise MarkerNotLocalizedException
        # from here maker has roi
        #print('vid thread, marker has roi in {}'.format(cam_idx))
        #print self.marker.roi
        o = self.marker.observations[cam_idx]
        o.keypoints, o.kps_img = self.detectors[cam_idx].detect(img, self.marker.observations[cam_idx].roi)    
        if len(o.keypoints) != 4:
            raise MarkerNotDetectedException # FIXME...


    def identify_marker_in_roi(self, cam_idx):
        
        o = self.marker.observations[cam_idx]
        o.marker_of_kp = np.array([-2, -2, -2, -2])
        # first use distance to discriminate center and front
        dists = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                dists[i, j] = np.linalg.norm(o.kps_img[i]- o.kps_img[j])
        #print('dists {}'.format(dists))
        sum_dists = np.sum(dists, axis=1)
        #print('sum dists {}'.format(sum_dists))
        sorted_idx = np.argsort(sum_dists)
        #print sorted_idx
        o.marker_of_kp[sorted_idx[0]] = Marker.kp_id_c #self.marker_c
        o.marker_of_kp[sorted_idx[1]] = Marker.kp_id_f #self.marker_f
        # now use vector product to discriminate right and left
        cf = o.kps_img[sorted_idx[1]] - o.kps_img[sorted_idx[0]]
        c2 = o.kps_img[sorted_idx[2]] - o.kps_img[sorted_idx[0]]
        #c3 = self.detected_kp_img[sorted_idx[3]] - self.detected_kp_img[sorted_idx[0]]
        def vprod(a, b): return a[0]*b[1]-a[1]*b[0]
        s2 = vprod(cf, c2)
        #s3 = vprod(cf, c3)
        #print 's2 s3', s2, s3
        o.marker_of_kp[sorted_idx[2]] = Marker.kp_id_r if s2 > 0 else Marker.kp_id_l
        o.marker_of_kp[sorted_idx[3]] = Marker.kp_id_l if s2 > 0 else Marker.kp_id_r
        o.kp_of_marker = np.argsort(o.marker_of_kp)
        #print self.marker_of_kp
        # for now just use center point
        o.centroid_img = o.kps_img[o.kp_of_marker[Marker.kp_id_c]]
        return True

    def track_marker(self, cam_idx, verbose=0):
        ''' This is a tracker using bundle adjustment.
            Position and orientation are constrained to the floor plane '''
        obs = self.marker.observations[cam_idx]
        kps_img = obs.kps_img[obs.kp_of_marker]
        cam = self.cameras[cam_idx]

        def irm_to_cam_T_of_params(params):
            x, y, theta = params
            irm_to_world_r, irm_to_world_t = np.array([0., 0, theta]), [x, y, self.marker.heigth_above_floor]
            irm_to_world_T = utils.T_of_t_r(irm_to_world_t, irm_to_world_r)
            return np.dot(cam.world_to_cam_T, irm_to_world_T) 
            
        def residual(params):
            irm_to_cam_T = irm_to_cam_T_of_params(params)
            irm_to_cam_t, irm_to_cam_r = utils.tr_of_T(irm_to_cam_T)
            projected_kps = cv2.projectPoints(self.marker.kps_m, irm_to_cam_r, irm_to_cam_t, cam.K, cam.D)[0].squeeze()
            return (projected_kps - kps_img).ravel()

        def params_of_irm_to_world_T(irm_to_world_T):
            ''' return x,y,theta from irm_to_world transform '''
            _angle, _dir, _point = tf.transformations.rotation_from_matrix(irm_to_world_T)
            return (irm_to_world_T[0,3], irm_to_world_T[1,3], _angle)

        p0 =  params_of_irm_to_world_T(self.marker.irm_to_world_T if self.marker.is_localized else np.eye(4)) 
        res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf')
        #return irm_to_cam_T_of_params(res.x) 
        irm_to_cam_T = irm_to_cam_T_of_params(res.x)
        cam_to_irm_T = np.linalg.inv(irm_to_cam_T)
        world_to_irm_T = np.dot(cam_to_irm_T, cam.world_to_cam_T)
        self.marker.set_world_pose(world_to_irm_T)
        
    
   
    def keypoints_detected(self):
        return self._keypoints_detected

    def keypoints_identified(self):
        return True
     
    
        

    def draw_debug_on_image(self, img, camera_idx, draw_kp_id=True, draw_roi=True):
        if self.cameras[camera_idx].img_encoding == 'mono8':
            debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # make a copy of image, insuring it is a color one
        else:
            debug_img = img

        #cv2.drawKeypoints(img, self.keypoints, debug_img[self.marker.roi], (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.drawKeypoints(img, self.keypoints, debug_img, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if  draw_kp_id and self.keypoints_detected() and self.keypoints_identified():
            for i in range(len(self.keypoints)):
                cv2.putText(debug_img[self.marker.roi], Marker.kps_names[self.marker_of_kp[i]],
                            round_pt(self.keypoints[i].pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if draw_roi and self.marker.roi is not None:
            pt1 = (self.marker.roi[1].start, self.marker.roi[0].start)
            pt2 = (self.marker.roi[1].stop, self.marker.roi[0].stop) 
            cv2.rectangle(debug_img, pt1, pt2, (0, 0, 255), 2)

        

            
        return debug_img
