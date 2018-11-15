#!/usr/bin/env python

import logging, os, timeit#, yaml
import math, numpy as np, cv2, tf.transformations
import scipy.optimize
#import scipy.spatial.distance
#import sklearn.cluster

import utils
import pdb

import smocap.shapes
from smocap.camera_system import *
from smocap.camera import *
from smocap.detector import *
#import smocap.camera

LOG = logging.getLogger('smocap')

def round_pt(p): return (int(p[0]), int(p[1]))

class MarkerNotDetectedException(Exception):
    pass

class MarkerNotInFrustumException(Exception):
    pass

class MarkerNotLocalizedException(Exception):
    pass
        
            

class Observation:
    def __init__(self):
        self.roi = None                # region of interest in wich to look for the marker
        self.centroid_img = None       # coordinates of marker centroid in image frame
        self.orientation = None        # orientation of the shape
        self.tracked = False
        self.kps_img = None
        
    def set_roi(self, roi):
        self.roi = roi

    def set_centroid(self, centroid):
        self.centroid = centroid

    def set_orientation(self, orientation):
        self.orientation = orientation

        
class Marker:
    # keypoints index
    kp_id_c, kp_id_l, kp_id_r, kp_id_f = range(4)
    # keypoints names
    kps_names = ['c', 'l', 'r', 'f']
    
    def __init__(self, nb_cams, ref_shape):

        self.ref_shape = ref_shape
        # Constants:
        #   coordinates of keypoints in marker frame
        #self.kps_m = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
        #   constrain marker to floor plan 
        self.heigth_above_floor = 0
        self.irm_to_body_T = np.eye(4)
        self.irm_to_world_T =  np.eye(4)
        
        # Marker and body poses
        self.set_world_pose(None)
        # timing statistics
        self.detection_duration = 0. # duration of blob detection
        self.tracking_duration = 0.

        self.ff_observations = [Observation() for i in range(nb_cams)]
        self.observations = [Observation() for i in range(nb_cams)]
        

    def set_height_ablove_floor(self, _h):
        self.heigth_above_floor = _h
        if 1:
            self.irm_to_body_T[2,3] = _h # FIXME!!!! WTF, this is backwards too!!!
        else:
            self.irm_to_body_T[2,3] = -_h # FIXME!!!! WTF, this is backwards too!!!
        
    def set_roi(self, cam_idx, roi):
        self.observations[cam_idx].set_roi(roi)
        
    def  set_ff_observation(self, cam_idx, roi, centroid, orientation):
        #print 'setting observation {} {}'.format(cam_idx, roi)
        self.ff_observations[cam_idx].set_roi(roi)
        self.ff_observations[cam_idx].set_centroid(centroid)
        self.ff_observations[cam_idx].set_orientation(orientation)
        

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
            self.pts_world = np.array([utils.transform(self.irm_to_world_T, pt_m) for pt_m in self.ref_shape._pts])
        else:
            self.is_localized = False
            self.world_to_irm_T = np.eye(4)


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
        self.markers = [Marker(nb_cams, rs) for rs in self.shape_database.shapes]
        self.marker = Marker(nb_cams, self.shape_database.shapes[0])
        self._keypoints_detected = False

         

    def detect_markers_in_full_frame(self, img, cam_idx, stamp):
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
            self.marker.set_ff_observation(cam_idx, self.cameras[cam_idx].compute_roi(pts_img), None, None)
        else:
            self.marker.set_ff_observation(cam_idx, None, None, None)
        #
        
        for idx_marker, marker in enumerate(self.markers):
            try:
                idx_shape = found_shapes_ids.index(idx_marker)
                roi = self.cameras[cam_idx].compute_roi(shapes[idx_shape].pts, margin=70)
                marker.set_ff_observation(cam_idx, roi, shapes[idx_shape].Xc, shapes[idx_shape].theta)
            except ValueError: # idx_marker was not in found_shapes_ids
                marker.set_ff_observation(cam_idx, None, None, None)                


    def has_unlocalized_marker(self):
        return not self.marker.is_localized


    def detect_marker_in_roi2(self, img, cam_idx, stamp, marker_id):
        marker = self.markers[marker_id]
        obs = marker.observations[cam_idx]
        if marker.is_localized:
            # use previous position (or predicted...) to find roi
            in_frustum, roi = marker.is_in_frustum(self.cameras[cam_idx])
            if not in_frustum:
                raise MarkerNotInFrustumException
            marker.set_roi(cam_idx, roi)
            #print 'localized'
        elif marker.has_ff_observation(cam_idx):
            # use full frame detection - might be old...
            #print 'ff localized'
            marker.set_roi(cam_idx, marker.ff_observations[cam_idx].roi)
        else:
            raise MarkerNotLocalizedException    

        #print marker_id, obs.roi

        obs.keypoints, obs.kps_img = self.detectors[cam_idx].detect(img, obs.roi)
        if len(obs.kps_img) == 0:
            raise MarkerNotDetectedException
        #print obs.kps_img
        #print('in detect_marker_in_roi2 {}'.format(obs.kps_img))
        shape = smocap.shapes.Shape(obs.kps_img)
        shape_id, shape_ref = self.shape_database.find(shape)
        if shape_id == -1 or shape_ref != marker.ref_shape:
            if shape_ref != marker.ref_shape: print 'detected wrong shape'
            raise MarkerNotDetectedException
        shape.sort_points(debug=False, sort_cw=True)
        #print('in detect_marker_in_roi2 found shape {}'.format(shape_id))
        #print shape.pts[shape.angle_sort_idx]
        #print shape_ref.pts[shape.angle_sort_idx]
        obs.shape = shape # store that for now to allow debug
        obs.keypoints_img_sorted = shape.pts_sorted
        
    def track_marker2(self, marker, cam_idx, verbose=0, add_bug=False):
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
            irm_to_world_T = utils.T_of_t_r(irm_to_world_t, irm_to_world_r)
            return np.dot(cam.world_to_cam_T, irm_to_world_T) 
            
        def residual(params):
            irm_to_cam_T = irm_to_cam_T_of_params(params)
            irm_to_cam_t, irm_to_cam_r = utils.tr_of_T(irm_to_cam_T)
            projected_kps = cv2.projectPoints(pts_marker, irm_to_cam_r, irm_to_cam_t, cam.K, cam.D)[0].squeeze()
            return (projected_kps - observation.keypoints_img_sorted).ravel()

        def params_of_irm_to_world_T(irm_to_world_T):
            ''' return x,y,theta from irm_to_world transform '''
            _angle, _dir, _point = tf.transformations.rotation_from_matrix(irm_to_world_T)
            return (irm_to_world_T[0,3], irm_to_world_T[1,3], _angle)

        p0 =  params_of_irm_to_world_T(marker.irm_to_world_T if marker.is_localized else np.eye(4)) 
        res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf')
        #res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='lm')
        irm_to_cam_T = irm_to_cam_T_of_params(res.x)
        #print('in track_marker2 irm_to_cam_T\n{}'.format(irm_to_cam_T))
        cam_to_irm_T = np.linalg.inv(irm_to_cam_T)
        world_to_irm_T = np.dot(cam_to_irm_T, cam.world_to_cam_T)
        marker.set_world_pose(world_to_irm_T)
        marker.rep_err = res.cost


        
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
        o.keypoints, o.kps_img = self.detectors[cam_idx].detect(img, o.roi)
        #print('in detect_marker_in_roi {}'.format(o.kps_img))
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
        #o.kp_of_marker = np.argsort(o.marker_of_kp)
        #o.keypoints_img_sorted = o.kps_img[o.kp_of_marker]
        kp_of_marker = np.argsort(o.marker_of_kp)
        o.keypoints_img_sorted = o.kps_img[kp_of_marker]
        #print self.marker_of_kp
        # for now just use center point
        o.centroid_img = o.keypoints_img_sorted[Marker.kp_id_c]
        return True

    def track_marker(self, cam_idx, verbose=0):
        ''' This is a tracker using bundle adjustment.
            Position and orientation are constrained to the floor plane '''
        observation = self.marker.observations[cam_idx]
        cam = self.cameras[cam_idx]
        #print('in track_marker kps_img\n{}'.format(observation.keypoints_img_sorted))
        #print('in track_marker kps_marker\n{}'.format(self.marker.ref_shape._pts))
        
        def irm_to_cam_T_of_params(params):
            x, y, theta = params
            irm_to_world_r, irm_to_world_t = np.array([0., 0, theta]), [x, y, self.marker.heigth_above_floor]
            irm_to_world_T = utils.T_of_t_r(irm_to_world_t, irm_to_world_r)
            return np.dot(cam.world_to_cam_T, irm_to_world_T) 
            
        def residual(params):
            irm_to_cam_T = irm_to_cam_T_of_params(params)
            irm_to_cam_t, irm_to_cam_r = utils.tr_of_T(irm_to_cam_T)
            projected_kps = cv2.projectPoints(self.marker.ref_shape._pts, irm_to_cam_r, irm_to_cam_t, cam.K, cam.D)[0].squeeze()
            return (projected_kps - observation.keypoints_img_sorted).ravel()

        def params_of_irm_to_world_T(irm_to_world_T):
            ''' return x,y,theta from irm_to_world transform '''
            _angle, _dir, _point = tf.transformations.rotation_from_matrix(irm_to_world_T)
            return (irm_to_world_T[0,3], irm_to_world_T[1,3], _angle)

        p0 =  params_of_irm_to_world_T(self.marker.irm_to_world_T if self.marker.is_localized else np.eye(4)) 
        #res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf')
        res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='lm')
        #return irm_to_cam_T_of_params(res.x) 
        irm_to_cam_T = irm_to_cam_T_of_params(res.x)
        self.marker.irm_to_cam_T = irm_to_cam_T
        #print('in track_marker irm_to_cam_T\n{}'.format(irm_to_cam_T))
        cam_to_irm_T = np.linalg.inv(irm_to_cam_T)
        world_to_irm_T = np.dot(cam_to_irm_T, cam.world_to_cam_T)
        self.marker.set_world_pose(world_to_irm_T)
        self.marker.rep_err = res.cost
        
    
   
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
