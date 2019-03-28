#!/usr/bin/env python

import logging, os, timeit, threading#, yaml
import math, numpy as np, cv2, tf.transformations
import scipy.optimize
#import scipy.spatial.distance
#import sklearn.cluster

#import utils
import pdb

import smocap.utils
import smocap.shapes
from smocap.camera_system import *
from smocap.camera import *
from smocap.detector import *
#import smocap.camera

LOG = logging.getLogger('smocap')

def round_pt(p): return (int(p[0]), int(p[1]))

class MarkerLostException(Exception):
    pass

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
        self.is_valid = False
        
    def set_roi(self, roi):
        self.roi = roi

    def set_centroid(self, centroid):
        self.centroid_img = centroid

    def set_orientation(self, orientation):
        self.orientation = orientation

    def set_keypoints(self, keypoints):
        self.keypoints = keypoints
        
        
class Marker:
    # keypoints index
    kp_id_c, kp_id_l, kp_id_r, kp_id_f = range(4)
    # keypoints names
    kps_names = ['c', 'l', 'r', 'f']
    
    def __init__(self, nb_cams, ref_shape, heigth_above_floor):

        self.ref_shape = ref_shape
        # Constants:
        #   coordinates of keypoints in marker frame
        #self.kps_m = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
        #   constrain marker to floor plan 
        self.irm_to_body_T = np.eye(4)
        self.set_height_ablove_floor(heigth_above_floor)
        
        # Marker and body poses
        self.set_world_pose(None)

        self.ff_observations = [Observation() for i in range(nb_cams)]
        self.observations = [Observation() for i in range(nb_cams)]
        

    def set_height_ablove_floor(self, _h):
        self.heigth_above_floor = _h
        self.irm_to_body_T[2,3] = _h # not used anymore ??

        
    def set_roi(self, cam_idx, roi):
        self.observations[cam_idx].set_roi(roi)
        
    def  set_ff_observation(self, cam_idx, roi, centroid, orientation, keypoints):
        #print 'setting observation {} {}'.format(cam_idx, roi)
        self.ff_observations[cam_idx].set_roi(roi)
        self.ff_observations[cam_idx].set_centroid(centroid)
        self.ff_observations[cam_idx].set_orientation(orientation)
        self.ff_observations[cam_idx].set_keypoints(keypoints)
        

    def has_ff_observation(self, cam_idx): return self.ff_observations[cam_idx].roi is not None
    def get_ff_observation(self, cam_idx): return self.ff_observations[cam_idx]
    
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
            self.irm_to_world_T = np.eye(4)

        


