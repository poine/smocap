import logging, threading, numpy as np
import tf.transformations
import scipy.optimize
import cv2

LOG = logging.getLogger('smocap_single')

import smocap
import smocap.utils, smocap.shapes, smocap.detector

import pdb

#
#  This is the naive single marker version of smocap 
#




class SMocapMonoMarker:
    ''' Original naive version: tracks a single marker '''
    def __init__(self, cameras, detector_cfg_path, height_above_floor):
        LOG.info(' detector cfg path {}'.format( detector_cfg_path ))
        self.detectors = [smocap.detector.Detector(cam.img_encoding, detector_cfg_path) for cam in cameras]
        self.ff_detectors = [smocap.detector.Detector(cam.img_encoding, detector_cfg_path) for cam in cameras]
        self.cameras = cameras
        # we're just getting the coordinates of the 4 keypoints
        self.shape_database = smocap.shapes.Database()
        self.marker = smocap.Marker(len(cameras), self.shape_database.shapes[0], height_above_floor)
        self.optimize_lock = threading.Lock() # fuck....

    def detect_markers_in_full_frame(self, img, cam_idx, stamp):
        keypoints, pts_img = self.ff_detectors[cam_idx].detect_ff(img)
        if len(keypoints) == 4: # marker is found, to be adapted...
            roi = self.cameras[cam_idx].compute_roi(pts_img)
            self.marker.set_ff_observation(cam_idx, roi, None, None, keypoints)
        else:
            self.marker.set_ff_observation(cam_idx, None, None, None, None)
    

    def detect_marker_in_roi(self, img, cam_idx):
        ''' called by video stream threads '''
        # Finds the region of interest for the marker in this camera
        if self.marker.is_localized:
            in_frustum, roi = self.marker.is_in_frustum(self.cameras[cam_idx])
            if not in_frustum:
                raise smocap.MarkerNotInFrustumException
            self.marker.set_roi(cam_idx, roi)
        elif self.marker.has_ff_observation(cam_idx):
            self.marker.set_roi(cam_idx, self.marker.ff_observations[cam_idx].roi)
        else:
            raise smocap.MarkerNotLocalizedException
        # from here marker has roi
        o = self.marker.observations[cam_idx]
        o.keypoints, o.kps_img = self.detectors[cam_idx].detect(img, o.roi)
        
        if len(o.keypoints) == 4:
            o.tracked = True
        else:
            if o.tracked:
                o.tracked = False
                o.is_valid = False
                raise smocap.MarkerLostException
            else:
                raise smocap.MarkerNotDetectedException

    
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
        o.marker_of_kp[sorted_idx[0]] = smocap.Marker.kp_id_c #self.marker_c
        o.marker_of_kp[sorted_idx[1]] = smocap.Marker.kp_id_f #self.marker_f
        # now use vector product to discriminate right and left
        cf = o.kps_img[sorted_idx[1]] - o.kps_img[sorted_idx[0]]
        c2 = o.kps_img[sorted_idx[2]] - o.kps_img[sorted_idx[0]]
        #c3 = self.detected_kp_img[sorted_idx[3]] - self.detected_kp_img[sorted_idx[0]]
        def vprod(a, b): return a[0]*b[1]-a[1]*b[0]
        s2 = vprod(cf, c2)
        #s3 = vprod(cf, c3)
        #print 's2 s3', s2, s3
        o.marker_of_kp[sorted_idx[2]] = smocap.Marker.kp_id_r if s2 > 0 else smocap.Marker.kp_id_l
        o.marker_of_kp[sorted_idx[3]] = smocap.Marker.kp_id_l if s2 > 0 else smocap.Marker.kp_id_r
        #o.kp_of_marker = np.argsort(o.marker_of_kp)
        #o.keypoints_img_sorted = o.kps_img[o.kp_of_marker]
        kp_of_marker = np.argsort(o.marker_of_kp)
        o.keypoints_img_sorted = o.kps_img[kp_of_marker]
        #print self.marker_of_kp
        # for now just use center point
        o.centroid_img = o.keypoints_img_sorted[smocap.Marker.kp_id_c]
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
            irm_to_world_T = smocap.utils.T_of_t_r(irm_to_world_t, irm_to_world_r)
            return np.dot(cam.world_to_cam_T, irm_to_world_T)
            #return np.dot(cam.cam_to_world_T, irm_to_world_T) # wtf.... both work (or don't)
            
        def residual(params):
            irm_to_cam_T = irm_to_cam_T_of_params(params)
            irm_to_cam_t, irm_to_cam_r = smocap.utils.tr_of_T(irm_to_cam_T)
            projected_kps = cv2.projectPoints(self.marker.ref_shape._pts, irm_to_cam_r, irm_to_cam_t, cam.K, cam.D)[0].squeeze()
            return (projected_kps - observation.keypoints_img_sorted).ravel()

        def params_of_irm_to_world_T(irm_to_world_T):
            ''' return x,y,theta from irm_to_world transform '''
            _angle, _dir, _point = tf.transformations.rotation_from_matrix(irm_to_world_T)
            return (irm_to_world_T[0,3], irm_to_world_T[1,3], _angle)

        p0 =  params_of_irm_to_world_T(self.marker.irm_to_world_T if self.marker.is_localized else np.eye(4))
        self.optimize_lock.acquire()
        #res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf')
        res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='lm')
        self.optimize_lock.release()
        if verbose:
            print('start  {}'.format(p0))
            print('success  {}'.format(res.success))
            print('solution {}'.format(res.x))
            print('residual {}'.format(res.cost))
        if res.success:
            observation.rep_err = res.cost
            observation.irm_to_cam_T = irm_to_cam_T_of_params(res.x)
            #_angle, _dir, _point = tf.transformations.rotation_from_matrix(observation.irm_to_cam_T)
            #observation.orientation = _angle
            observation.is_valid = True
        else:
            observation.is_valid = False
            raise smocap.MarkerNotLocalizedException

    def localize_marker_in_world(self, cam_idx):
        observation, cam = self.marker.observations[cam_idx], self.cameras[cam_idx]

        valid_obs = []
        for o in self.marker.observations:
            if o.is_valid: valid_obs.append(o)
        if len(valid_obs) == 0:
            self.marker.set_world_pose(None)
            return
        
        rep_errs = [o.rep_err for o in valid_obs]
        best_obs = np.argmin(rep_errs)
        
        if observation == valid_obs[best_obs]:
            #print('wolrd loc with cam {} ({})'.format(cam_idx, rep_errs))
            #pdb.set_trace()
            cam_to_irm_T = np.linalg.inv(observation.irm_to_cam_T)
            world_to_irm_T = np.dot(cam_to_irm_T, cam.world_to_cam_T)
            self.marker.set_world_pose(world_to_irm_T)

  
        
    def draw(self, img, cam):
        if self.marker.has_ff_observation(cam._id):
            o = self.marker.get_ff_observation(cam._id)
            #pdb.set_trace()
            cv2.drawKeypoints(img, o.keypoints, img, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            pt1 = (o.roi[1].start, o.roi[0].start)
            pt2 = (o.roi[1].stop, o.roi[0].stop) 
            cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
            # ff_observations does not have centroid nor orientation
        if 0: 
            o = self.marker.observations[cam._id]
            pt1 = o.centroid_img
            pt2 = pt1 + 20*np.array([math.cos(o.orientation), math.sin(o.orientation)])
            cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 255,255), thickness=2)
