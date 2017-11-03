#!/usr/bin/env python

import logging, os, timeit
import math, numpy as np, cv2, tf.transformations
import scipy.optimize

import utils
import pdb

LOG = logging.getLogger('smocap')

def round_pt(p): return (int(p[0]), int(p[1]))


class Marker:
    # keypoints index
    kp_id_c, kp_id_l, kp_id_r, kp_id_f = range(4)
    # keypoints names
    kps_names = ['c', 'l', 'r', 'f']
    
    def __init__(self):
        self.roi = None                # region of interest in wich to look for the marker
        self.detection_duration = 0.   # duration of blob detection
        self.centroid_img = None       # coordinates of marker centroid in image frame
        # coordinates of keypoints in marker frame
        self.kps_m = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])

        # constant
        self.irm_to_body_T = np.eye(4); self.irm_to_body_T[2,3] = 0.09 # FIXME!!!! WTF, this is backwards too!!!
        self.heigth_above_floor = 0
        
        # marker and body poses
        self.cam_to_irm_T = np.eye(4)
        self.irm_to_world_T = np.eye(4)
        self.cam_to_body_T, self.world_to_body_T = None, None
        self.body_to_world_T = np.eye(4)
        # timing statistics
        self.detection_duration = 0.
        self.tracking_duration = 0.
        
    def set_roi(self, x_lu, y_lu, x_rd, y_rd):
        self.roi = slice(y_lu, y_rd), slice(x_lu, x_rd)

    def tracking_succeeded(self): return self.cam_to_irm_T is not None
        

class Camera:
    def __init__(self, name):
        self.name = name
        # camera matrix, distortion coefficients, inverted camera matrix
        self.K, self.D, self.invK = None, None, None
        # world to camera transform
        self.world_to_cam_T, self.world_to_cam_t, self.world_to_cam_q, self.world_to_cam_r = None, None, None, None
        self.cam_to_world_T = None
        
    def set_calibration(self, K, D, w, h):
        self.K, self.D, self.w, self.h = K, D, w, h
        self.invK = np.linalg.inv(self.K)

    def set_location(self,  world_to_camo_t, world_to_camo_q):
        self.world_to_cam_t, self.world_to_cam_q = world_to_camo_t, world_to_camo_q 
        self.world_to_cam_T = utils.T_of_t_q(world_to_camo_t, world_to_camo_q)
        self.world_to_cam_r, _unused = cv2.Rodrigues(self.world_to_cam_T[:3,:3])
        self.cam_to_world_T = np.linalg.inv(self.world_to_cam_T)

    def has_calibration(self): return self.K is not None
    def is_localized(self): return self.world_to_cam_t is not None
        
        
class SMoCap:

    def __init__(self, detect_rgb, undistort, min_area=2):
        LOG.info(' detect rgb {}'.format( detect_rgb ))
        LOG.info(' undistort {}'.format( undistort ))
        LOG.info(' min area {}'.format( min_area ))
        self.undistort = undistort
        self.detect_rgb = detect_rgb
        self.lower_red_hue_range = np.array([0,  100,100]), np.array([10,255,255]) 
        self.upper_red_hue_range = np.array([160,100,100]), np.array([179,255,255])
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 2;
        params.maxThreshold = 256;
        # Filter by Area.
        params.filterByArea = True
        params.minArea = min_area#16 #2
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        self.detector = cv2.SimpleBlobDetector_create(params)

        self.marker = Marker()
        self.cameras = []        
        self.floor_planes = []
        self._keypoints_detected = False

        self.tracking_method = self.track_in_plane

    def add_camera(self, cam):
         LOG.info('adding camera: {}'.format(cam.name))
         self.cameras.append(cam)
         # FIXME is world_to_cam really cam_to_world???
         # yes!!!
         fp_n = cam.world_to_cam_T[:3,2]                    # image of [0 0 1]_world in cam frame
         fp_d = -np.dot(fp_n ,cam.world_to_cam_T[:3,3])     # 
         fp = {'n': fp_n, 'd': fp_d}
         self.floor_planes.append(fp)
         

    def detect_keypoints(self, img):
        # if no Region Of Interest exists yet, set it to full image
        if self.marker.roi is None:
            self.marker.set_roi(0, 0, *img.shape[1::-1])
        
        _start = timeit.default_timer()
        if self.detect_rgb:
            hsv = cv2.cvtColor(img[self.marker.roi], cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, *self.lower_red_hue_range)
            #mask2 = cv2.inRange(hsv, *self.upper_red_hue_range)
            self.keypoints = self.detector.detect(255-mask1)
        else:
            self.keypoints = self.detector.detect(255-img[self.marker.roi])

        # compute keypoints coordinates in image frame
        self.detected_kp_img = np.array([kp.pt for kp in self.keypoints])
        if len(self.detected_kp_img) > 0:
            self.detected_kp_img += [self.marker.roi[1].start, self.marker.roi[0].start]
        _end = timeit.default_timer()
        # assume our detection was sucessfull is 4 keypoints were detected
        self._keypoints_detected = (len(self.detected_kp_img) == 4)

        # if we did not detect 4 keypoints, reset ROI to full image
        # that sucks as we can not draw anymore :(
        if not self._keypoints_detected:
           self.marker.set_roi(0, 0, *img.shape[1::-1])

        self.marker.detection_duration = _end - _start
        return self.keypoints

    def keypoints_detected(self):
        return self._keypoints_detected

    def keypoints_identified(self):
        return True
     
    
    def identify_keypoints(self):
        ''' naive identification of markers '''
        if len(self.detected_kp_img) != 4: return
        self.marker_of_kp = np.array([-2, -2, -2, -2])
        # first use distance to discriminate center and front
        dists = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                dists[i, j] = np.linalg.norm(self.detected_kp_img[i]- self.detected_kp_img[j])
        #print('dists {}'.format(dists))
        sum_dists = np.sum(dists, axis=1)
        #print('sum dists {}'.format(sum_dists))
        sorted_idx = np.argsort(sum_dists)
        #print sorted_idx
        self.marker_of_kp[sorted_idx[0]] = Marker.kp_id_c #self.marker_c
        self.marker_of_kp[sorted_idx[1]] = Marker.kp_id_f #self.marker_f
        # now use vector product to discriminate right and left
        cf = self.detected_kp_img[sorted_idx[1]] - self.detected_kp_img[sorted_idx[0]]
        c2 = self.detected_kp_img[sorted_idx[2]] - self.detected_kp_img[sorted_idx[0]]
        #c3 = self.detected_kp_img[sorted_idx[3]] - self.detected_kp_img[sorted_idx[0]]
        def vprod(a, b): return a[0]*b[1]-a[1]*b[0]
        s2 = vprod(cf, c2)
        #s3 = vprod(cf, c3)
        #print 's2 s3', s2, s3
        if s2 > 0:
            self.marker_of_kp[sorted_idx[2]] = Marker.kp_id_r
            self.marker_of_kp[sorted_idx[3]] = Marker.kp_id_l
        else:
            self.marker_of_kp[sorted_idx[2]] = Marker.kp_id_l
            self.marker_of_kp[sorted_idx[3]] = Marker.kp_id_r
        self.kp_of_marker = np.argsort(self.marker_of_kp)
        #print self.marker_of_kp
        # for now just use center point
        self.marker.centroid_img = self.detected_kp_img[self.kp_of_marker[Marker.kp_id_c]]
        w = 75
        x1, x2 = int(max(self.marker.centroid_img[0]-w, 0)), int(min(self.marker.centroid_img[0]+w, self.cameras[0].w))
        y1, y2 = int(max(self.marker.centroid_img[1]-w, 0)), int(min(self.marker.centroid_img[1]+w, self.cameras[0].h))
        self.marker.roi = slice(y1, y2), slice(x1, x2)

        
    def track_foo(self):
        h, status = cv2.findHomography(self.detected_kp_img[self.kp_of_marker], self.markers_body[:,:2])
        #h, status = cv2.findHomography(self.markers_body[:,:2], self.detected_kp_img[self.kp_of_marker])
        #h, status = cv2.findHomography(self.detected_kp_img[self.markers_of_kp], self.markers_body[:,:2])
        invC = np.linalg.inv(self.K)
        h1, h2, h3 = h[:,0], h[:,1], h[:,2] 
        _lambda = 1./np.linalg.norm(np.dot(invC, h1))
        #_lambda = 1./np.linalg.norm(np.dot(invC, h2))
        r1 = _lambda*np.dot(invC, h1)
        r2 = _lambda*np.dot(invC, h2)
        r3 = np.cross(r1, r2)
        t = _lambda*np.dot(invC, h3)
        #pdb.set_trace()

    def track_pnp(self, debug=False):
        cam_to_irm_t, cam_to_irm_r = utils.tr_of_T(self.marker.cam_to_irm_T)
        kps_img = self.detected_kp_img[self.kp_of_marker].reshape(4,1,2)

        # projectPoints and solvePnP use irm_to_cam.. .maybe not :(
        if debug:
            print(' current cam_to_irm: t {} r {}'.format(cam_to_irm_t, cam_to_irm_r))
            pm = cv2.projectPoints(self.marker.kps_m, cam_to_irm_r, cam_to_irm_t, self.cameras[0].K, self.cameras[0].D)[0].squeeze()
            print(' projected markers (current cam_to_irm_r)\n{}'.format(pm))
            #print('markers\n{}'.format(self.marker.kps_m))
            print(' detected markers\n{}'.format(kps_img.squeeze()))
            rep_err = np.mean(np.linalg.norm(pm - kps_img.squeeze(), axis=1))
            print('reprojection error {:.2f} px'.format(rep_err))
        # success, r_vec_, t_vec_ = cv2.solvePnP(self.marker.kps_m, kps_img, self.camera.K, self.camera.D, flags=cv2.SOLVEPNP_P3P) # doesn't work...
        # success, r_vec_, t_vec_ = cv2.solvePnP(self.marker.kps_m, kps_img, self.camera.K, self.camera.D, flags=cv2.SOLVEPNP_EPNP) # doesn't work
        success, irm_to_cam_r, irm_to_cam_t = cv2.solvePnP(self.marker.kps_m, kps_img, self.cameras[0].K, self.cameras[0].D,
                                                           np.array([cam_to_irm_r]), np.array([cam_to_irm_t]),
                                                           useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        #success, r_vec_, t_vec_ = cv2.solvePnP(self.marker.kps_m, kps_img, self.camera.K, self.camera.D, flags=cv2.SOLVEPNP_DLS) # doesn't work...
        if debug:
            print('PnP computed irm_to_cam t {} r {} (in {:.2e}s) '.format(irm_to_cam_t.squeeze(), irm_to_cam_r.squeeze()))
            pm2 = cv2.projectPoints(self.marker.kps_m, irm_to_cam_r, irm_to_cam_t, self.cameras[0].K, self.cameras[0].D)[0].squeeze()
            print(' projected markers (new irm_to_cam)\n{}'.format(pm2))
            rep_err = np.mean(np.linalg.norm(pm2 - kps_img.squeeze(), axis=1))
            print('reprojection error {:.2f} px'.format(rep_err))

        irm_to_cam_T = utils.T_of_t_r(irm_to_cam_t.squeeze(), irm_to_cam_r) 
        self.marker.cam_to_irm_T = irm_to_cam_T#np.linalg.inv(irm_to_cam_T)
       #pdb.set_trace()



    def track_in_plane(self, verbose=0):
        ''' This is a tracker using bundle adjustment.
            Position and orientation are constrained to the floor plane '''
        kps_img = self.detected_kp_img[self.kp_of_marker]

        def irm_to_cam_T_of_params(params):
            x, y, theta = params
            irm_to_world_r, irm_to_world_t = np.array([0., 0, theta]), [x, y, self.marker.heigth_above_floor]
            irm_to_world_T = utils.T_of_t_r(irm_to_world_t, irm_to_world_r)
            return np.dot(self.cameras[0].world_to_cam_T, irm_to_world_T) 
            
        def residual(params):
            irm_to_cam_T = irm_to_cam_T_of_params(params)
            irm_to_cam_t, irm_to_cam_r = utils.tr_of_T(irm_to_cam_T)
            projected_kps = cv2.projectPoints(self.marker.kps_m, irm_to_cam_r, irm_to_cam_t, self.cameras[0].K, self.cameras[0].D)[0].squeeze()
            return (projected_kps - kps_img).ravel()

        def params_of_irm_to_cam_T(irm_to_cam_T):
            ''' return x,y,theta of irm_to_world transform '''
            irm_to_world_T = np.dot(self.cameras[0].cam_to_world_T, irm_to_cam_T) 
            _angle, _dir, _point = tf.transformations.rotation_from_matrix(irm_to_world_T)
            #pdb.set_trace()
            return (irm_to_world_T[0,3], irm_to_world_T[1,3], _angle)

        p0 =  params_of_irm_to_cam_T(self.marker.cam_to_irm_T)
        res = scipy.optimize.least_squares(residual, p0, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf')
        self.marker.irm_to_cam_T = irm_to_cam_T_of_params(res.x) ## WTF!!!
        self.marker.cam_to_irm_T = np.linalg.inv(self.marker.irm_to_cam_T)
        

        
    def track_dumb(self):
        # this is a dumb tracking to use as baseline
        m_c_i = self.detected_kp_img[self.kp_of_marker[Marker.kp_id_c]]
        m_f_i = self.detected_kp_img[self.kp_of_marker[Marker.kp_id_f]]

        if self.undistort:
            distorted_points = np.array([m_c_i, m_f_i])
            m_c_i_rect, m_f_i_rect = cv2.undistortPoints(distorted_points.reshape((2,1,2)), self.cameras[0].K, self.cameras[0].D, P=self.cameras[0].K)
            m_c_i, m_f_i =  m_c_i_rect.squeeze(), m_f_i_rect.squeeze()
          
        cf_i = m_f_i - m_c_i
        yaw = math.atan2(cf_i[1], cf_i[0])
        self.marker.cam_to_irm_T = tf.transformations.euler_matrix(math.pi, 0, yaw, 'sxyz')
        m_c_c = np.dot(self.cameras[0].invK, utils.to_homo(m_c_i))
        self.marker.cam_to_irm_T[:3,3] = m_c_c*(self.cameras[0].cam_to_world_t[2]-0.15)
        


    def track(self):
        _start = timeit.default_timer()
        self.tracking_method()
        _end = timeit.default_timer()
        self.marker.tracking_duration = _end - _start
        self.marker.world_to_irm_T = np.dot(self.marker.cam_to_irm_T, self.cameras[0].world_to_cam_T)
        self.marker.irm_to_world_T = np.linalg.inv(self.marker.world_to_irm_T)
        self.marker.cam_to_body_T = np.dot(self.marker.irm_to_body_T, self.marker.cam_to_irm_T)
        self.marker.world_to_body_T = np.dot(self.marker.cam_to_body_T, self.cameras[0].world_to_cam_T)
        self.marker.body_to_world_T = np.linalg.inv(self.marker.world_to_body_T)
            

    def project(self, body_to_world_T):
        self.markers_world = np.array([np.dot(body_to_world_T, m_b) for m_b in self.markers_body])
        self.projected_markers_img = cv2.projectPoints(np.array([self.markers_world[:,:3]]),
                                                       self.cameras[0].world_to_cam_r, np.array(self.cameras[0].world_to_cam_t), self.K, self.D)[0].squeeze()    
        



    def draw_debug_on_image(self, img, draw_kp_id=True, draw_roi=True):
        if self.detect_rgb:
            debug_img = img
        else:
            debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # make a copy of image, insuring it is a color one

        cv2.drawKeypoints(img, self.keypoints, debug_img[self.marker.roi], (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if  draw_kp_id and self.keypoints_detected() and self.keypoints_identified():
            for i in range(len(self.keypoints)):
                cv2.putText(debug_img[self.marker.roi], Marker.kps_names[self.marker_of_kp[i]],
                            round_pt(self.keypoints[i].pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if draw_roi and self.marker.roi is not None:
            pt1 = (self.marker.roi[1].start, self.marker.roi[0].start)
            pt2 = (self.marker.roi[1].stop, self.marker.roi[0].stop) 
            cv2.rectangle(debug_img, pt1, pt2, (0, 0, 255), 2)

        

            
        return debug_img
