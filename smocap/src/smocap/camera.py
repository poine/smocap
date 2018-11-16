import numpy as np, cv2
import pdb
import smocap.utils

class Camera:
    def __init__(self, _id, name, encoding='mono8'):
        self._id, self.name = _id, name
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
        self.world_to_cam_T = smocap.utils.T_of_t_q(world_to_camo_t, world_to_camo_q)
        self.world_to_cam_r, _unused = cv2.Rodrigues(self.world_to_cam_T[:3,:3])
        self.cam_to_world_T = np.linalg.inv(self.world_to_cam_T)
        # compute floor plan normal and distance
        # FIXME is world_to_cam really cam_to_world???
        # yes!!!
        if 1:
            self.fp_n = self.world_to_cam_T[:3,2]                      # image of [0 0 1]_world in cam frame
            self.fp_d = -np.dot(self.fp_n , self.world_to_cam_T[:3,3]) #
        else:
            self.fp_n = self.cam_to_world_T[:3,2]                      # image of [0 0 1]_world in cam frame
            self.fp_d = -np.dot(self.fp_n , self.cam_to_world_T[:3,3]) #

        
    def project(self, points_world):
        return cv2.projectPoints(points_world, self.world_to_cam_r, self.world_to_cam_t, self.K, self.D)[0]

    def compute_roi(self, pts_img, margin=70):
        ''' Compute RegionOfInterest for a set_of_points in this camera '''
        x_lu, y_lu = np.min(pts_img, axis=0).squeeze().astype(int)
        x_rd, y_rd = np.max(pts_img, axis=0).squeeze().astype(int)
        roi = slice(max(0, y_lu-margin), min(self.w, y_rd+margin)), slice(max(0, x_lu-margin), min(self.w, x_rd+margin))
        return roi

    def is_localized(self): return self.world_to_cam_t is not None
    
    
