import numpy as np, cv2
import yaml
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


    def set_encoding(self, encoding):
        self.img_encoding = encoding

            
    def project(self, points_world):
        return cv2.projectPoints(points_world, self.world_to_cam_r, self.world_to_cam_t, self.K, self.D)[0]

    def compute_roi(self, pts_img, margin=70):
        ''' Compute RegionOfInterest for a set_of_points in this camera '''
        x_lu, y_lu = np.min(pts_img, axis=0).squeeze().astype(int)
        x_rd, y_rd = np.max(pts_img, axis=0).squeeze().astype(int)
        roi = slice(max(0, y_lu-margin), min(self.w, y_rd+margin)), slice(max(0, x_lu-margin), min(self.w, x_rd+margin))
        return roi

    def is_localized(self): return self.world_to_cam_t is not None

    def load_intrinsics(self, filename):
        self.intrinsics_filename = filename
        camera_matrix, dist_coeffs, w, h = load_intrinsics(filename)
        self.set_calibration(camera_matrix, dist_coeffs, w, h)
    
    def load_all(self, kwargs):
        print kwargs
        self.load_intrinsics(kwargs['intrinsics'])
        self.set_encoding(kwargs['encoding'])
        t_world_to_camo_t = np.array(kwargs['world_to_camo_t'])
        t_world_to_camo_q = np.array(kwargs['world_to_camo_q'])
        self.set_location(t_world_to_camo_t, t_world_to_camo_q)
        #pdb.set_trace()


    def to_yaml(self):
        txt = '''intrinsics: {}
  encoding: {}
  world_to_camo_t: {}
  world_to_camo_q: {}'''.format(self.intrinsics_filename, self.img_encoding, self.world_to_cam_t.tolist(), self.world_to_cam_q.tolist())
        return txt
        
### Utils

# Load camera model
# I should use something from camera_calibration_parsers
def load_intrinsics(filename, verbose=False):
    with open(filename) as f:
        _dict = yaml.load(f)
        camera_matrix = np.array(_dict.get('camera_matrix')['data']).reshape(3, 3)
        dist_coeffs = np.array(_dict['distortion_coefficients']['data'])
        w, h = _dict['image_width'], _dict['image_height']
        if verbose:
            print('loading camera calibration from {}'.format(filename))
            print(' camera_matrix\n{}'.format(camera_matrix))
            print(' distortion\n{}'.format(dist_coeffs))
    return camera_matrix, dist_coeffs, w, h

# stolen from /opt/ros/kinetic/lib/python2.7/dist-packages/camera_calibration/calibrator.py
def write_extrinsics(filename, cam_info_msg, cname='unknown'):
    #print cam_info_msg
    txt = (""
           + "image_width: " + str(cam_info_msg.width) + "\n"
           + "image_height: " + str(cam_info_msg.height) + "\n"
           + "camera_name: " + cname + "\n"
           + "camera_matrix:\n"
           + "  rows: 3\n"
           + "  cols: 3\n"
           + "  data: [" + ", ".join(["{:.12f}".format(i) for i in  np.array(cam_info_msg.K).reshape(1,9)[0]]) + "]\n"
           + "distortion_model: " + ("rational_polynomial" if len(cam_info_msg.D) > 5 else "plumb_bob") + "\n"
           + "distortion_coefficients:\n"
           + "  rows: 1\n"
           + "  cols: 5\n"
           + "  data: [" + ", ".join(["%8f" % cam_info_msg.D[i] for i in range(len(cam_info_msg.D))]) + "]\n"
           + "rectification_matrix:\n"
           + "  rows: 3\n"
           + "  cols: 3\n"
           + "  data: [" + ", ".join(["%8f" % i for i in np.array(cam_info_msg.R).reshape(1,9)[0]]) + "]\n"
           + "projection_matrix:\n"
           + "  rows: 3\n"
           + "  cols: 4\n"
           + "  data: [" + ", ".join(["%8f" % i for i in np.array(cam_info_msg.P).reshape(1,12)[0]]) + "]\n"
           + "\n")
    with open(filename, 'w') as f:
        #f.write("%YAML:1.0\n")
        #yaml.dump(calib, f)
        f.write(txt)
