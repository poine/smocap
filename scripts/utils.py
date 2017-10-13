import rospy, numpy as np, tf.transformations, yaml
import pdb

# Load camera model
def load_camera_model(filename):
    with open(filename) as f:
        _dict = yaml.load(f)
        camera_matrix = np.array(_dict.get('camera_matrix')['data']).reshape(3, 3)
        dist_coeffs = np.array(_dict['distortion_coefficients']['data'])
        w, h = _dict['image_width'], _dict['image_height']
        print('loading camera calibration from {}'.format(filename))
        print(' camera_matrix\n{}'.format(camera_matrix))
        print(' distortion\n{}'.format(dist_coeffs))
        return camera_matrix, dist_coeffs, w, h


def T_of_rpy_t(rpy, t):
    T = tf.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], 'sxyz')
    T[:3,3] = t#-np.dot(T[:3,:3], t)
    return T

def T_of_quat_t(quat, t):
    T = tf.transformations.quaternion_matrix(quat)
    T[:3,3] = t#-np.dot(T[:3,:3].T, t)
    return T

def T_of_t_q(t, q):
    T = tf.transformations.quaternion_matrix(q)
    T[:3,3] = t#-np.dot(T[:3,:3].T, t)
    return T

def T_of_tq_foo(t, q):
    #q1 = [-q[0], -q[1], -q[2], q[3]]
    T = tf.transformations.quaternion_matrix(q)
    T[:3,3] = t#-np.dot(T[:3,:3].T, t)
    return T

def tq_of_T(T):
    return T[:3, 3], tf.transformations.quaternion_from_matrix(T)

def list_of_position(p): return (p.x, p.y, p.z)
def list_of_orientation(q): return (q.x, q.y, q.z, q.w)

def position_and_orientation_from_T(p, q, T):
    p.x, p.y, p.z = T[:3, 3]
    q.x, q.y, q.z, q.w = tf.transformations.quaternion_from_matrix(T)

def to_homo(p): return np.array([p[0], p[1], 1.])

                
import geometry_msgs.msg
class TfListener:

    def __init__(self):
        self.tfl = tf.TransformListener()


    def get(self, _from, _to):
        #from: http://wiki.ros.org/tf/TfUsingPython
        # lookupTransform(target_frame, source_frame, time) 
        #try:
        (_t, _q) = self.tfl.lookupTransform(_to, _from, rospy.Time(0))
            #print('{}_to_{} {} {}'.format(_from, _to, _t, _q))
        #except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #    _t, _q = None, None
        return (_t, _q)

    def transformPoint(self, p1, _from, _to):
        _p = geometry_msgs.msg.PointStamped()
        _p.header.frame_id = _from
        _p.header.stamp = rospy.Time(0)
        _p.point.x, _p.point.y, _p.point.z = p1[0], p1[1], p1[2]
        p2 = self.tfl.transformPoint(_to, _p)
        #print p1, _from, _to, [p2.point.x, p2.point.y, p2.point.z]
        return [p2.point.x, p2.point.y, p2.point.z]
