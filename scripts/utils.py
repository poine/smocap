import numpy as np, tf.transformations
import pdb

def T_of_rpy_t(rpy, t):
    T = tf.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], 'sxyz')
    T[:3,3] = t#-np.dot(T[:3,:3], t)
    return T

def T_of_quat_t(quat, t):
    T = tf.transformations.quaternion_matrix(quat)
    T[:3,3] = t#-np.dot(T[:3,:3].T, t)
    return T

def list_of_position(p): return (p.x, p.y, p.z)
def list_of_orientation(q): return (q.x, q.y, q.z, q.w)

def position_and_orientation_from_T(p, q, T):
    p.x, p.y, p.z = T[:3, 3]
    q.x, q.y, q.z, q.w = tf.transformations.quaternion_from_matrix(T)

def to_homo(p): return np.array([p[0], p[1], 1.])

    
import rospy, gazebo_msgs.msg
''' Subscribe to gazebo/model_states messages and store the robot pose and twist '''
class TruthListener:
    def __init__(self, model_name="rosmip"):
        self.model_name, self.model_idx = model_name, None
        rospy.Subscriber('/gazebo/model_states', gazebo_msgs.msg.ModelStates, self.callback)
        self.pose, self.twist = None, None
        self.body_to_world_T = None

    def callback(self, msg):
        if self.model_idx is None:
            try:
                self.model_idx = msg.name.index(self.model_name)
            except:
                rospy.logerr('model {} not found in gazebo {}'.format(self.model_name, msg.name))
        if self.model_idx is not None:
            self.pose, self.twist = msg.pose[self.model_idx], msg.twist[self.model_idx]
            self.body_to_world_T = None


    def get_body_to_world_T(self):
        if self.pose is not None and self.body_to_world_T is None:
            _t, _q = list_of_position(self.pose.position), list_of_orientation(self.pose.orientation)
            #print('thruth listener base to world {} {}'.format(_t, _q))
            self.body_to_world_T = T_of_quat_t(_q, _t, )
            #pdb.set_trace()
        return self.body_to_world_T
            
import geometry_msgs.msg
class TfListener:

    def __init__(self):
        self.tfl = tf.TransformListener()


    def get(self, _from, _to):
        #try:
        (_t, _q) = self.tfl.lookupTransform(_from, _to, rospy.Time(0))
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
