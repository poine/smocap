import rospy, math, numpy as np, tf.transformations, yaml, cv2
import matplotlib, matplotlib.pyplot as plt, mpl_toolkits.mplot3d
import pdb


def deg_of_rad(_r): return _r/math.pi*180.

# Load camera model
# I should use something from camera_calibration_parsers
def load_camera_model(filename, verbose=False):
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
# what a pos!
def write_camera_model(filename, cam_info_msg, cname='unknown'):
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



# TF messages
def list_of_position(p): return (p.x, p.y, p.z)
def list_of_orientation(q): return (q.x, q.y, q.z, q.w)


# This is bullshit... backwards!!!!! idiot!!!!
#def position_and_orientation_from_T(p, q, T):
#    p.x, p.y, p.z = T[:3, 3]
#    q.x, q.y, q.z, q.w = tf.transformations.quaternion_from_matrix(T)

def _position_and_orientation_from_T(p, q, T):
    p.x, p.y, p.z = T[:3, 3]
    q.x, q.y, q.z, q.w = tf.transformations.quaternion_from_matrix(np.linalg.inv(T))
    q.w = -q.w
    #p.x, p.y, p.z = 1, 1, 0
    #q.x, q.y, q.z, q.w = 0, 0, 0, 1
    
def t_q_of_transf_msg(transf_msg):
    return list_of_position(transf_msg.translation), list_of_orientation(transf_msg.rotation)

    
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


# matplotlib drawing

# 3D scene
def draw_thriedra(ax, T_thriedra_to_world__thriedra, alpha=1., colors=['r', 'g', 'b'], scale=1., ls='-', id=None):
    ''' Draw thriedra in w frame '''
    t, R = tR_of_T(T_thriedra_to_world__thriedra)
    for i in range(3):
        p1 = t + scale*R[:,i] # aka p1 = t + np.dot(R, v) with v axis vector ([1 0 0], [0 1 0], [0 0 1])
        ax.plot([t[0], p1[0]], [t[1], p1[1]], [t[2], p1[2]], ls, color=colors[i], alpha=alpha)
    if id is not None:
        annotate3D(ax, s=str(id), xyz= T_thriedra_to_world__thriedra[:3,3], fontsize=10, xytext=(-3,3),
                   textcoords='offset points', ha='right',va='bottom') 

        
def draw_camera(ax, T_c_to_w__c, id=None, color='k'):
    ''' draw a camera as a pyramid '''
    draw_thriedra(ax, T_c_to_w__c, scale=0.1, id=id)
    w, h, d = 0.1, 0.05, 0.25
    pts_c = [[ 0,  0, 0, 1],
             [ w,  h, d, 1],
             [-w,  h, d, 1],
             [ 0,  0, 0, 1],
             [ w, -h, d, 1],
             [-w, -h, d, 1],
             [ 0,  0, 0, 1],
             [ w,  h, d, 1],
             [ w, -h, d, 1],
             [-w, -h, d, 1],
             [-w,  h, d, 1]]
    pts_w = np.array([np.dot(T_c_to_w__c, pt_c) for pt_c in pts_c])
    ax.plot(pts_w[:,0], pts_w[:,1], pts_w[:,2], color=color)

        
def set_3D_axes_equal(ax=None):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    if ax is None: ax = plt.gca()
    
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



    # http://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
class Annotation3D(matplotlib.text.Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        matplotlib.text.Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mpl_toolkits.mplot3d.proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        matplotlib.text.Annotation.draw(self, renderer)


def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)




'''
   Plotting
'''
my_title_spec = {'color'    : 'k', 'fontsize'   : 20 }

def prepare_fig(fig=None, window_title=None, figsize=(20.48, 10.24), margins=None):
    if fig == None:
        fig = plt.figure(figsize=figsize)
    #else:
    #    plt.figure(fig.number)
    if margins:
        left, bottom, right, top, wspace, hspace = margins
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                            hspace=hspace, wspace=wspace)
    if window_title:
         fig.canvas.set_window_title(window_title)
    return fig

def decorate(ax, title=None, xlab=None, ylab=None, legend=None, xlim=None, ylim=None):
    ax.xaxis.grid(color='k', linestyle='-', linewidth=0.2)
    ax.yaxis.grid(color='k', linestyle='-', linewidth=0.2)
    if xlab:
        ax.xaxis.set_label_text(xlab)
    if ylab:
        ax.yaxis.set_label_text(ylab)
    if title:
        ax.set_title(title, my_title_spec)
    if legend <> None:
        ax.legend(legend, loc='best')
    if xlim <> None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim <> None:
        ax.set_ylim(ylim[0], ylim[1])
