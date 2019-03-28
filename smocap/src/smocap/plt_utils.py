import numpy as np
import matplotlib, matplotlib.pyplot as plt, mpl_toolkits.mplot3d



# matplotlib drawing
import smocap.utils


# 3D scene

def draw_thriedra(ax, T_thriedra_to_world__thriedra, alpha=1., colors=['r', 'g', 'b'], scale=1., ls='-', id=None):
    ''' Draw thriedra in w frame '''
    t, R = smocap.utils.tR_of_T(T_thriedra_to_world__thriedra)
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
