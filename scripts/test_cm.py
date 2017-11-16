#!/usr/bin/env python
import logging, math, numpy as np, matplotlib.pyplot as plt
LOG = logging.getLogger('test_cm')
import fractions # math.gcd for python >= 3.5
import pdb

m0  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
m1  = np.array([[0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
m5  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.02, 0, 0]])
m6  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0., 0.0225, 0]])
m7  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0., -0.0225, 0]])
m8  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.04, 0.045, 0]])
m9  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.04, -0.045, 0]])
m10 = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.04, -0.045, 0], [0.08, 0., 0]])
m12 = np.array([[0, 0, 0], [0, 0.045, 0], [0.045, 0.045, 0],  [0.045, 0.0, 0]]) # a square

def norm_angle(_a):
    while _a <= -math.pi: _a += 2*math.pi
    while _a >   math.pi: _a -= 2*math.pi    
    return _a

class Marker():

    def __init__(self, pts, name='unknown'):
        self.pts, self.name = pts, name

    def normalize(self):
        ''' normalize points (translation and scale) around centroid '''
        self.Xc = np.mean(self.pts, axis=0)
        self.pts_c = self.pts - self.Xc
        self.scale = np.sum(np.linalg.norm(self.pts_c, axis=1))
        self.pts_cs = self.pts_c/self.scale

    def compute_cms(self, K=5):
        ''' compute the K first complex moments '''
        self.zsk = np.zeros((K, len(self.pts)), dtype=complex) # points as complex numbers
        self.zsk[0] = [complex(pt[0], pt[1]) for pt in self.pts_cs]
        for i in range(1,K):
            self.zsk[i] = self.zsk[i-1]*self.zsk[0]
        self.mus = np.sum(self.zsk, axis=1)

    def cm_analysis(self):
        ''' WTF!!!! '''
        argmus = np.angle(self.mus)
        def get_angle(k, m):
            arg_muk, arg_mum = argmus[k-1], argmus[m-1]
            foo = arg_mum - m/k*arg_muk
            LOG.debug('foo {}'.format(foo))
            l = 0
            while l < k:
                bar = norm_angle(foo - 2*math.pi/k*l)
                if bar > -math.pi/k and bar <= math.pi/k:
                    break
                l+=1
            LOG.debug('l {}'.format(l))
            LOG.debug('theta {}-{}: {}'.format(k, m,  arg_muk/k + 2*math.pi/k*l))
            return arg_muk/k + 2*math.pi/k*l
        self.theta = get_angle(2, 3)


def plot_marker(m, label_points=True):
    margins = (0.08, 0.06, 0.97, 0.95, 0.15, 0.33)
    fig = plt.figure(figsize=(10.24, 10.24))
    fig.canvas.set_window_title(m.name)#utils.prepare_fig(window_title='{}'.format(m.name), figsize=(10.24, 10.24), margins=margins)

    ax = plt.subplot(1,2,1)
    plt.scatter(m.pts[:,0], m.pts[:,1])
    plt.scatter(m.Xc[0], m.Xc[1], color='r')
    if label_points:
        for i, p in enumerate(m.pts):
            ax.text(p[0], p[1], '{}'.format(i))
    plt.plot(m.pts[:,0], m.pts[:,1])
    ax.set_title('original')
    ax.set_aspect('equal')

    ax = plt.subplot(1,2,2)
    n_pts = m.pts_cs
    plt.scatter(n_pts[:,0], n_pts[:,1])
    plt.scatter(0, 0, color='r')
    if label_points:
        for i, p in enumerate(n_pts):
            ax.text(p[0], p[1], '{}'.format(i))
    ax.set_title('trans-scale normalized')
    ax.set_aspect('equal')
    
def permutation(m):
    return m[np.random.permutation(len(m))]

def rotate(pts, angle):
    ct, st = math.cos(angle), math.sin(angle)
    R = np.array([[ct, -st],[st, ct]])
    return np.array([np.dot(R, p) for p in pts])

def transform_points(_pts, _trans=[0., 0.], _rot=0., _scale=1., _noise=0., _permutation=True):
    _pts1 = _pts[:,:2] + _trans
    _pts2 = rotate(_pts1, _rot)
    _pts3 = _scale * _pts2
    _pts4 = _pts3 + np.random.normal(scale=_noise, size=(len(_pts), 2))
    _pts5 = permutation(_pts4) if _permutation else _pts4
    return _pts5

def test_rotation(_angle, _pts, _id):
    _m1 = Marker(_pts, _id)
    _m1.normalize()
    _m1.compute_cms()
    _m1.cm_analysis()
    plot_marker(_m1)

    _ptsbis = transform_points(_pts, _trans=[0., 0.], _rot=_angle, _scale=1., _noise=0., _permutation=False)

    _m1prim = Marker(_ptsbis, '{}prim'.format(_id))
    _m1prim.normalize()
    _m1prim.compute_cms()
    _m1prim.cm_analysis()
    plot_marker(_m1prim)

    print 'rotation: computed {} truth {}'.format( _m1prim.theta - _m1.theta, _angle )
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=300, suppress=True)

    for angle in np.linspace(-0.2, 0.2, 5):
        test_rotation(angle, m1, "m1")
    plt.show()
