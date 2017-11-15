#!/usr/bin/env python
import logging, os, sys, math, numpy as np, cv2, matplotlib, matplotlib.pyplot as plt, pickle
import scipy.spatial.distance
import fractions # math.gcd for python >= 3.5
import test_detect, utils
import pdb

# https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
# http://techlab.bu.edu/files/resources/articles_tt/pena_lopez_rios_corona2005.pdf
# https://www.math.uci.edu/icamp/summer/research_11/park/shape_descriptors_survey_part3.pdf
# http://users.isr.ist.utl.pt/~jxavier/icip2008a.pdf
# http://www.vis.uky.edu/~ryang/Teaching/cs635-2016spring/Lectures/16-representation.pdf
# http://users.isr.ist.utl.pt/~aguiar/2011-TIP-crespo.pdf

m0  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
m1  = np.array([[0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
m5  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.02, 0, 0]])
m6  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0., 0.0225, 0]])
m7  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0., -0.0225, 0]])
m8  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.04, 0.045, 0]])
m9  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.04, -0.045, 0]])
m10 = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.04, -0.045, 0], [0.08, 0., 0]])

m12 = np.array([[0, 0, 0], [0, 0.045, 0], [0.045, 0.045, 0],  [0.045, 0.0, 0]]) # a square


class MarkersDatabase:
    def __init__(self, verbose=False):
        self.ms = [{'pts':m0, 'id':'m0'},
                   {'pts':m1, 'id':'m1'},
                   {'pts':m5, 'id':'m5'},
                   {'pts':m6, 'id':'m6'},
                   {'pts':m7, 'id':'m7'},
                   {'pts':m8, 'id':'m8'},
                   {'pts':m9, 'id':'m9'},
                   {'pts':m10, 'id':'m10'}]
        for m in self.ms:
            m['marker'] = Marker(m['pts'], m['id'])
            m['marker'].compute_sig(verbose)
        self.compute_stats()

    def find(self, m):
        ds = [np.linalg.norm(m._sig - _m['marker']._sig) for _m in self.ms]
        print('distances {}'.format(ds))
        match = self.ms[np.argmin(ds)]
        print('matched {}'.format(match['id']))
        return match

    def plot(self, n_col=7):
        margins = (0.08, 0.06, 0.97, 0.95, 0.15, 0.33)
        fig = utils.prepare_fig(window_title='database', margins=margins)
        n_row = int(math.ceil(len(self.ms)/float(n_col))) + 2
        for i in range(n_row):
            for j in range(n_col):
                k = i*n_col+j
                if k >= len(self.ms): break
                ax = plt.subplot(n_row, n_col, k+1)
                _m = self.ms[k]['marker']
                plt.scatter(_m.pts[:,0], _m.pts[:,1])
                plt.scatter(_m.Xc[0], _m.Xc[1], color='r')
                s = 0.025
                #pdb.set_trace()
                has = ax.arrow(_m.Xc[0], _m.Xc[1], _m.smallest_evec[0]*s, _m.smallest_evec[1]*s,
                               head_width=0.005, head_length=0.01, fc='g', ec='k', alpha=0.5)
                hal = ax.arrow(_m.Xc[0], _m.Xc[1], _m.largest_evec[0]*s, _m.largest_evec[1]*s,
                               head_width=0.005, head_length=0.01, fc='r', ec='k', alpha=0.5)
                leg = plt.legend(handles=[has, hal], labels=['smallest ev', 'largest ev'])
                leg.get_frame().set_alpha(0.2)
                utils.decorate(ax, title=_m.name, xlab='x', ylab='y')#, legend=['1', '2'])

                ax.set_aspect('equal')

        for i in range(n_col):
            ax = plt.subplot(n_row, n_col, n_col+i+1)
            m = self.ms[i]['marker']
            plt.scatter(m.pts_c2[:,0], m.pts_c2[:,1])
            ax.set_aspect('equal')
            utils.decorate(ax, title='normalized', xlab='x', ylab='y')
             
        ax = plt.subplot(n_row, 1, n_row)
        plt.text(0.5, 0.5, "{}".format(self.inter_marker_dists), family='monospace')
        
    def compute_stats(self):
        self.inter_marker_dists = np.zeros((len(self.ms), len(self.ms)))
        for i in range(len(self.ms)):
            for j in range(i+1, len(self.ms)):
                mi, mj = self.ms[i]['marker'], self.ms[j]['marker']
                self.inter_marker_dists[i, j] = np.linalg.norm( mi._sig - mj._sig)
        #print self.inter_marker_dists




        
def unit_circle_points(n):
    return [complex(math.cos(alpha), math.sin(alpha)) for alpha in np.linspace(0, 2*math.pi, n, endpoint=False)]

def ansig(zs, xi):
    return 1./len(zs)*np.sum(np.exp(zs*xi))

def _sig(pts, n=8):
    zs = np.array([np.complex(p[0], p[1]) for p in pts])
    _s = [ansig(zs, xi) for xi in unit_circle_points(n)]
    return np.array(_s)

def norm_angle(_a):
    while _a <= -math.pi: _a += 2*math.pi
    while _a >   math.pi: _a -= 2*math.pi    
    return _a

def norm_angle2(_a):
    while _a <= 0:        _a += 2*math.pi
    while _a > 2*math.pi: _a -= 2*math.pi    
    return _a

class Marker():

    _sig_nb_pt = 16
    
    def __init__(self, pts, name='unknown'):
        self.pts = pts
        self.name = name
        
    def compute_centroid(self):
        self.Xc = np.mean(self.pts, axis=0)

    def compute_axis(self, verbose=False):
        cov = np.cov(self.pts_c1[:,:2], rowvar=False)
        evals, evecs = np.linalg.eig(cov)
        evals_sort_idx = np.argsort(evals)
        sorted_evals = evals[evals_sort_idx]
        sorted_evecs = evecs[evals_sort_idx]
        #print evals, evecs, sorted_evecs
        #print('sorted evals {}'.format(sorted_evals))
        if verbose: print('sorted evecs\n{}'.format(sorted_evecs))
        self.smallest_evec = sorted_evecs[:,0]
        self.largest_evec = sorted_evecs[:,1]
        #if verbose:
        #    pdb.set_trace()
        self.axis_angle = math.atan2(self.smallest_evec[1], self.smallest_evec[0]) 

    def compute_normalized(self, verbose=False):
        if verbose: print('{}:\n{}'.format(self.name, [p[0:2].tolist() for p in self.pts]))
        # compute centroid and points in centroid centered frame
        self.compute_centroid()
        if verbose: print('Xc {}'.format(self.Xc))
        self.pts_c = self.pts - self.Xc
        if verbose: print('pts_c\n{}'.format(self.pts_c[:,:2]))
        # normalize scale
        self.scale = np.sum(np.linalg.norm(self.pts_c, axis=1))
        if verbose: print('scale {}'.format(self.scale))
        self.pts_c1 = self.pts_c/self.scale
        if verbose: print('pts_c1\n{}'.format(self.pts_c1[:,:2]))

    def compute_power_sums(self, K=5):
        self.zsk = np.zeros((K, len(self.pts)), dtype=complex) # points as complex numbers
        self.zsk[0] = [complex(pt[0], pt[1]) for pt in self.pts_c1]
        for i in range(1,K):
            self.zsk[i] = self.zsk[i-1]*self.zsk[0]
        self.mus = np.sum(self.zsk, axis=1)

    def pm_analysis(self, tau=1e-3):
        print '{} pm_analysis\n--------------\n mus {}'.format(self.name, self.mus)
        K = np.argwhere(np.absolute(self.mus) > tau) + 1 # find non zero moments
        print (' K {}'.format(K.squeeze()))
        gamma = reduce(fractions.gcd, K)                 # find fold number
        print (' gamma {}'.format(gamma))
        argmus = np.angle(self.mus)
        print ' args: {}'.format(argmus)
        #pdb.set_trace()
        def get_angle(k, m):
            arg_muk, arg_mum = argmus[k-1], argmus[m-1]
            darg = arg_mum - m/k*arg_muk
            print 'darg {}'.format(darg)
            #pdb.set_trace()
            l = 0
            while l < k:
                bar = norm_angle(darg - 2*math.pi/k*l)
                if bar > -math.pi/k and bar <= math.pi/k:
                    break
                l+=1
            if l >= k: print 'l failed'
            print 'l', l
            print('theta {}-{}: {}'.format(k, m,  arg_muk/k + 2*math.pi/k*l))
            return arg_muk/k + 2*math.pi/k*l
        self.theta = get_angle(2, 3)
        get_angle(3, 4)
        get_angle(4, 5)
        print (' theta {}'.format(self.theta))
        self.emjtheta = complex(math.cos(-self.theta), math.sin(-self.theta))
        self.nmus = np.array([mu*complex(math.cos(-(i+1)*self.theta), math.sin(-(i+1)*self.theta)) for i, mu in enumerate(self.mus)]) # rotation normalized moments
        print ('normalized mus: {}'.format(self.nmus))
            
    def compute_sig(self, verbose=False):
        self.compute_normalized(verbose)
        # compute main axis
        self.compute_axis(verbose)
        if verbose: print('axis angle {}'.format(self.axis_angle))
        # rotate
        ct, st = math.cos(-self.axis_angle), math.sin(-self.axis_angle)
        R = np.array([[ct, -st],[st, ct]])
        self.pts_c2 = np.array([np.dot(R, p[:2]) for p in self.pts_c1])
        if verbose: print('pts_c2\n{}'.format(self.pts_c2))
        self._sig = _sig(self.pts_c2, Marker._sig_nb_pt)



                

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

def plot_marker(m, label_points=True, plot_pc=False):
    margins = (0.08, 0.06, 0.97, 0.95, 0.15, 0.33)
    fig = utils.prepare_fig(window_title='{}'.format(m.name), figsize=(10.24, 10.24), margins=margins)
    ax = plt.subplot(1,2,1)
    plt.scatter(m.pts[:,0], m.pts[:,1])
    plt.scatter(m.Xc[0], m.Xc[1], color='r')
    if label_points:
        for i, p in enumerate(m.pts):
            ax.text(p[0], p[1], '{}'.format(i))
    if plot_pc:
        s = 0.25*m.scale
        ax.arrow(m.Xc[0], m.Xc[1], m.smallest_evec[0]*s, m.smallest_evec[1]*s,
                 head_width=0.005, head_length=0.01, fc='g', ec='g', alpha=0.5)
        ax.arrow(m.Xc[0], m.Xc[1], m.largest_evec[0]*s, m.largest_evec[1]*s,
                 head_width=0.005, head_length=0.01, fc='r', ec='r', alpha=0.5)
    plt.plot(m.pts[:,0], m.pts[:,1])
    utils.decorate(ax, title='original', xlab='x', ylab='y')
    ax.set_aspect('equal')

    ax = plt.subplot(1,2,2)
    n_pts = m.pts_c1#m.pts_c2
    plt.scatter(n_pts[:,0], n_pts[:,1])
    plt.scatter(0, 0, color='r')
    if label_points:
        for i, p in enumerate(n_pts):
            ax.text(p[0], p[1], '{}'.format(i))
    utils.decorate(ax, title="normalized")
    ax.set_aspect('equal')
    

def get_markers_in_image(img_name='image_11.png', img_enc='bgr8', verbose=False):
    args = {
        'detector_cfg':'/home/poine/work/smocap.git/params/gazebo_detector_cfg.yaml',
        'image_path':'/home/poine/work/smocap.git/test/' + img_name,
        'image_encoding' : img_enc
    }
    print 'loading image {}'.format(args['image_path'])
    td = test_detect.Model(args['detector_cfg'], args['image_encoding'])
    td.load_image(args['image_path'])
    td.correct_gamma()
    td.detect_blobs()
    clusters = td.cluster_blobs()
    ms = [Marker(c) for c in clusters]
    for m in ms:
        m.compute_sig(verbose)
    return ms


def test_marker(_pts, _id, _trans=[0, 0], _rot=0., _scale=1., noise=0, _permutation=True):
    _pts1 = _pts[:,:2] + _trans
    _pts2 = rotate(_pts1, _rot)
    _pts3 = _scale * _pts2
    _pts4 = _pts3 + np.random.normal(scale=noise, size=(len(_pts), 2))
    _pts5 = permutation(_pts4) if permutation else _pts4
    mt = Marker(_pts5, _id)
    mt.compute_sig(verbose=True)
    mdb.find(mt)
    plot_marker(mt)


def test_foo(db, m2):
    ''' test of oxford paper: visual correspondance problem '''
    m2.compute_sig(verbose=False)
    m1 = db.ms[2]['marker']
    pt1 = m1.pts_c1[:,:2]
    n1 = len(pt1)
    print m1.name
    print pt1
    
    pt2 = m2.pts_c1
    n2 = len(pt2)
    print pt2
    dists = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dists[i, j] = np.linalg.norm(pt1[i]-pt2[j])
    print 'dists\n',dists

    G, sigma = np.zeros((n1, n2)), 1
    for i in range(n1):
        for j in range(n2):
            G[i, j] = math.exp(-dists[i, j]/2/sigma)
    print 'G\n', G
    #Y1 = scipy.spatial.distance.pdist(kps_img)
    #Y2 = scipy.spatial.distance.squareform(Y1)
    U, s, V = np.linalg.svd(G, full_matrices=True)
    print' U\n', U
    print 's\n', s
    print 'V\n', V
    S = np.zeros((n1, n2), dtype=complex)
    n3 = min(n1, n2)
    S[:n3, :n3] = np.diag(s)
    print'USV\n',  np.dot(U, np.dot(S, V))
    P = np.dot(U, V)
    print 'P\n', P
    corresp = np.zeros((n1, n2))
    for i in range(n1):
        j = np.argmax(P[i])
        corresp[i, j] = 1
    print 'corresp\n', corresp
    plot_marker(m1)
    plot_marker(m2)
    #pdb.set_trace()



def test_rotation(_rot, _pts, _id):
    _m1 = Marker(_pts, _id)
    _m1.compute_sig(verbose=False)
    _m1.compute_power_sums()
    _m1.pm_analysis()
    plot_marker(_m1)
    print 
    _ptsbis = transform_points(_pts, _trans=[0., 0.], _rot=_rot, _scale=1., _noise=0., _permutation=False)
    _m1bis = Marker(_ptsbis, '{}bis'.format(_id))
    _m1bis.compute_sig(verbose=False)
    _m1bis.compute_power_sums()
    _m1bis.pm_analysis()
    plot_marker(_m1bis)
    
    print 'rot found {} truth {}'.format( _m1bis.theta - _m1.theta, _rot )


def test_baz(ms):
    for i, m in enumerate(ms):
        m.name = 'unkown {}'.format(i)
        m.compute_normalized(verbose=False)
        m.compute_power_sums()
        m.pm_analysis()
        plot_marker(m)
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=300, suppress=True)

    Marker._sig_nb_pt = 16
    
    mdb = MarkersDatabase()
    #mdb.plot()

    
    
    if 0:
        test_marker(m6, 'm6_test', _trans=[1, 0], _rot=0.5, _scale=10.1, noise=0.01, _permutation=False)#1e-16)
    if 0:
        ms = get_markers_in_image()
        for m in ms:
            match = mdb.find(m)
            m.name = 'matched to {}'.format(match['id'])
            plot_marker(m)
    if 0:
        ms = get_markers_in_image(img_name='2_markers_diff_01.png', img_enc='mono8')
        test_foo(mdb, ms[0])
    if 1:
        #ms = get_markers_in_image(img_name='2_markers_diff_01.png', img_enc='mono8')
        test_rotation(0.1, m1, "m1")
        #test_rotation(0.1, m9, "m9")
        
    plt.show()
    #m = m0
    #print m, permutation(m)
    #print centroid(m)
    #print dists(m)
    #test_id(m)
    #test_id(permutation(m))

    #test_id(m9)
    #test_id(permutation(m9))
    
    
    #get_axis(m)
    #test(m0)
    #get_axis(rotate(m, 0.75))
    #plot(rotate(m, 0.75))
    #plot(m)
