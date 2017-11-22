import math, numpy as np
import pdb

import smocap.utils as ut

class Shape:
    ''' A shape is a set of unordered 2D points '''
    def __init__(self, pts):
        self._pts = pts # 3D
        self.pts = pts[:,:2]
        self.compute_signature()
        
    def compute_signature(self):
        self.normalize()
        self.compute_cms()
        self.cms_analysis()

    def normalize(self):
        ''' normalize points (translation and scale) around centroid '''
        self.Xc = np.mean(self.pts, axis=0)
        self.pts_c = self.pts - self.Xc
        self.scale = np.sum(np.linalg.norm(self.pts_c, axis=1))
        self.pts_cs = self.pts_c/(self.scale if self.scale else 1.)

    def compute_cms(self, K=5):
        ''' compute the K first complex moments '''
        self.zsk = np.zeros((K, len(self.pts)), dtype=complex) # points as complex numbers
        self.zsk[0] = [complex(pt[0], pt[1]) for pt in self.pts_cs]
        for i in range(1,K):
            self.zsk[i] = self.zsk[i-1]*self.zsk[0]
        self.mus = np.sum(self.zsk, axis=1)
        
    def cms_analysis(self):
        ''' WTF!!!! '''
        #print('mus {}'.format(self.mus))
        argmus = np.angle(self.mus)
        def get_angle(k, m):
            arg_muk, arg_mum = argmus[k-1], argmus[m-1]
            foo = arg_mum - m/k*arg_muk
            l = 0
            while l < k:
                bar = ut.norm_angle(foo - 2.*math.pi/k*l)
                if bar > -math.pi/k and bar <= math.pi/k:
                    break
                l+=1
            theta = arg_muk/k + 2.*math.pi/k*l
            return theta
        self.theta = get_angle(3, 4)
        # rotation normalized moments
        self.nmus = np.array([mu*complex(math.cos(-(i+1)*self.theta), math.sin(-(i+1)*self.theta)) for i, mu in enumerate(self.mus)]) 
        #print ('normalized mus: {}'.format(self.nmus))


    def sort_points(self, debug=False, sort_cw=False):
        # invert sort if for the fucking y down on images :(

        self.zs_r_normalized = np.array([z*complex(math.cos(-self.theta), math.sin(-self.theta)) for z in self.zsk[0]])
        # Sort by angle only doen't work. I do not always get the correct starting point :(
        # what to do???
        # do I sort with norm? that is not discrimitative either in my fucking triangle
        # my naive way was sorting first by norm, then by angles (the other way around, really)...
        if 0: # ok that can't work - I'd say because of -pi;pi line :(
            self.args_zs_r_normalized = np.angle(self.zs_r_normalized)
            self.angle_sort_idx = np.argsort(self.args_zs_r_normalized)
            tmp = self.zs_r_normalized[self.angle_sort_idx]
            self.abs_zs_r_normalized = np.absolute(tmp)
            self.abs_sort_idx = np.argsort(self.abs_zs_r_normalized)
            self.sort_idx = self.angle_sort_idx[self.abs_sort_idx]
            self.pts_sorted = self.pts[self.sort_idx]
            self._pts_sorted = self._pts[self.sort_idx]
        pt_ref_idx = np.argmax(self.zs_r_normalized.real)
        mask = np.equal(np.arange(len(self.zs_r_normalized)),  pt_ref_idx)
        other_pts_in_ref = np.ma.masked_array(self.zs_r_normalized - self.zs_r_normalized.real[pt_ref_idx] , mask)
        if sort_cw:
            other_pts_in_ref_arg = np.arctan2(-other_pts_in_ref.imag, -other_pts_in_ref.real)
        else:
            other_pts_in_ref_arg = np.arctan2(other_pts_in_ref.imag, -other_pts_in_ref.real)
        self.sort_idx = np.argsort(other_pts_in_ref_arg)[::-1]
        self.pts_sorted = self.pts[self.sort_idx]
        self._pts_sorted = self._pts[self.sort_idx]
        if debug:
            pdb.set_trace()

            
class Database:

    def __init__(self):
        m0  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
        m1  = np.array([[0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
        self.shapes = [Shape(m0), Shape(m1)]
        for s in self.shapes:
            #s.compute_signature() # done in shape constructor
            s.sort_points()
        self.sigs = [s.nmus for s in self.shapes]

    def find(self, sx):
        dists = [np.linalg.norm(sx.nmus - s.nmus) for s in self.shapes]
        match_idx = np.argmin(dists)
        if len(sx.pts) == len(self.shapes[match_idx].pts):
            return match_idx, self.shapes[match_idx]
        else:
            return -1, None
