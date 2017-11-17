import math, numpy as np

import smocap.utils as ut

class Shape:
    ''' A shape is a set of unordered 2D points '''
    def __init__(self, pts):
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


    def sort_points(self):
        self.zs_r_normalized = [z*complex(math.cos(-self.theta), math.sin(-self.theta)) for z in self.zsk[0]]
        self.args_zs_r_normalized = np.angle(self.zs_r_normalized)
        print self.zs_r_normalized
        print self.args_zs_r_normalized
        self.angle_sort_idx = np.argsort(self.args_zs_r_normalized)
        print self.angle_sort_idx
        self.zs_sorted = self.zsk[0, self.angle_sort_idx]
        print 'sorted zs ', self.zs_sorted

class Database:

    def __init__(self):
        m0  = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
        m1  = np.array([[0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
        self.shapes = [Shape(m0), Shape(m1)]
        for s in self.shapes:
            s.compute_signature()
        self.sigs = [s.nmus for s in self.shapes]

    def find(self, sx):
        dists = [np.linalg.norm(sx.nmus - s.nmus) for s in self.shapes]
        match_idx = np.argmin(dists)
        if len(sx.pts) == len(self.shapes[match_idx].pts):
            return match_idx, self.shapes[match_idx]
        else:
            return -1, None
