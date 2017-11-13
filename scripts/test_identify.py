#!/usr/bin/env python
import logging, os, sys, math, numpy as np, cv2, matplotlib, matplotlib.pyplot as plt, pickle
import scipy.spatial.distance
import test_detect, utils
import pdb


class Marker():
    def __init__(self):
        pass


# https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/

# http://techlab.bu.edu/files/resources/articles_tt/pena_lopez_rios_corona2005.pdf

m0 = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0]])
m5 = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.02, 0, 0]])
m6 = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0., 0.0225, 0]])
m7 = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0., -0.0225, 0]])

m8 = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.04, 0.045, 0]])
m9 = np.array([[0, 0, 0], [0, 0.045, 0], [0, -0.045, 0], [0.04, 0, 0], [0.04, -0.045, 0]])



ms = [{ 'id':'m0', 'pts':m0},
      { 'id':'m5', 'pts':m5},
      { 'id':'m6', 'pts':m6}]
      
def permutation(m):
    return m[np.random.permutation(len(m))]

def rotate(m, theta):
    ct, st = math.cos(theta), math.sin(theta)
    R = np.array([[ct, -st, 0],[st, ct, 0], [0, 0, 1]])
    return np.array([np.dot(R, p) for p in m])

def centroid(m):
    return np.mean(m, axis=0)
    #pdb.set_trace()

def dists(m):
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(m))

def get_axis(m):
    c = centroid(m)
    mprim = m - c
    cov = np.cov(mprim[:,:2].T)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    print cov
    print evals, evecs
    print math.atan2(evecs[0, 1], evecs[1, 1]) 

def test_id(m):
    print m
    d =  dists(m)
    print d
    print np.sum(d, axis=0)
    pdb.set_trace()


def sort(pts, Xc):
    pts_in_c = pts-Xc
    norms = np.linalg.norm(pts_in_c[:,:2], axis=1)
    angles = np.arctan2(pts_in_c[:,1], pts_in_c[:,0])
    print 'pts ', pts_in_c
    print 'norms ', norms
    print 'angles ',angles
    sort_norms_idx = np.argsort(norms)
    sort_idx = np.argsort(angles[sort_norms_idx])
    print pts_in_c[sort_idx]
    
def compute_sig(pts, name):
    foo = [p[0:2].tolist() for p in pts]
    print('{}:\n{}'.format(name, foo))
    Xc = centroid(pts)
    print('Xc {}'.format(Xc))
    pts_c = pts - Xc
    print('pts_c\n{}'.format(pts_c[:,:2]))
    cov = np.cov(pts_c[:,:2].T)
    evals, evecs = np.linalg.eig(cov)
    evals_sort_idx = np.argsort(evals)
    sorted_evals = evals[evals_sort_idx]
    sorted_evecs = evecs[evals_sort_idx]
    #print evals, evecs, sorted_evecs
    print('sorted evals {}'.format(sorted_evals))
    print('sorted evecs\n{}'.format(sorted_evecs))
    smallest_evec = sorted_evecs[0]
    axis_angle = math.atan2(smallest_evec[1], smallest_evec[0]) 
    print('axis angle {}'.format(axis_angle))
    ct, st = math.cos(axis_angle), math.sin(axis_angle)
    R = np.array([[ct, -st],[st, ct]])
    pts_c1 = np.array([np.dot(R, p[:2]) for p in pts_c])
    print('pts_c1\n{}'.format(pts_c1))
    sig = pts_c1[:,:2]
    angles = np.arctan2(sig[:,1], sig[:,0])
    norms = np.linalg.norm(sig[:,:2], axis=1)
    print('angles {}'.format(angles))
    print('norms {}'.format(norms))
    
    sig = sig[np.argsort(angles)]
    norm_sig = np.linalg.norm(sig)
    sig /= norm_sig

    print('sig\n{}'.format(sig.T))
    #sort(pts, Xc)
    
    
def plot(m):
    plt.scatter(m[:,0], m[:,1])
    plt.gca().set_aspect('equal')


def identify_marker():
    args = {
        'detector_cfg':'/home/poine/work/smocap.git/params/gazebo_detector_cfg.yaml',
        'image_path':'/home/poine/work/smocap.git/test/image_2.png',
        'image_encoding':'bgr8'
    }
    td = test_detect.Model(args['detector_cfg'], args['image_encoding'])
    td.load_image(args['image_path'])
    td.correct_gamma()
    td.detect_blobs()
    clusters = td.cluster_blobs()
    compute_sig(clusters[0], '1')
    return clusters[0]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=300, suppress=True)
    #for m in ms:
    #    compute_sig(m['pts'], m['id'])
    compute_sig(m0, 'm0')
    plot(m0)
    print
    m_foo = identify_marker()
    plt.figure()
    plot(m_foo)
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
