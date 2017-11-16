#!/usr/bin/env python
import logging, math, numpy as np, matplotlib.pyplot as plt

import smocap.shapes
import test_detect

def permutation(m):
    return m[np.random.permutation(len(m))]

def rotate(pts, angle):
    ct, st = math.cos(angle), math.sin(angle)
    R = np.array([[ct, -st],[st, ct]])
    return np.array([np.dot(R, p) for p in pts])

def transform_points(_pts, _trans=[0., 0.], _rot=0., _scale=1., _noise=0., _permutation=False):
    _pts1 = _pts[:,:2] + _trans
    _pts2 = rotate(_pts1, _rot)
    _pts3 = _scale * _pts2
    _pts4 = _pts3 + np.random.normal(scale=_noise, size=(len(_pts), 2))
    _pts5 = permutation(_pts4) if _permutation else _pts4
    return _pts5


def get_clusters_in_image(img_name='2_markers_diff_01.png', img_enc='mono8'):
    args = {
        'detector_cfg':'/home/poine/work/smocap.git/params/f111_detector_default.yaml',
        'image_path':'/home/poine/work/smocap.git/test/' + img_name,
        'image_encoding' : img_enc
    }
    print 'loading image {}'.format(args['image_path'])
    td = test_detect.Model(args['detector_cfg'], args['image_encoding'])
    td.load_image(args['image_path'])
    td.correct_gamma()
    td.detect_blobs()
    return td.cluster_blobs(), td


def test_1(db):
    ''' transform a marker from the database and searches it '''
    m_x = db.shapes[1]
    pts_x = transform_points(m_x.pts, _trans=[1., 0.], _rot=0.5, _scale=2., _noise=0., _permutation=False)
    s_x = smocap.shapes.Shape(pts_x)
    s_x.compute_signature()

    id_sx, s_ref = db.find(s_x)
    print('found idx:{} with trans {}, rot {}, scale {}'.format(id_sx, s_x.Xc-s_ref.Xc, s_x.theta-s_ref.theta, s_x.scale/s_ref.scale))


def test_2(db):
    ''' find markers in image '''
    clusters, td = get_clusters_in_image(img_name='2_markers_diff_03.png')
    shapes = [smocap.shapes.Shape(pts) for pts in clusters]
    for s in shapes: s.compute_signature()
    matches = [db.find(s) for s in shapes]
    for s_x, (id_sx, s_ref) in zip(shapes, matches):
        print('found idx:{} with trans {}, rot {}, scale {}'.format(id_sx, s_x.Xc-s_ref.Xc, s_x.theta-s_ref.theta, s_x.scale/s_ref.scale))
    plt.imshow(td.img_res)
    for s_x, (id_sx, s_ref) in zip(shapes, matches):
        xc = s_x.Xc; dx = 50*np.array([math.cos(s_x.theta), math.sin(s_x.theta)])
        plt.scatter(*xc)
        plt.arrow(xc[0], xc[1], dx[0], dx[1], ec='r', width=0.1)
        plt.text(xc[0], xc[1], "{}".format(id_sx), family='monospace', color='r')
    plt.show()
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=300, suppress=True)
    db = smocap.shapes.Database()
    
    #test_1(db)
    test_2(db)
