#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os, cv2, numpy as np

def create_aruco_textures(_dir='/home/poine/work/smocap.git/smocap_gazebo/media/materials/',
                          _max_id=64):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    for _id in range(_max_id):
        img = cv2.aruco.drawMarker(aruco_dict, _id, 1280)
        img_rot = np.rot90(img) # warning, I am rotating the texture here as I don't know how to do it in OGRE!!!!
        cv2.imwrite(os.path.join(_dir, 'textures/aruco_{:03d}.jpg'.format(_id)), img_rot)
    with open(os.path.join(_dir, 'scripts/aruco.material'), 'w') as f:
        for _id in range(_max_id):
            txt = '''
material Aruco{:d} {{
    technique {{
        pass {{
            texture_unit {{
                texture aruco_{:03d}.jpg
                scale 1 1
            }}
        }}
    }}
}}'''.format(_id, _id)
            f.write(txt)
        

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    create_aruco_textures()
