import numpy as np
import matplotlib, matplotlib.pyplot as plt, mpl_toolkits.mplot3d
import yaml
import pdb

import smocap.camera
import smocap.plt_utils

class CameraSystem:

    def __init__(self, **kwargs):
        self.cam_names = kwargs.get('cam_names', ['camera_1'])
        self.cameras = [smocap.camera.Camera(i, cam_name) for i, cam_name in enumerate(self.cam_names)]
    
    def read_from_file(self, filename):
        with open(filename) as f:
            _dict = yaml.load(f)
        #print _dict
        cam_names = _dict.keys()
        self.cameras = [smocap.camera.Camera(i, cam_name) for i, cam_name in enumerate(cam_names)]
        for cam, cam_name in zip(self.cameras, cam_names):
            print cam_name
            cam.load_all(_dict[cam_name])
            
        #pdb.set_trace()

    def write_to_file(self, filename):
        txt=''
        for c in self.cameras:
            txt += '{}:\n  {}\n'.format(c.name, c.to_yaml())

        with open(filename, 'w') as f:
            f.write(txt)
            

        
    def get_cameras(self):
        return self.cameras

    def get_camera(self, idx):
        return self.cameras[idx]

    def nb_cams(self):
        return len(self.cameras)



    def plot(self):
        figure = plt.figure()
        ax = figure.add_subplot(111, projection='3d')
        # draw origin
        smocap.plt_utils.draw_thriedra(ax, np.eye(4), id='World') # draw World
        # draw cameras
        for cam in self.cameras:
            smocap.plt_utils.draw_camera(ax, cam.cam_to_world_T, cam.name) # draw camera
