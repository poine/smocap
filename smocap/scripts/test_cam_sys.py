#!/usr/bin/env python
import os, logging
import rospy, rospkg
import matplotlib.pyplot as plt

import smocap.camera_system


def fetch_from_ros(cam_names):
    pass


def write_cam_sys(filename):
    cam_sys = smocap.camera_system.CameraSystem()
    cam_sys.read_from_file(filename)
    cam_sys.write_to_file('/tmp/foo')
    
def plot_cam_sys(filename):
    cam_sys = smocap.camera_system.CameraSystem()
    cam_sys.read_from_file(filename)
    cam_sys.plot()
    plt.show()


def main():
    swd = rospkg.RosPack().get_path('smocap')
    write_cam_sys(filename = os.path.join(swd, 'params/enac_demo_z/camera_system_full.yaml'))
    plot_cam_sys(filename = '/tmp/foo') #swd+'/params/enac_demo_z/camera_system_full.yaml')


    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    
