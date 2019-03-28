---
title: Smocap Simulation
layout: default
---

## Gazebo


# starts gazebo, rc marker, cameras
 roslaunch smocap_gazebo demo_gazebo_cfg_z1.launch
# starts smocap
 rosrun smocap smocap_node.py _cameras:=camera_1 _detector_cfg_path:=`rospack find smocap`/params/gazebo_detector_cfg.yaml _run_mono_tracker:=false
# watch
rviz -d `rospack find smocap_viz`/smocap.rviz


 1.  Start gazebo and load a single (remotely controlled) infrared marker:
```
roslaunch smocap_gazebo gazebo_hog_marker.launch
``` 

###
closing the loop


roslaunch smocap_gazebo demo_gazebo_cfg_2.launch
rosrun smocap smocap_node.py _cameras:=camera_1 detector_cfg_path:=`rospack find smocap`/params/gazebo_detector_cfg.yaml _run_mono_tracker:=true _trap_losses:=false


~/work/oscar.git/oscar_simulator/scripts $ ./hog_simulator.py
~/work/overlay_ws/src/oscar/oscar_control/scripts $ ./pp_guidance_node.py


## Running real setup

 * have a fixed camera output an image topic in ROS
 








### Aruco


#### One marker, one camera
 1. starting gazebo spawning an hog aruco marker:
```
roslaunch smocap_gazebo gazebo_hog_marker.launch marker_urdf:=hog_chessboard.urdf.xacro marker_texture:=Aruco0
```
 2. spawning a single fixed camera in gazebo:
```
roslaunch smocap_gazebo  single_cam.launch pos_x:=0. pos_y:=0. pos_z:=3. rot_P:=1.5707963267948966
```

 3. launching the mocap node
```
rosrun smocap smocap_aruco_node.py _cameras:=camera_1
```

#### One marker, two cameras
 * second camera:
```
roslaunch smocap_gazebo  single_cam.launch camera_id:=2 pos_x:=1. pos_y:=1. pos_z:=2. rot_P:=1.5707963267948966
```
* mocap node with two cameras
```
rosrun smocap smocap_aruco_node.py _cameras:=camera_1,camera_2
```

