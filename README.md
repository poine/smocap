# Smocap

## 1. Introduction
Smocap is a simple ROS mocap node. It uses fixed (wall or ceiling mounted) cameras to detect and localize moving markers.


<table>
  <tr>
  <td><img src="https://lh4.googleusercontent.com/Krs0lGRR-d27itPQVsvEPa2Y8CNcFzQyiw5EC-Wtf49iDOlMmXB-KBIYJGIUjqpqTskmwMb3jtMZOfQllg9a9MTqbuG-0_z8wcmCDFEqZ-3Tcr5TFOi0-G1rV7gV_cErPMQXAAJG" alt="Ceiling mounted camera" width="470" /></td>
  <td>
<a href="https://youtu.be/rcXm4QCaq64"><img src="http://img.youtube.com/vi/rcXm4QCaq64/0.jpg" width="300" /></a>
<a href="https://youtu.be/X8M0IHWhTcs"><img src="http://img.youtube.com/vi/X8M0IHWhTcs/0.jpg" width="300" /></a>
<a href="https://youtu.be/_u4qhHbuV6Q"><img src="http://img.youtube.com/vi/_u4qhHbuV6Q/0.jpg" width="300" /></a>
<a href="https://youtu.be/LCjn09UPtN0"><img src="http://img.youtube.com/vi/LCjn09UPtN0/0.jpg" width="300" /></a>
  </td>
  </tr>
  <tr>
	<td>Ceiling mounted camera</td> 	
	<td>Action videos</td> 
  </tr>
  </table>

[Documentation index](https://poine.github.io/smocap/)

## 2. Quickstart

### 2.1 Building the package

Read [Installing from source](http://wiki.ros.org/Installation/Source) or

  1. Install ROS, then create a catkin workspace:
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
cd ~/catkin_ws
catkin_make
```

  2. Download sources:
```
cd ~/catkin_ws/src
git clone https://github.com/poine/smocap.git
```

  3. Build the package
```
cd ~/catkin_ws/
catkin_make
```


### 2.2 Running a simulation

 1.  Start gazebo and load a single (remotely controlled) infrared marker:
```
roslaunch smocap_gazebo gazebo_hog_marker.launch
``` 
 2. Load a camera in gazebo:
```
roslaunch smocap_gazebo single_cam.launch camera_id:=1 pos_x:=0. pos_y:=0. pos_z:=3. rot_P:=1.5707963267948966 fps:=30.
 ``` 
 3. Start the mocap node:
``` 
rosrun smocap smocap_node.py _cameras:=camera_1 _detector_cfg_path:=`rospack find smocap`/params/gazebo_detector_cfg.yaml _img_encoding:=rgb8
```
 
 4. View what's going on 
```
rviz
```
look for topics staring with smocap...
