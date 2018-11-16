# Smocap
Simple ROS mocap node

Smocap uses fixed (wall or ceiling mounted) cameras to detect and localize moving markers.


<table>
  <tr>
  <td><img src="https://lh4.googleusercontent.com/Krs0lGRR-d27itPQVsvEPa2Y8CNcFzQyiw5EC-Wtf49iDOlMmXB-KBIYJGIUjqpqTskmwMb3jtMZOfQllg9a9MTqbuG-0_z8wcmCDFEqZ-3Tcr5TFOi0-G1rV7gV_cErPMQXAAJG" alt="Ceiling mounted camera" width="304" /></td>
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


## Running in simulation

 1.  Start gazebo and load a single (remotely controlled) infrared marker:
```
roslaunch smocap_gazebo gazebo_hog_marker.launch
``` 
 2. Load a camera in gazebo:
```
roslaunch smocap_gazebo single_cam.launch camera_id:=1 pos_x:=0. pos_y:=0. pos_z:=3. rot_P:=1.5707963267948966 fps=30.
 ``` 
 3. Start the mocap node:
``` 
rosrun smocap smocap_node.py _cameras:=camera_1 _detector_cfg_path:=/home/poine/work/smocap.git/smocap/params/gazebo_dettor_cfg.yaml _img_encoding:=rgb8
```
 
 4. View what's going on 
```
rviz
```



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
roslaunch smocap_gazebo  single_cam.launch camera_id:=2 pos_x:=1. pos_y:=1. pos_z:=3. rot_P:=1.5707963267948966
```
* mocap node with two cameras
```
rosrun smocap smocap_aruco_node.py _cameras:=camera_1,camera_2
```

