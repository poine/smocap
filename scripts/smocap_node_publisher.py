import sys , math, numpy as np, rospy, cv2, cv_bridge
import std_msgs.msg, sensor_msgs.msg, geometry_msgs.msg
import visualization_msgs.msg

import utils

import pdb
def make_2d_line(p0, p1, spacing=200, endpoint=True):
    dist = np.linalg.norm(p1-p0)
    n_pt = dist/spacing
    if endpoint: n_pt += 1
    return np.stack([np.linspace(p0[j], p1[j], n_pt, endpoint=endpoint) for j in range(2)], axis=-1)


def test_proj(orig_img_pts, world_pts_cam, cam):
    #print world_pts_cam
    world_to_cam_r ,world_to_cam_t = np.array([0., 0, 0]), np.array([0., 0, 0])
    proj_img_pts = cv2.projectPoints(world_pts_cam, world_to_cam_r, world_to_cam_t, cam.K, cam.D)[0]
    for i in range(len(orig_img_pts)):
        print orig_img_pts[i], world_pts_cam[i], proj_img_pts[i]

def get_points_on_plane(rays, plane_n, plane_d):
    return np.array([-plane_d/np.dot(ray, plane_n)*ray for ray in rays])

    
        
class SmocapNodePublisher:
    def __init__(self, smocap):
        self.cams_fov_pub = rospy.Publisher('/smocap/cams_fov', visualization_msgs.msg.MarkerArray, queue_size=1)
        self.draw_cameras_fov(smocap)
        self.rois_pub =  rospy.Publisher('/smocap/cams_rois', visualization_msgs.msg.MarkerArray, queue_size=1)
        self.status_pub = rospy.Publisher('/smocap/status', std_msgs.msg.String, queue_size=1)
        
    def draw_cameras_fov(self, _smocap):
       
        self.cam_fov_msg = visualization_msgs.msg.MarkerArray()
        for idx_cam, (cam) in enumerate(_smocap.cameras):
        
            img_corners = np.array([[0., 0], [cam.w, 0], [cam.w, cam.h], [0, cam.h], [0, 0]])
            borders_img = np.zeros((0,2))
            for i in range(len(img_corners)-1):
                borders_img = np.append(borders_img, make_2d_line(img_corners[i], img_corners[i+1], endpoint=True), axis=0)
            # ideal border of image ( aka undistorted ) in pixels
            borders_undistorted = cv2.undistortPoints(borders_img.reshape((1, len(borders_img), 2)), cam.K, cam.D, None, cam.K)
            # border of image in optical plan
            borders_cam = [np.dot(cam.invK, [u, v, 1]) for (u, v) in borders_undistorted.squeeze()]
            # border of image projected on floor plane (in cam frame)
            borders_floor_plane_cam = get_points_on_plane(borders_cam, cam.fp_n, cam.fp_d)
            # border of image projected on floor plane (in world frame)
            borders_floor_plane_world = np.array([np.dot(cam.cam_to_world_T[:3], p.tolist()+[1]) for p in borders_floor_plane_cam])
                
            marker = visualization_msgs.msg.Marker()
            marker.header.frame_id = "world"
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.id = idx_cam
            marker.text = cam.name
            marker.scale.x = 0.01
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = 0
            marker.pose.position.y = 0
            marker.pose.position.z = 0
            for x, y, z in borders_floor_plane_world:
                p1 = geometry_msgs.msg.Point()
                p1.x=x; p1.y=y;p1.z=z
                marker.points.append(p1)
            self.cam_fov_msg.markers.append(marker)
            #test_proj(borders_img, borders_cam, cam)

            
    def publish_camera_fov(self):
        self.cams_fov_pub.publish(self.cam_fov_msg)


    def publish_rois(self, _smocap):
        pass

    def publish_status(self, _timer, _smocap):
        txt = '\nTime: {}\n'.format(rospy.Time.now().to_sec())
        txt += 'Timing:\n'
        for i, (fps, skipped) in enumerate(zip(_timer.fps, _timer.skipped)):
            txt += (' camera {}: fps {:4.1f} skipped {:d}\n'.format(i, fps, skipped))
        txt += 'Markers:\n'
        txt += ' marker 0:\n'
        txt += '   localized: {}\n'.format(_smocap.marker.is_localized)
        txt += '   observations: {}\n'.format([o.roi is not None for o in _smocap.marker.observations] )
        t_w_to_m = ' '.join(['{:6.3f}'.format(p) for p in _smocap.marker.world_to_irm_T[:3,3]])
        a_w_to_m = math.atan2(_smocap.marker.world_to_irm_T[0,1], _smocap.marker.world_to_irm_T[0,0])
        txt += '   pose {} m {:5.2f} deg'.format(t_w_to_m, utils.deg_of_rad(a_w_to_m))
        self.status_pub.publish(txt)

    def publish_test(self, _smocap):
        print 'irm_to_world\n', _smocap.marker.irm_to_world_T
        print _smocap.marker.is_in_frustum(_smocap.cameras[0])
        print _smocap.marker.is_in_frustum(_smocap.cameras[1])
        print
        


    def publish(self, _timer, _smocap):
        self.publish_camera_fov()
        self.publish_status(_timer, _smocap)
