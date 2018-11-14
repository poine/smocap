#!/usr/bin/env python
import logging, math, numpy as np, cv2
import smocap, utils
import pdb

LOG = logging.getLogger('test_on_one_image')


def test_identify(keypoints, keypoints_img):
    print keypoints_img
    

def test(img_path, img_enc, dtctr_cfg_path, cam_intr_path, cam_extr):

    try:
        camera_matrix, dist_coeffs, w, h = utils.load_camera_model(cam_intr_path)
    except:
        print('unable to read camera calibration: {}'.format(cam_intr_path))
        return

    cam = smocap.Camera("noname", encoding=img_enc)
    cam.set_calibration(camera_matrix, dist_coeffs, w, h)
    cam.set_location(*cam_extr)
    
    _smocap = smocap.SMoCap([cam], undistort=False, detector_cfg_path=dtctr_cfg_path)


    img = cv2.imread(img_path)
    if img is None:
        print('unable to read image: {}'.format(img_path))
        return
    else:
        LOG.info(' loaded image {}'.format(img_path)) 

        
    cam_id = 0
    _smocap.detect_markers_in_full_frame(img, cam_idx=cam_id, stamp=None)
    
    marker_id = 0
    _smocap.detect_marker_in_roi2(img, cam_idx=cam_id, stamp=None, marker_id=marker_id)
    marker = _smocap.markers[marker_id]
    _smocap.track_marker2( marker, cam_id, verbose=2)
    print  marker.world_to_irm_T

    marker_id = 1
    _smocap.detect_marker_in_roi2(img, cam_idx=cam_id, stamp=None, marker_id=marker_id)
    marker = _smocap.markers[marker_id]
    _smocap.track_marker2( marker, cam_id, verbose=2)
    print  marker.world_to_irm_T    
    
  
    debug_img = np.array(img)#cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # draw full frame observation
    for m in _smocap.markers:
        pt1 = (m.ff_observations[0].roi[1].start, m.ff_observations[0].roi[0].start)
        pt2 = (m.ff_observations[0].roi[1].stop, m.ff_observations[0].roi[0].stop)
        cv2.rectangle(debug_img, pt1 , pt2, (0, 255,255), thickness=1)
        pt1 = m.ff_observations[0].centroid
        pt2 = pt1 + 20*np.array([math.cos(m.ff_observations[0].orientation), math.sin(m.ff_observations[0].orientation)])
        cv2.line(debug_img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 255,255), thickness=2)



    cv2.imshow('image', debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = {
        'img_path':  '/home/poine/work/smocap.git/test/2_markers_diff_01.png',
        'img_enc':   'mono8',
        'dtctr_cfg_path': '/home/poine/work/smocap.git/params/f111_detector_default.yaml',
        'cam_intr_path': '/home/poine/work/smocap.git/test/camera_ueye_enac_ceiling_1_6mm.yaml',
        'cam_extr': [[1.34435324,  1.77016933,  2.82543134], [-0.70610063,  0.70809591,  0.00423334,  0.00204123]]
    }
    test(**args)

    
    #test('../test/gazebo_samples_no_distortion/image_00.png', '../test/gazebo_samples_no_distortion/camera_info.yaml', min_area=2)

    #test('../test/image_13.png', '../test/gazebo_samples_no_distortion/camera_info.yaml', min_area=2)
    #test('../test/ueye_ceiling_1_6mm_02.png', '../test/camera_ueye_enac_ceiling_1_6mm.yaml', min_area=12, detect_rgb=False)

