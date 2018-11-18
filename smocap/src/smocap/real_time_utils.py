import os, threading, numpy as np, time, cv2
import rospy # remove me - profiler uses ros time types :(

####
#### Profiling tool

class Profiler:
    '''
    Records fps, skipped frames and processing duration for every video stream
    '''
    def __init__(self, nb_cam):
        self.last_frame_time = [None]*nb_cam
        self.processing_duration = [rospy.Duration.from_sec(0.)]*nb_cam
        self.frame_duration = [0.]*nb_cam
        self.fps = [0.]*nb_cam
        self.fps_lp = 0.7
        self.last_frame_seq = [0]*nb_cam
        self.skipped = [0]*nb_cam

        self.ff_duration = [1.]*nb_cam

    #def start(self, t):
    #    self.last_frame_time = [t]*len(self.last_frame_time)
        
        
    def signal_start(self, cam_idx, stamp, seq):
        if self.last_frame_time[cam_idx] is not None:
            self.frame_duration[cam_idx] = stamp - self.last_frame_time[cam_idx]
            if self.frame_duration[cam_idx].to_sec() > np.finfo(float).eps:
                self.fps[cam_idx] = self.fps_lp*self.fps[cam_idx] + (1-self.fps_lp)/self.frame_duration[cam_idx].to_sec()
            self.skipped[cam_idx] += seq - self.last_frame_seq[cam_idx] - 1
        self.last_frame_time[cam_idx] = stamp
        self.last_frame_seq[cam_idx] = seq
        
    def signal_done(self, cam_idx, stamp):
        if self.last_frame_time[cam_idx] is not None:
            self.processing_duration[cam_idx] = stamp - self.last_frame_time[cam_idx]

    def set_ff_duration(self, _cam_idx, _dur):
        self.ff_duration[_cam_idx] = _dur



####
#### 
class LossesTrapper:
    '''
    Save images and poses upon marker loss
    '''
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.lost_log_lock = threading.Lock()
        self.lost_log = []
        
    def record(self, cam_idx, img, pose):
        img_path = self.img_dir+'/img_{:03d}.png'.format(len(self.lost_log))
        print 'lost in cam {} at {} ({})'.format(cam_idx, pose[:3,3], img_path)
        with self.lost_log_lock:
            self.lost_log.append(pose[:3,3])
        cv2.imwrite(img_path, img)

    def stop(self):
        with open(os.path.join(self.img_dir, 'lost_log.npz'), 'wb') as f:
            np.savez(f, self.lost_log)



        
        
####
#### thread waiting on image delivery and running full frame localization ()
class FrameSearcher(threading.Thread):
    '''
    Encapsulates the threads responsible for searching markers in full frames at a lower framerate
    One for each camera
    
    '''
    def __init__(self, _cam_idx, _smocap, _profiler):
        super(FrameSearcher, self).__init__(name='FrameSearcher_{}'.format(_smocap.cameras[_cam_idx].name))
        self.cam_idx = _cam_idx
        self.smocap = _smocap
        self.profiler = _profiler
        self.condition = threading.Condition()
        self._quit = False
        # the processed image
        self.seq = None
        self.img = None
        self.stamp = None
        # start thread with mainloop in self.run
        self.start()
        

    def run(self):
        while not self._quit:
            with self.condition:
                self.condition.wait()
                if self.seq is not None:
                    #print 'working '+ self.name + " " + str(self.seq)
                    _start = time.time()
                    self.smocap.detect_markers_in_full_frame(self.img, self.cam_idx, self.stamp)
                    self.profiler.set_ff_duration(self.cam_idx, time.time() - _start)
                    self.seq = None # indicates image has been processed
                elif self._quit:
                    return
            
    def put_image(self, img, seq, stamp):
        if not self.condition.acquire(blocking=False):
            return
        else:
            self.seq = seq
            self.img = np.copy(img)
            self.stamp = stamp
            self.condition.notify()
            self.condition.release()
        
    def stop(self):
        with self.condition:
            self._quit = True
            self.condition.notify()

