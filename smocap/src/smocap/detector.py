import numpy as np, cv2, yaml, scipy.spatial.distance, sklearn.cluster

class Detector:

    param_names = [
        'blobColor',
        'filterByArea',
        'filterByCircularity',
        'filterByColor',
        'filterByConvexity',
        'filterByInertia',
        'maxArea',
        'maxCircularity',
        'maxConvexity',
        'maxInertiaRatio',
        'maxThreshold',
        'minArea',
        'minCircularity',
        'minConvexity',
        'minDistBetweenBlobs',
        'minInertiaRatio',
        'minRepeatability',
        'minThreshold',
        'thresholdStep' ]
    
    
    def __init__(self, img_encoding, cfg_path=None):
        self.img_encoding = img_encoding
        if img_encoding in ['rgb8', 'bgr8']:
            self.lower_red_hue_range = np.array([0,  100,100]), np.array([10,255,255]) 
            self.upper_red_hue_range = np.array([160,100,100]), np.array([179,255,255])
        self.params = cv2.SimpleBlobDetector_Params()
        if cfg_path is not None:
            self.load_cfg(cfg_path)
            print(f'detector loading config from {cfg_path}')
        else:
            self.detector = cv2.SimpleBlobDetector_create(self.params)
            print(f'detector using default config (unlikely to work)')

    def detect(self, img, roi):
        if self.img_encoding == 'rgb8':
            hsv = cv2.cvtColor(img[roi], cv2.COLOR_RGB2HSV)
            self.sfc = cv2.inRange(hsv, *self.lower_red_hue_range)
            #mask2 = cv2.inRange(hsv, *self.upper_red_hue_range)
        elif self.img_encoding == 'bgr8':
            hsv = cv2.cvtColor(img[roi], cv2.COLOR_BGR2HSV)
            self.sfc = cv2.inRange(hsv, *self.lower_red_hue_range)
            #mask2 = cv2.inRange(hsv, *self.upper_red_hue_range)
        elif self.img_encoding == 'mono8':
            self.sfc = img[roi]
        keypoints = self.detector.detect(self.sfc)

        img_coords = np.array([kp.pt for kp in keypoints])
        if len(img_coords) > 0:
            img_coords += [roi[1].start, roi[0].start] 
        return keypoints, img_coords


    def detect_ff(self, img):
        h, w = img.shape[:2]
        roi = slice(0, h), slice(0, w)
        return self.detect(img, roi)


    def cluster_keypoints(self, kps_img, max_dist=50.):
        Y1 = scipy.spatial.distance.pdist(kps_img)
        Y2 = scipy.spatial.distance.squareform(Y1)
        db = sklearn.cluster.DBSCAN(eps=max_dist, min_samples=1, metric='precomputed')
        y_db = db.fit_predict(Y2)
        return y_db

        
    def load_cfg(self, path):
        with open(path, 'r') as stream:   
            d = yaml.load(stream, Loader=yaml.FullLoader)
        for k in d:
            setattr(self.params, k, d[k])
        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def save_cfg(self, path):
        d = {}
        for p in Detector.param_names:
            d[p] =  getattr(self.params, p)
        with open(path, 'w') as stream:
            yaml.dump(d, stream, default_flow_style=False)
    
    def update_param(self, name, value):
        setattr(self.params, name, value)
        self.detector = cv2.SimpleBlobDetector_create(self.params)

