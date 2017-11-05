#!/usr/bin/env python
import logging, sys, os, math, numpy as np, cv2, gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib, GObject
import matplotlib
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

import pdb

#http://www.learnopencv.com/blob-detection-using-opencv-python-c/
params_fields = ['blobColor',
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
                 'thresholdStep']

class GUI:
    def __init__(self):
        self.b = Gtk.Builder()
        gui_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_detector_gui.xml')
        self.b.add_from_file(gui_xml_path)
        self.window = self.b.get_object("window")
        self.window.set_title('BlobDetector')

        self.f = matplotlib.figure.Figure()
        self.ax = self.f.add_subplot(111)
        self.f.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0, wspace=0)

        self.canvas = FigureCanvas(self.f)
        self.b.get_object("alignment_img").add(self.canvas)

        grid = self.b.get_object("grid_params")
        self.labels = {}
        for i, p in enumerate(params_fields):
            label = Gtk.Label('{}'.format(p))
            label.set_justify(Gtk.Justification.LEFT)
            grid.attach(label, 0, i, 1, 1)
            self.labels[p] = Gtk.Label('')
            grid.attach(self.labels[p], 1, i, 1, 1)
        
        self.window.show_all()

    def display_image(self, img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img2)
        self.canvas.draw()

    def display_params(self, params):
        for i, p in enumerate(params_fields):
            self.labels[p].set_text('{}'.format(getattr(params, p)))

    def request_path(self, action):
        dialog = Gtk.FileChooserDialog("Please choose a file", self.window, action,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        ret = dialog.run()
        file_path = dialog.get_filename() if ret == Gtk.ResponseType.OK else None
        dialog.destroy()
        return file_path

class App:
    def __init__(self):
        self.gui = GUI()
        self.register_gui()

        self.params = cv2.SimpleBlobDetector_Params()
        self.params.minDistBetweenBlobs = 8
        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.gui.display_params(self.params)
        self.load_image("../test/image_12.png")

    def register_gui(self):
        self.gui.window.connect("delete-event", self.quit)
        self.gui.b.get_object("button1").connect("clicked", self.on_open_clicked)

    def on_open_clicked(self, b):
        path = self.gui.request_path(Gtk.FileChooserAction.OPEN)
        if path is not None: self.load_image(path)
        
    def load_image(self, path):
        self.img = cv2.imread(path)
        
        #self.gui.display_image(self.img)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_red_hue_range = np.array([0,  100,100]), np.array([10,255,255]) 
        upper_red_hue_range = np.array([160,100,100]), np.array([179,255,255]) 
        self.mask1 = cv2.inRange(self.hsv, *lower_red_hue_range)
        self.mask2 = cv2.inRange(self.hsv, *upper_red_hue_range)
        #mask = cv2.bitwise_or(mask1)
        self.masked = cv2.bitwise_and(self.img, self.img, mask=self.mask1)
        self.keypoints = self.detector.detect(255-self.mask1)

        self.img_res = cv2.drawKeypoints(self.img, self.keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.gui.display_image(self.img_res)
         
    def run(self):
        Gtk.main()

    def quit(self, a, b):
        Gtk.main_quit() 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=3, linewidth=300)
    App().run()

    
def test():
    im = cv2.imread("../test/image_14.png")
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    lower_red_hue_range = np.array([0,  100,100]), np.array([10,255,255]) 
    upper_red_hue_range = np.array([160,100,100]), np.array([179,255,255]) 
    mask1 = cv2.inRange(hsv, *lower_red_hue_range)
    mask2 = cv2.inRange(hsv, *upper_red_hue_range)
    #mask = cv2.bitwise_or(mask1)
    masked = cv2.bitwise_and(im, im, mask= mask1)

    #red_min, red_max = [17, 15, 100], [50, 56, 200]
    #mask = cv2.inRange(im, np.array(red_min, dtype = "uint8"), np.array(red_max, dtype = "uint8"))
    #masked = cv2.bitwise_and(im, im, mask = mask)
    
    params = cv2.SimpleBlobDetector_Params()
    fields = ['blobColor',
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
              'thresholdStep']
    for f in fields:
        print('{} -> {}'.format(f, getattr(params, f)))

    params.minDistBetweenBlobs = 8
    if 1:
        # Change thresholds
        params.minThreshold = 2;
        params.maxThreshold = 256;
        
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(255-mask1)
    #print keypoints
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    cv2.imshow("im", im)
    cv2.imshow("mask1", mask1)
    cv2.imshow("mask2", mask2)
    #cv2.imshow("masked", masked)
    cv2.imshow("Keypoints", im_with_keypoints)

    cv2.waitKey(0)



    
