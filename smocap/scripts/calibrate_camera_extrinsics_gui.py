#!/usr/bin/env python
import argparse, logging, sys, os, math, numpy as np, cv2, gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib, GObject
import matplotlib
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import rospkg, yaml
import pdb

LOG = logging.getLogger('calibrate_extrinsics')
import calibrate_camera_extrinsics

class GUI:
    def __init__(self):
        self.b = Gtk.Builder()
        gui_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibrate_camera_extrinsics_gui.xml')
        self.b.add_from_file(gui_xml_path)
        self.window = self.b.get_object("window")
        self.window.set_title('CamExtrinsicCalibrator')

        self.f = matplotlib.figure.Figure()
        self.ax = self.f.add_subplot(111)
        self.f.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0, wspace=0)

        self.canvas = FigureCanvas(self.f)
        self.b.get_object("alignment_img").add(self.canvas)
        self.image_display_mode = None
        self.window.show_all()

    def display_image(self, model):
        label = self.b.get_object("label_image")
        label.set_text(model.image_path)
        
        img = model.get_image(self.image_display_mode)
        if len(img.shape) == 2:
            img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img2)
        self.canvas.draw()

        
class Model:
    def __init__(self, img_path, pts_path):
        self.load_image(img_path)

    def load_image(self, path):
        self.image_path = path
        self.img = cv2.imread(path)

    def get_image(self, which):
        return self.img

    def load_points(self,path):
        LOG.info(' loading points: {}'.format(path))
        self.pts_id, self.pts_img, self.pts_world = calibrate_camera_extrinsics.load_points(path)


        
class App:

    def __init__(self, img_path, cam_intr_path, pts):
        print img_path, cam_intr_path, pts
        self.gui = GUI()
        self.model = Model(img_path, pts)
        self.register_gui()
        self.load_image(img_path)

    def register_gui(self):
        self.gui.window.connect("delete-event", self.quit)

    def load_image(self, path):
        LOG.info(' loading image: {}'.format(path))
        self.model.load_image(path)
        self.gui.display_image(self.model)
        
    def run(self):
        Gtk.main()

    def quit(self, a, b):
        Gtk.main_quit() 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Calibrate camera extrinsics.')
    parser.add_argument('-i', '--img', default='/home/poine/work/smocap/smocap/test/enac_demo_z/cam1.png')
    parser.add_argument('-c', '--cam_intr', default='/home/poine/work/smocap/smocap/params/enac_demo_z/ueye_enac_z_1.yaml')
    parser.add_argument('-p', '--pts', default='/home/poine/work/smocap/smocap/test/enac_demo_z/cam1_floor_extrinsic_points.yaml')
    args = parser.parse_args()
    App(**{'img_path':args.img, 'cam_intr_path':args.cam_intr, 'pts':args.pts}).run()
