import smocap

class CameraSystem:

    def __init__(self, **kwargs):
        cam_names = kwargs.get('cam_names', ['camera_1'])
        self.cameras = [smocap.camera.Camera(cam_name) for cam_name in cam_names]
        
    
    def read_from_file(self, dir_path):
        pass
    


    def get_cameras(self):
        return self.cameras

    def get_camera(self, idx):
        return self.cameras[idx]
