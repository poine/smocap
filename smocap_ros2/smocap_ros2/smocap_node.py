#
# ros2 run usb_cam usb_cam_node_exe
#

import rclpy
from rclpy.node import Node
import sensor_msgs.msg

from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2

# https://stoglrobotics.github.io/ros_team_workspace/master/use-cases/index.html

def get_default_cv_text_params(): return cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2

class MinimalPublisher(Node):

    def __init__(self, period_s=0.4):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(period_s, self.timer_callback)
        self.i = 0
        self.get_logger().info('Subscribing to video')
        self.subscription = self.create_subscription(sensor_msgs.msg.Image, '/image_raw', self.img_callback, 1)
        self.br = CvBridge()
        self.img_publisher_ = self.create_publisher(sensor_msgs.msg.Image, '/smocap/image_debug', 1)

    def img_callback(self, data):
        #self.get_logger().info('Receiving video frame')
        self.img = self.br.imgmsg_to_cv2(data)
    
    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World1: %d' % self.i
        self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
        debug_img = np.zeros((480, 640, 3), dtype=np.uint8)# self.img.copy()
        f, h, c, w = get_default_cv_text_params()
        x0, y0, txt = 10, 30, f'Hello {self.i}'
        cv2.putText(debug_img, txt, (x0, y0), f, h, c, w)
        self.img_publisher_.publish(self.br.cv2_to_imgmsg(debug_img, encoding="bgr8"))

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
