# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

import threading

from std_msgs.msg import ByteMultiArray
from .connection import ArrivalCamConnection

class ImgPublisher(Node):

    def __init__(self):
        super().__init__('img_publisher')
        self.publisher_ = self.create_publisher(ByteMultiArray, 'image', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.arrival_cam_connection = ArrivalCamConnection()
        self.img = None
        self.cam_thread = threading.Thread(target=self._recv_img)
        self.cam_thread.start()

    def timer_callback(self):
        msg = ByteMultiArray()
        if self.img is None:
            self.get_logger().info("Waiting for img from sim")
            return
        msg.data = [bytes([i]) for i in list(self.img)]
        self.get_logger().info(f'Pub topic=image, len={len(msg.data)}')
        self.publisher_.publish(msg)
        self.i += 1
    
    def _recv_img(self):
        while True:
            self.img = self.arrival_cam_connection.recv()


def main(args=None):
    rclpy.init(args=args)

    img_publisher = ImgPublisher()

    rclpy.spin(img_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    img_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
