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
import numpy as np
import struct
import threading

from std_msgs.msg import ByteMultiArray
from .connection import ArrivalPoseConnection

IN_MSG_HEADER_FMT = '=fbbffffffffffffdddffffffffffff'

class PosePublisher(Node):

    def __init__(self):
        super().__init__('pose_publisher')
        self.publisher_ = self.create_publisher(ByteMultiArray, 'pose', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.arrival_pose_connection = ArrivalPoseConnection()
        self.pose = None
        self.pose_thread = threading.Thread(target=self._recv_pose)
        self.pose_thread.start()

    def _recv_pose(self):
        while True:
            self.pose = self.arrival_pose_connection.recv()

    def timer_callback(self):
        msg = ByteMultiArray()
        if self.pose is None:
            self.get_logger().info("Waiting for pose from sim")
            return
        self.get_logger().info(f'Publishing pose: {struct.unpack(IN_MSG_HEADER_FMT, self.pose)}')
        msg.data = [bytes([i]) for i in list(self.pose)]
        self.publisher_.publish(msg)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    pose_publisher = PosePublisher()

    rclpy.spin(pose_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pose_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
