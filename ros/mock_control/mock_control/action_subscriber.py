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
import struct

from std_msgs.msg import ByteMultiArray

from .connection import ArrivalActionConnection

OUT_MSG_HEADER_FMT = "=ffb"
OUT_MSG_HEADER_LENGTH = struct.calcsize(OUT_MSG_HEADER_FMT)


class ActionSubscriber(Node):
    def __init__(self):
        super().__init__("action_subscriber")
        self.subscription = self.create_subscription(
            ByteMultiArray, "action", self.listener_callback, 10
        )
        self.subscription  # prevent unused variable warning
        self.arrival_action_conn = ArrivalActionConnection()

    def listener_callback(self, msg: ByteMultiArray):
        action = b"".join(msg.data)
        self.get_logger().info(
            f"Action recved: {struct.unpack(OUT_MSG_HEADER_FMT, action)}"
        )
        self.arrival_action_conn.send(action)


def main(args=None):
    rclpy.init(args=args)

    action_subscriber = ActionSubscriber()

    rclpy.spin(action_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    action_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
