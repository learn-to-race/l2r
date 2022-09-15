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

from std_msgs.msg import String, Float32MultiArray


class ActionSubscriber(Node):

    def __init__(self):
        super().__init__('action_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'action',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        if len(msg.data) == 2:
            self.get_logger().info('Action recved: "%s"' % msg.data)
        else:
            self.get_logger().info('Unexpected action: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    action_subscriber = ActionSubscriber()

    rclpy.spin(action_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    action_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
