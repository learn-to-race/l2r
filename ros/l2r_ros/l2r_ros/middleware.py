import rclpy
from rclpy.node import Node
import numpy as np
import struct

from . import connection
import threading
from std_msgs.msg import ByteMultiArray

IN_MSG_HEADER_FMT = '=fbbffffffffffffdddffffffffffff'
IN_MSG_HEADER_LENGTH = struct.calcsize(IN_MSG_HEADER_FMT)

class L2RMiddleware():
    def __init__(self):
        self.l2r_cam_connection = connection.Connection()
        self.l2r_pose_connection = connection.Connection()
        self.l2r_action_connection = connection.Connection()
        self.target_cam_connection = connection.Connection()
        self.target_pose_connection = connection.Connection()
        self.target_action_connection = connection.Connection()
    
    def proxy_action(self):
        while True:
            l2r_action = self.l2r_action_connection.recv()
            target_action = self.convert_action(l2r_action)
            self.target_action_connection.send(target_action)
            # possible delay for a while
    
    def proxy_pose(self):
        while True:
            target_obs = self.target_pose_connection.recv()
            l2r_obs = self.convert_pose(target_obs)
            self.l2r_pose_connection.send(l2r_obs)
            # possible delay for a while
    
    def proxy_cam(self):
        while True:
            target_obs = self.target_cam_connection.recv()
            l2r_obs = self.convert_camera(target_obs)
            self.l2r_cam_connection.send(l2r_obs)
            # possible delay for a while

    def convert_action(self, l2r_action):
        print("Not implemented")

    def convert_camera(self, target_camera):
        print("Not implemented")

    def convert_pose(self, target_pose):
        print("Not implemented")

    def start_threads(self):
        self.action_thread = threading.Thread(target=self.proxy_action)
        self.pose_thread = threading.Thread(target=self.proxy_pose)
        self.cam_thread = threading.Thread(target=self.proxy_cam)

        self.action_thread.start()
        self.pose_thread.start()
        self.cam_thread.start()
    
    def join(self):
        self.action_thread.join()
        self.pose_thread.join()
        self.cam_thread.join()



class ArrivalMiddleware(L2RMiddleware, Node):
    def __init__(self):
        Node.__init__("arrival_middleware")
        self.l2r_pose_connection = connection.L2RPoseConnection()
        self.l2r_cam_connection = connection.L2RCamConnection()
        self.l2r_action_connection = connection.L2RActionConnection()
        self.target_pose_connection = connection.ArrivalPoseConnection()
        self.target_cam_connection = connection.ArrivalCamConnection()
        self.target_action_connection = connection.ArrivalActionConnection()

    def convert_action(self, l2r_action):
        return l2r_action

    def convert_camera(self, target_camera):
        return target_camera

    def convert_pose(self, target_pose):
        return target_pose


class MockROS2Middleware(L2RMiddleware, Node):
    def __init__(self):
        Node.__init__(self, "mock_ros2_middleware")
        self.l2r_pose_connection = connection.L2RPoseConnection()
        self.l2r_cam_connection = connection.L2RCamConnection()
        self.l2r_action_connection = connection.L2RActionConnection()
        self.target_pose_connection = connection.ROS2SubConnection("pose_sub", "pose", ByteMultiArray)
        self.target_cam_connection = connection.ArrivalCamConnection()
        self.target_action_connection = connection.ROS2PubConnection("action_pub", "action", ByteMultiArray)
        self.spin_pose_thread = threading.Thread(target=self._spin_pose_conn)
        self.spin_pose_thread.start()

    def _spin_pose_conn(self):
        rclpy.spin(self.target_pose_connection)

    def convert_action(self, l2r_action):
        return [bytes([i]) for i in list(l2r_action)]

    def convert_camera(self, target_camera):
        return target_camera

    def convert_pose(self, target_pose):
        return b''.join(target_pose)
