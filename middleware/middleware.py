import connection
import threading

class L2RMiddleware:
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



class ArrivalMiddleware(L2RMiddleware):
    def __init__(self):
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