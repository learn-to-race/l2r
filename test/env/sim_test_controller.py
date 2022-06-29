import logging
import socket
import time
import unittest

import numpy as np

from l2r.constants import LAUNCHING_DELAY
from l2r.core import ActionInterface
from l2r.core import CameraConfig
from l2r.core import CameraInterface
from l2r.core import PoseInterface
from l2r.env import SimulatorController


class ControllerIntegrationTest(unittest.TestCase):
    """Integration tests which interact with the simulator"""

    logging.root.setLevel(logging.INFO)
    controller = SimulatorController(ip=socket.gethostbyname("arrival-sim"))

    # pose interface
    pose_if = PoseInterface()
    pose_if.start()

    # interface to send actions to simulator
    action_if = ActionInterface(ip=socket.gethostbyname("arrival-sim"))

    # front camera interface
    cam_width = 256
    cam_height = 192

    camera_if = CameraInterface(
        CameraConfig(
            name="CameraFrontRGB",
            Addr="tcp://0.0.0.0:8008",
            Width=cam_width,
            Height=cam_height,
            sim_addr="tcp://arrival-sim:8008",
        )
    )

    def test_1_controller_is_connected(self):
        """First test that the controller is connected to the simulator"""
        status = str(self.controller.conn.getstatus())
        self.assertTrue(status.startswith("1") or status.startswith("2"))

    def test_2_simulator_is_at_main_menu(self):
        """Assert that the simulator is at the main menu"""
        self.assertEqual(self.controller.get_level(), "MainMenu")

    def test_3_set_level(self):
        """Set level and validate successful"""
        self.controller.set_level("Thruxton")
        self.assertEqual(self.controller.get_level(), "thruxton")

    def test_4_enable_and_configure_camera(self):
        """Enable and configure a camera"""
        self.controller.enable_sensor(sensor_name=self.camera_if.camera_name)

        # Set parameters
        self.controller.set_sensor_params(
            sensor=self.camera_if.camera_name, params=self.camera_if.camera_param_dict
        )

        # Validate camera is on
        params = self.controller.get_sensor_params(sensor=self.camera_if.camera_name)
        self.assertTrue(params["enabled"])

        # Validate camera parameters are correct
        for param in params["parameters"]:
            if param["name"] == "Width":
                self.assertEqual(param["value"], self.cam_width)

            if param["name"] == "Height":
                self.assertEqual(param["value"], self.cam_height)

    def test_5_validate_receiving_images(self):
        """Validate images can be received"""
        self.camera_if.start()

        for _ in range(5):
            img = self.camera_if.get_data()

        # Assert correct dimesion
        self.assertEqual(img.shape, (self.cam_height, self.cam_width, 3))

        # Validate pixels aren't all same color
        self.assertNotEqual(np.std(img), 0)

    def test_6_configure_driver(self):
        """Test that the driver mode is successfully set"""
        self.controller.set_sensor_params(
            sensor="ArrivalVehicleDriver",
            params={
                "DriverAPIClass": "VApiUdp",
                "DriverAPI_UDP_SendAddress": socket.gethostbyname("l2r"),
                "InputSource": "AI",
            },
        )
        self.controller.set_mode_ai()
        time.sleep(LAUNCHING_DELAY)

        passed = 0
        for param in self.controller.get_vehicle_driver_params()["parameters"]:
            if param["name"] == "DriverAPIClass":
                self.assertEqual(param["value"], "VApiUdp")
                passed += 1

            if param["name"] == "bADModeInput":
                self.assertEqual(param["value"], True)
                passed += 1

            if param["name"] == "DriverAPI_UDP_SendAddress":
                self.assertEqual(param["value"], socket.gethostbyname("l2r"))
                passed += 1

        self.assertEqual(passed, 3)

    def test_7_action_pose_interface(self):
        """Validate ations can be sent to move vehicle. Note that we provide
        roughly 30 seconds for this to happen since the simulator can take
        some time to respond to actions"""
        for i in range(300):
            self.action_if.act([0.0, 1.0])
            time.sleep(0.1)
            pose = self.pose_if.get_data()

        # Assert vehicle is moving
        self.assertTrue(np.mean(pose[3:6] > 0.05))


if __name__ == "__main__":
    unittest.main()
