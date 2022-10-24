import logging
import socket
import unittest

from l2r import build_env
from l2r import RacingEnv


L2R_HOST = socket.gethostbyname("l2r")
ARRIVAL_SIM_HOST = socket.gethostbyname("arrival-sim")


class EnvIntegrationTest(unittest.TestCase):
    # Integration tests which interact with the simulator

    logging.root.setLevel(logging.INFO)

    def test_racing_env(self):
        # First build the environmeny
        logging.info("Attempting to build environment")
        env = build_env(
            levels=["Thruxton"],
            controller_kwargs={"ip": ARRIVAL_SIM_HOST},
            camera_cfg=[
                {
                    "name": "CameraFrontRGB",
                    "Addr": "tcp://0.0.0.0:8008",
                    "Width": 256,
                    "Height": 192,
                    "sim_addr": f"tcp://{ARRIVAL_SIM_HOST}:8008",
                }
            ],
            action_cfg={"ip": ARRIVAL_SIM_HOST},
            env_ip=L2R_HOST,
            env_kwargs={
                "multimodal": True,
                "eval_mode": True,
                "n_eval_laps": 1,
                "max_timesteps": 5000,
                "obs_delay": 0.1,
                "not_moving_timeout": 100,
                "reward_pol": "custom",
                "provide_waypoints": False,
                "active_sensors": ["CameraFrontRGB", "CameraRightRGB", "CameraLeftRGB"],
                "vehicle_params": False,
            },
        )

        self.assertIsInstance(env, RacingEnv)
        observation = env.reset()

        # Validate driver is appropriately configured
        logging.info("Validating driver configuration...")
        for param in env.controller.get_vehicle_driver_params()["parameters"]:
            if param["name"] == "DriverAPIClass":
                self.assertEqual(param["value"], "VApiUdp")

            if param["name"] == "bADModeInput":
                self.assertEqual(param["value"], True)

            if param["name"] == "DriverAPI_UDP_SendAddress":
                self.assertEqual(param["value"], L2R_HOST)

        # Reset and validate observation
        logging.info(observation)
        logging.info(observation["pose"])
        logging.info(observation["images"])


if __name__ == "__main__":
    unittest.main()
