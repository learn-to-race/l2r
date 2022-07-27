import logging
import random
import time
from typing import Any
from typing import List
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import gym
import numpy as np
from gym.spaces import Box

from l2r.core import ActionInterface
from l2r.core import CameraInterface
from l2r.core import PoseInterface
from l2r.constants import CAR_DIMS, N_SEGMENTS, OBS_DELAY
from l2r.utils.space import convert_ll_to_enu
from l2r.track import load_track

from .controller import SimulatorController
from .reward import GranTurismo
from .tracker import ProgressTracker


# Racetrack IDs
RACETRACKS = {
    "VegasNorthRoad": 0,
    "Thruxton": 1,
    "AngleseyNational": 2,
}

LEVEL_Z_DICT = {"Thruxton": 63.0, "VegasNorthRoad": 0.4, "AngleseyNational": 14.0}

COORD_MULTIPLIER = {"Thruxton": -1, "VegasNorthRoad": -1}


class RacingEnv(gym.Env):
    """A reinforcement learning environment for autonomous racing."""

    def __init__(
        self,
        controller: SimulatorController,
        action_interface: ActionInterface,
        camera_interfaces: List[CameraInterface],
        pose_interface: PoseInterface,
        observation_delay: float = OBS_DELAY,
        reward_kwargs: Dict[str, Any] = dict(),
        env_ip: str = "0.0.0.0",
    ):
        # Interfaces with the simulator
        self.controller = controller
        self.action_interface = action_interface
        self.camera_interfaces = camera_interfaces
        self.pose_interface = pose_interface
        self.reward = GranTurismo(**reward_kwargs)

        # delay between action and observation
        self.observation_delay = observation_delay

        # ip address of the env
        self.env_ip = env_ip

        # openAI gym compliance - action space
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

    def make(self, levels: List[str], evaluate: Optional[bool] = False):
        """This sequence of steps must be run when first interacting with the
        simulator - including if the simulator process was restarted. In particular,
        the cameras, sensors, and vehicle need to be configured.
        """
        logging.info("Making l2r environment")
        self.evaluate = evaluate

        # Set the level in the simulator
        self.levels = levels
        self.active_level = random.choice(levels)
        self.controller.set_level(self.active_level)

        # Start pose interface
        self.pose_interface.start()

        # Load active map
        self._load_map()

        # Camera configuration
        for camera_if in self.camera_interfaces:
            _ = self.controller.enable_sensor(camera_if.camera_name)
            _ = self.controller.set_sensor_params(
                sensor=camera_if.camera_name, params=camera_if.camera_param_dict
            )
            camera_if.start()

        # Configure driver
        self.controller.set_sensor_params(
            sensor="ArrivalVehicleDriver",
            params={
                "DriverAPIClass": "VApiUdp",
                "DriverAPI_UDP_SendAddress": self.env_ip,
                "InputSource": "AI",
            },
        )

        return self

    def step(self, action):
        """The primary method of the environment. Executes the desired action,
        receives the observation from the simulator, and evaluates termination
        conditions.

        :param dict action: the action and acceleration requests
        :return: observation, reward, done, info
        :rtype: if multimodal, the observation is a dict of numpy arrays with
          keys 'pose' and 'img' and shapes (30,) and (height, width, 3),
          respectively, otherwise the observation is just the image array.
          reward is of type float, done bool, and info dict
        """

        # Send the action via the action interface
        self.action_interface.act(action)

        # Receive data from the simulator
        observation = self._observe()
        _data = observation["pose"]

        # Check if the episode is complete
        done, info = self._is_complete()

        # Calculate reward of the current state
        reward = self.reward.get_reward(
            state=(_data, self.nearest_idx), oob_flag=info.get("oob", False)
        )

        return observation, reward, done, info

    def reset(
        self,
        level: Optional[str] = None,
        random_pos: Optional[bool] = False,
        segment_pos: Optional[bool] = True,
        evaluate: Optional[bool] = False,
    ):
        """Resets the vehicle to start position. A small time delay is used
        allow for the simulator to reset.

        :param str level: if specified, will set the simulator to this level,
          otherwise set to a random track
        :param bool random_pos: true/false for random starting position on the track
        :param bool segment_pos: true/false for track starting positions that adhere
          to segment boundaries
        :return: an intial observation as in the *step* method
        :rtype: see **step()** method
        """

        if level:
            new_level = level
            logging.info(f"[Env] Setting to level: {new_level}")
        elif self.levels:
            new_level = random.choice(self.levels)
            logging.info(f"[Env] New random level: {new_level}")
        else:
            new_level = self.level
            logging.info(f"[Env] Continuing with level: {new_level}")

        if new_level is self.active_level:
            self.controller.reset_level()

        else:
            self.active_level = new_level
            self.controller.set_level(self.active_level)
            self._load_map()

        self.controller.set_mode_ai()
        self.nearest_idx = None

        # set location
        # self.controller.set_location(coords, rot)
        self.tracker.wrong_way = False  # reset
        self.tracker.idx_sequence = [0] * 5  # reset

        # reset simulator sensors
        self.reward.reset()
        self.pose_interface.reset()

        for camera in self.camera_interfaces:
            camera.reset()

        # no delay is causing issues with the initial starting index
        self.poll_simulator()

        observation = self._observe()
        self.tracker.reset(start_idx=self.nearest_idx, segmentwise=segment_pos)

        # Evaluation mode
        self.evaluate = evaluate

        return observation

    def poll_simulator(self):
        """Poll the simulator until it receives an action"""
        logging.info("Polling simulator...")
        logging.info("Validating driver configuration for polling...")

        for _ in range(500):
            self.action_interface.act(action=(1.0, 1.0))
            if abs(self.pose_interface.get_data()[0]) > 0.05:
                logging.info("Successful")
                return
            time.sleep(0.1)

        raise Exception("Failed to connect to simulator")

    def render(self):
        """Not implmeneted. By default, the simulator provides a graphical
        interface, but can also be run on a server.
        """
        return self.imgs

    def _observe(self) -> Dict[str, Union[np.array, Dict[str, np.array]]]:
        """Perform an observation action by getting the most recent data from
        the pose and camera interfaces. To prevent observating immediately
        after executing an action, we include a small delay prior to actually
        requesting data from the sensor interfaces. Position coordinates are
        converted to a local ENU coordinate system to be consistent with the
        racetrack maps.

        :return: a tuple of numpy arrays (pose_data, images) with shapes
          (30,) and (height, width, 3), respectively
        :rtype: tuple
        """
        time.sleep(self.observation_delay)
        pose = self.pose_interface.get_data()
        self.imgs = {c.camera_name: c.get_data() for c in self.camera_interfaces}

        yaw = pose[12]
        bp = pose[22:25]
        a = pose[6:9]

        # provide racetrack ID in the observation
        pose[2] = RACETRACKS[self.active_level]

        # convert to local coordinate system
        x, y, z = pose[16], pose[15], pose[17]
        enu_x, enu_y, enu_z = convert_ll_to_enu(
            center=[x, y, z], ref_point=self.racetrack.ref_point
        )
        pose[16], pose[15], pose[17] = enu_x, enu_y, enu_z

        self.nearest_idx = self.racetrack.nearest_idx(np.asarray([enu_x, enu_y]))
        self.tracker.update(self.nearest_idx, enu_x, enu_y, enu_z, yaw, a, bp)

        return {"pose": pose, "images": self.imgs}

    def _is_complete(self):
        """Determine if the episode is complete. Termination conditions include
        car out-of-bounds, 3-laps successfully complete, not-moving-timeout,
        and max timesteps reached
        """
        return self.tracker.is_complete()

    def _load_map(self):
        """Load racetrack into a Racetrack object"""
        logging.info("Loading track")
        self.racetrack = load_track(level=self.active_level)

        self.tracker = ProgressTracker(
            n_indices=self.racetrack.n_indices,
            obs_delay=self.observation_delay,
            inner_track=self.racetrack.inside_path,
            outer_track=self.racetrack.outside_path,
            centerline=self.racetrack.centerline_arr,
            car_dims=CAR_DIMS,
            n_segments=N_SEGMENTS,
            segment_idxs=self.racetrack.local_segment_idxs,
            segment_tree=self.racetrack.segment_tree,
            eval_mode=self.evaluate,
            coord_multiplier=COORD_MULTIPLIER[self.active_level],
        )

        self.reward.set_track(
            inside_path=self.racetrack.inside_path,
            outside_path=self.racetrack.outside_path,
            centre_path=self.racetrack.centre_path,
            car_dims=CAR_DIMS,
        )

    def next_segment_start_location(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get spawn location at beginning of next segement"""
        if self.evaluate:
            segment_idx = self.tracker.current_segment
            segment_idx = segment_idx % (N_SEGMENTS)
        else:
            segment_idx = np.random.randint(
                0, len(self.racetrack.local_segment_idxs) - 1
            )

        pos = [0] * 4
        pos[0] = self.tracker.segment_coords["first"][segment_idx][0]  # x
        pos[1] = self.tracker.segment_coords["first"][segment_idx][1]  # y
        pos[2] = LEVEL_Z_DICT[self.active_level]  #
        pos[3] = self.racetrack.race_yaw[self.local_segment_idxs[segment_idx]]

        coords = {"x": pos[0], "y": pos[1], "z": pos[2]}
        rot = {"yaw": pos[3], "pitch": 0.0, "roll": 0.0}

        self.tracker.current_segment += 1

        print(
            "[Env] Spawning to {loc} segment start location".format(
                loc="next" if self.evaluate else "random"
            )
        )
        print(f"[Env] Current segment: {self.tracker.current_segment}")
        print(
            "[Env] Respawns: {n_spawns}; infractions: {n_infr}".format(
                n_spawns=self.tracker.respawns, n_infr=self.tracker.num_infractions
            )
        )
        print(f"[Env] Coords: {coords}")
        print(f"[Env] Rot: {rot}")

        return coords, rot
