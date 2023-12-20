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

import carla


# Racetrack IDs
RACETRACKS = {
    "VegasNorthRoad": 0,
    "Thruxton": 1,
    "AngleseyNational": 2,
}

LEVEL_Z_DICT = {"Thruxton": 63.0, "VegasNorthRoad": 0.4, "AngleseyNational": 14.0}

COORD_MULTIPLIER = {"Thruxton": -1, "VegasNorthRoad": -1}


class RacingEnvCarla(gym.Env):
    """A reinforcement learning environment for autonomous racing."""

    def __init__(
        self
    ):
        # openAI gym compliance - action space
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        self.actor_list = []
        print('getting client')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2000.0)
        print('getting world')
        self.world = self.client.get_world()
        print('getting blueprint')
        self.blueprint_library = self.world.get_blueprint_library()
        print('get vehicle blueprint')
        self.bp = random.choice(self.blueprint_library.filter('vehicle'))
        print('get transform')
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        print('setting up rgb camera part 1')
        self.camera_img = {"CameraFrontRGB": np.zeros((384, 512, 3), dtype=np.uint8)}
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

    def get_camera_img(self, data):
        array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
        
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        if np.max(array) < 2:
            array = array * 255
        try:
            array = array[:384, :512, :]
        except Exception as e:
            pass
        self.camera_img = {"CameraFrontRGB": array}

    def make(self, levels: List[str], evaluate: Optional[bool] = False):
        
        print("Callling RacingEnvCarla.make, returning self...")

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

        throttle, brake, steer = 0.0, 0.0, min(max(-action[1], -1.0), 1.0)
        if action[0] > 0:
            throttle = min(1.00, action[0])
        if action[0] < 0:
            brake = min(1.00, abs(action[0]))
        print('create vehicle control')
        self.control = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        print('apply vehicle control')
        self.vehicle.apply_control(self.control)
        print('tick!')
        self.world.tick()

        return self._observe(), 0, False, None

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

        print('destroying actors')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print('spawn the vehicle')
        self.vehicle = self.world.spawn_actor(self.bp, self.transform)
        print('add vehicle to actor_list')
        self.actor_list.append(self.vehicle)
        print('setting up rgb camera part 2')
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        print('setting up listen method')
        self.camera.listen(lambda image: self.get_camera_img(image))

        return self._observe()

    # def poll_simulator(self):
    #     """Poll the simulator until it receives an action"""
    #     logging.info("Polling simulator...")
    #     logging.info("Validating driver configuration for polling...")

    #     for _ in range(500):
    #         self.action_interface.act(action=(1.0, 1.0))
    #         if abs(self.pose_interface.get_data()[0]) > 0.05:
    #             logging.info("Successful")
    #             return
    #         time.sleep(0.1)

    #     raise Exception("Failed to connect to simulator")

    def render(self):
        """Not implmeneted. By default, the simulator provides a graphical
        interface, but can also be run on a server.
        """
        return self.camera_img

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
        
        pose = np.zeros(30)

        # bp = pose[22:25]
        # a = pose[6:9]

        vc = self.vehicle.get_control()
        pose[12] = vc.steer
        lc = self.vehicle.get_location()
        pose[16], pose[15], pose[17] = lc.x, lc.y, lc.z

        return {"pose": pose, "images": self.camera_img}

    # def _is_complete(self):
    #     """Determine if the episode is complete. Termination conditions include
    #     car out-of-bounds, 3-laps successfully complete, not-moving-timeout,
    #     and max timesteps reached
    #     """
    #     return self.tracker.is_complete()

    # def _load_map(self):
    #     """Load racetrack into a Racetrack object"""
    #     logging.info("Loading track")
    #     self.racetrack = load_track(level=self.active_level)

    #     self.tracker = ProgressTracker(
    #         n_indices=self.racetrack.n_indices,
    #         obs_delay=self.observation_delay,
    #         inner_track=self.racetrack.inside_path,
    #         outer_track=self.racetrack.outside_path,
    #         centerline=self.racetrack.centerline_arr,
    #         car_dims=CAR_DIMS,
    #         n_segments=N_SEGMENTS,
    #         segment_idxs=self.racetrack.local_segment_idxs,
    #         segment_tree=self.racetrack.segment_tree,
    #         eval_mode=self.evaluate,
    #         coord_multiplier=COORD_MULTIPLIER[self.active_level],
    #     )

    #     self.reward.set_track(
    #         inside_path=self.racetrack.inside_path,
    #         outside_path=self.racetrack.outside_path,
    #         centre_path=self.racetrack.centre_path,
    #         car_dims=CAR_DIMS,
    #     )

    # def next_segment_start_location(self) -> Tuple[Dict[str, float], Dict[str, float]]:
    #     """Get spawn location at beginning of next segement"""
    #     if self.evaluate:
    #         segment_idx = self.tracker.current_segment
    #         segment_idx = segment_idx % (N_SEGMENTS)
    #     else:
    #         segment_idx = np.random.randint(
    #             0, len(self.racetrack.local_segment_idxs) - 1
    #         )

    #     pos = [0] * 4
    #     pos[0] = self.tracker.segment_coords["first"][segment_idx][0]  # x
    #     pos[1] = self.tracker.segment_coords["first"][segment_idx][1]  # y
    #     pos[2] = LEVEL_Z_DICT[self.active_level]  #
    #     pos[3] = self.racetrack.race_yaw[self.local_segment_idxs[segment_idx]]

    #     coords = {"x": pos[0], "y": pos[1], "z": pos[2]}
    #     rot = {"yaw": pos[3], "pitch": 0.0, "roll": 0.0}

    #     self.tracker.current_segment += 1

    #     print(
    #         "[Env] Spawning to {loc} segment start location".format(
    #             loc="next" if self.evaluate else "random"
    #         )
    #     )
    #     print(f"[Env] Current segment: {self.tracker.current_segment}")
    #     print(
    #         "[Env] Respawns: {n_spawns}; infractions: {n_infr}".format(
    #             n_spawns=self.tracker.respawns, n_infr=self.tracker.num_infractions
    #         )
    #     )
    #     print(f"[Env] Coords: {coords}")
    #     print(f"[Env] Rot: {rot}")

    #     return coords, rot
