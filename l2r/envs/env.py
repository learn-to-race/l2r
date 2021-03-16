# ========================================================================= #
# Filename:                                                                 #
#    env.py                                                                 #
#                                                                           #
# Description:                                                              # 
#    Reinforcement learning environment for autonomous racing               #
# ========================================================================= #

import json
import os
import pathlib
import time

import gym
import matplotlib.path as mplPath
import numpy as np
from gym.spaces import Box, Discrete, Dict
from scipy.spatial import KDTree

import envs.utils as utils
from baselines.reward import CustomReward
from core.controller import SimulatorController
from envs.reward import GranTurismo
from core.tracker import ProgressTracker
from racetracks.mapping import level_2_trackmap

# Simulator Lag Delay
MEDIUM_DELAY = 3
TIMEOUT_DELAY = 30
LAUNCHING_DELAY = 15

# Restart simulator container every so often
SIM_RESET=60*20

# Vehicle dimensions in meters
CAR_DIMS = [3.0, 1.68]

# Raw action space boundaries
MIN_STEER_REQ = -1.0
MAX_STEER_REQ = 1.0
STEER_REQ_RANGE = MAX_STEER_REQ - MIN_STEER_REQ

MIN_ACC_REQ = -16.
MAX_ACC_REQ = 6.
ACC_REQ_RANGE = MAX_ACC_REQ - MIN_ACC_REQ

NEUTRAL_GEAR = 0
DRIVE_GEAR = 1
REVERSE_GEAR = 2
PARK_GEAR = 3
GEAR_REQ_RANGE = 4

# Pose observation space boundaries
MIN_OBS_ARR = [
    -1., -1., -1.,                  # steering, gear, mode
    -200., -200., -10.,             # velocity
    -100., -100., -100.,            # acceleration
    -1., -1., -5.,                  # angular velocity
    -6.2832, -6.2832, -6.2832,      # yaw, pitch, roll
    -2000., 2000., 2000.,           # location coordinates in the format (y, x, z)
    -2000., -2000., -2000., -2000., # rpm (per wheel)
    -1., -1., -1., -1.,             # brake (per wheel)
    -1., -1., -1300., -1300.]       # torq (per wheel)

MAX_OBS_ARR = [
    1., 4., 1.,                  # steering, gear, mode
    200., 200., 10.,             # velocity
    100., 100., 100.,            # acceleration
    1., 1., 5.,                  # angular velocity
    6.2832, 6.2832, 6.2832,      # yaw, pitch, roll
    2000., 2000., 2000.,         # location coordinates in the format (y, x, z)
    2500., 2500., 2500., 2500.,  # rpm (per wheel)
    1., 1., 2., 2.,              # brake (per wheel)
    1., 1., 1300., 1300.         # torq (per wheel)
]

class RacingEnv(gym.Env):
    """A reinforcement learning environment for autonomous racing.

    :param max_timesteps: maximimum number of timesteps per episode
    :type max_timesteps: int
    :param controller_kwargs: keyword args for the simulator controller
    :type controller_kwargs: dict
    :param reward_kwargs: keyword args the reward policy
    :type reward_kwargs: dict
    :param action_if_kwargs: keyword args for the action interface
    :type action_if_kwargs: dict
    :param camera_if_kwargs: keyword args for the camera interface
    :type camera_if_kwargs: dict
    :param pose_if_kwargs: keyword args for the position receive interface
    :type pose_if_kwargs: dict
    :param logger_kwargs: keyword args for the logger
    :type logger_kwargs: dict
    :param reward_pol: reward policy to use, defaults to GranTurismo
    :type reward_pol: str, optional
    :param not_moving_timeout: terminate if not moving for this many timesteps
    :type not_moving_timeout: int, optional
    :param provide_waypoints: flag to provide ground-truth, future waypoints
      on the track
    :type provide_waypoints: bool, optional
    :param obs_delay: time delay between action and observation
    :type obs_delay: float, optional
   """
    def __init__(self, max_timesteps, controller_kwargs, reward_kwargs,
                 action_if_kwargs, camera_if_kwargs, pose_if_kwargs,
                 logger_kwargs, reward_pol='default', not_moving_timeout=20,
                 provide_waypoints=False, obs_delay=0.05):
        self.controller = SimulatorController(**controller_kwargs)
        self.action_if = utils.ActionInterface(**action_if_kwargs)
        self.camera_if = utils.CameraInterface(**camera_if_kwargs)
        self.pose_if = utils.PoseInterface(**pose_if_kwargs)
        self.reward = GranTurismo(**reward_kwargs) if reward_pol == 'default' \
            else CustomReward(**reward_kwargs)
        self.max_timesteps = max_timesteps
        self.not_moving_timeout = not_moving_timeout
        self.observation_delay = obs_delay
        self.provide_waypoints = provide_waypoints
        self.last_restart = time.time()

        # openAI gym compliance - action space
        self.action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float64)

    def make(self, level, multimodal, camera_params, sensors, driver_params,
             vehicle_params=None, use_lidar=False, multi_agent=False,
             remake=False):
        """Unlike many environments, make does not start the simulator process.
        It does, however, configure the simulator's settings. The simulator
        process must be running prior to calling this method otherwise an error
        will occur when trying to establish a connection with the simulator.

        :param level: the desired racetrack map
        :type level: string
        :param multimodal: if false, then the agent is 'visual only' and only
          receives pixel values, if true, the agent also has access to pose data
        :type multimodal: boolean
        :param camera_params: camera parameters to set
        :type camera_params: dict
        :param sensors: sensors to enable for the simulation
        :type sensors: list
        :param driver_params: driver parameters to modify
        :type drive_params: dict
        :param vehicle_params: vehicle parameters to set
        :type vehicle_params: dict, optional
        :param use_lidar: activate all lidar sensors
        :type use_lidar: boolean, optional
        :param multi_agent: not currently supported
        :type multi_agent: boolean, optional
        :param remake: if remaking, reset the camera interface
        :type remake: boolean, optional
        """
        self.camera_dims = {
            'width': camera_params['Width'],
            'height': camera_params['Height']
        }
        self.level = level
        self.sensors = sensors
        self.use_lidar = use_lidar
        self.multimodal = multimodal
        self.multi_agent = multi_agent
        self.vehicle_params = vehicle_params
        self.camera_params = camera_params
        self.driver_params = driver_params
        
        self.controller.set_level(level)
        self.controller.reset_vehicle_params()
        self.controller.set_api_udp()
        self._load_map(level)

        self.controller.set_sensor_params(
            sensor='CameraFrontRGB',
            params=camera_params
        )
        self.controller.set_sensor_params(
            sensor='ArrivalVehicleDriver',
            params=driver_params
        )

        if remake:
            self.camera_if.reconnect()
        else:
            _shape = (camera_params['Width'], camera_params['Height'], 3)
            self.pose_if.start()
            self.camera_if.start(img_dims=_shape)

    def _restart_simulator(self):
        """Periodically need to restart the container for long running training
        """
        print('[RacingEnv] Periodic simulator restart')
        self.controller.restart_simulator()
        self.make(
            level=self.level,
            multimodal=self.multimodal,
            camera_params=self.camera_params,
            sensors=self.sensors,
            driver_params=self.driver_params,
            vehicle_params=self.vehicle_params,
            use_lidar=self.use_lidar,
            multi_agent=self.multi_agent,
            remake=True
        )

    def _check_restart(self, done):
        """Check if we should restart the simulator
        """
        if not done:
            return

        if time.time() - self.last_restart > SIM_RESET:
            self.last_restart = time.time()
            self._restart_simulator()

    def step(self, action):
        """The primary method of the environment. Executes the desired action,
        receives the observation from the simulator, and evaluates termination
        conditions.

        :param action: the action and acceleration requests (gear is optional)
        :type action: dict
        :return: observation, reward, done, info
        :rtype: if multimodal, the observation is a dict of numpy arrays with
          keys 'pose' and 'img' and shapes (30,) and (height, width, 3),
          respectively, otherwise the observation is just the image array. 
          reward is of type float, done boolean, and info dict
        """
        self.action_if.act(action)
        _observation = self._observe()
        _data, _img = _observation
        observation = _observation if self.multimodal else _img
        done, info = self._is_complete(_observation)
        reward = self.reward.get_reward(
            state=(_data, self.nearest_idx),
            oob_flag=info['oob']
        )
        _ = self._check_restart(done)

        if self.provide_waypoints:
            info['track_idx'] = self.nearest_idx
            info['waypoints'] = self._waypoints()
            
        return observation, reward, done, info

    def reset(self, random_pos=False):
        """Resets the vehicle to start position. A small time delay is used
        allow for the simulator to reset.

        :param random_pos: true/false for random starting position on the
          track. the yaw of the vehicle will align with the centerline
        :type random_pos: boolean
        :return: an intial observation as in the *step* method
        :rtype: see *step* method
        """
        self.controller.reset_level()
        self.nearest_idx, info = None, {}

        # give the simulator time to reset
        time.sleep(MEDIUM_DELAY)

        # randomly initialize starting location
        if random_pos:
            raise NotImplementedError('This feature causes unknown instability.')
            coords, rot = self.random_start_location()
            self.controller.set_location(coords, rot)
            time.sleep(MEDIUM_DELAY)

        # reset simulator sensors
        self.controller.set_mode_ai()
        self.controller.enable_sensor('CameraFrontRGB')

        if self.vehicle_params:
            self.controller.set_vehicle_params(vehicle_params)

        for sensor in self.sensors:
            self.controller.enable_sensor(sensor)

        if not self.use_lidar:
            for i in range(1, 5):
                self.controller.disable_sensor(f'iBeoLidar{i}')

        if not self.multi_agent:
            self.controller.disable_sensor('V2VHub')

        self.reward.reset()
        self.pose_if.reset()
        self.camera_if.reset()

        # no delay is causing issues with the initial starting index
        time.sleep(MEDIUM_DELAY)

        _observation = self._observe()
        _data, _img = _observation
        observation = _observation if self.multimodal else _img
        self.tracker.reset(start_idx=self.nearest_idx)

        if self.provide_waypoints:
            info['waypoints'] = self._waypoints()
            info['track_idx'] = self.nearest_idx

        return observation, info

    def render(self):
        """Not implmeneted. The simulator, by default, provides a graphical
        interface, but can also be run on a server.
        """
        raise NotImplementedError

    @property
    def multimodal(self):
        """Getter method for the multimodal property.

        :return: true if the environment is set to multimodal, false otherwise
        :rtype: boolean
        """
        return self._multimodal

    @multimodal.setter
    def multimodal(self, value):
        """Setter method for the multimodal property. Changing this value will
        cause the environment's observation space to change.

        :param value: value to self the multimodal property to. true sets the
          environment to multimodal and makes the observation space a
          a dictionary of the camera images and the pose data. false is visual
          only features.
        :type value: boolean
        """
        if not isinstance(value, bool):
            raise TypeError('Multimodal property must be of type: bool')

        self._multimodal = value
        _shape = (self.camera_dims['height'], self.camera_dims['width'], 3)
        _img_space = Box(low=0, high=255, shape=_shape, dtype=np.uint8)

        if self._multimodal:
            self.observation_space = Dict({
                'pose': Box(
                    low=np.array(MIN_OBS_ARR),
                    high=np.array(MAX_OBS_ARR)
                ),
                'img': _img_space
            })
        else:
            self.observation_space = _img_space

    def _observe(self):
        """Perform an observation action by getting the most recent data from
        the pose and camera interfaces. To prevent observating immediately
        after executing an action, we include a small delay prior to actually
        requesting data from the sensor interfaces. Position coordinates are
        converted to a local ENU coordinate system to be consistent with the
        racetrack maps.
        :return: a tuple of np arrays (pose_data, images) with shapes
          (30,) and (height, width, 3), respectively
        :rtype: tuple of numpy arrays
        """
        time.sleep(self.observation_delay)
        pose = self.pose_if.get_data()
        imgs = self.camera_if.get_data()

        yaw = pose[12]
        bp = pose[22:25]
        a = pose[6:9]

        # convert to local coordinate system
        x, y, z = pose[16], pose[15], pose[17]
        enu_x, enu_y, enu_z = self.geo_location.convert_to_ENU((x, y, z))
        pose[16], pose[15], pose[17] = enu_x, enu_y, enu_z

        self.nearest_idx = self.kdtree.query(np.asarray([enu_x, enu_y]))[1]
        self.tracker.update(self.nearest_idx, enu_x, enu_y, enu_z, yaw, a, bp)
        
        return (pose, imgs)

    def _is_complete(self, observation):
        """Determine if the episode is complete. Termination conditions include
        car out-of-bounds, 3-laps successfully complete, not-moving-timeout,
        and max timesteps reached
        """
        return self.tracker.is_complete()

    def _load_map(self, level):
        """Loads the racetrack map from a data file. The map is parsed into
        numerous arrays and matplotlib Path objects representing the inside
        and outside track boundaries along with the centerline.

        :param level: the racetrack name. must be in [...]
        :type level: str
        """
        map_file, self.random_poses = level_2_trackmap(level)

        with open(os.path.join(pathlib.Path().absolute(), map_file), 'r') as f:
            self.original_map = json.load(f)
            self.ref_point = self.original_map["ReferencePoint"]

        _out = np.asarray(self.original_map['Outside'])
        _in = np.asarray(self.original_map['Inside'])

        # self.outside_arr = np.asarray(self.original_map['Outside'])[:, :-1]
        # self.inside_arr = np.asarray(self.original_map['Inside'])[:, :-1]
        self.outside_arr = _out if _out.shape[-1] == 2 else _out[:, :-1]
        self.inside_arr = _in if _in.shape[-1] == 2 else _in[:, :-1]
        self.centerline_arr = np.asarray(self.original_map['Centre'])

        self.centre_path = mplPath.Path(self.centerline_arr)
        self.outside_path = mplPath.Path(self.outside_arr)
        self.inside_path = mplPath.Path(self.inside_arr)

        self.geo_location = utils.GeoLocation(self.ref_point)
        self.n_indices = len(self.centerline_arr)
        self.kdtree = KDTree(self.centerline_arr)

        self.tracker = ProgressTracker(
            n_indices=len(self.centerline_arr),
            obs_delay=self.observation_delay,
            max_timesteps=self.max_timesteps,
            inner_track=self.inside_path,
            outer_track=self.outside_path,
            not_moving_ct=self.not_moving_timeout,
            centerline=self.centerline_arr,
            car_dims=CAR_DIMS
        )

        self.reward.set_track(
            inside_path=self.inside_path,
            outside_path=self.outside_path,
            centre_path=self.centre_path, 
            car_dims=CAR_DIMS
        )

    def record_manually(self, output_dir, fname='thruxton', num_imgs=5000,
                        sleep_time=0.03):
        """Record observations, including images, to an output directory. This
        is useful for collecting images from the environment. This method does
        not use the an agent to take environment steps; instead, it just
        listens for observations while a user manually drives the car in the
        simulator.

        :param output_dir: path of the output directory
        :type output_dir: str
        :param fname: file name for output
        :type fname: str, optional
        :param num_imgs: number of images to record
        :type num_imgs: int, optional
        :param sleep_time: time to sleep between images in seconds
        :type sleep_time: float, optional
        """
        self.multimodal = True
        self.reset()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        observations = []
        for i in range(num_imgs):
            observations.append(self._observe())
            time.sleep(sleep_time)

        for n, observation in enumerate(observations):
            pose, img = observation
            filename = f'{output_dir}/{fname}_{n}'
            np.savez_compressed(filename, pose_data=pose, image=img)

        print(f'Complete')

    def random_start_location(self):
        """Randomly selects an index on the centerline of the track and 
        returns the ENU coordinates of the selected index along with the yaw of
        the centerline at that point.

        NOT FUNCTIONAL
        TODO: adjust vehicle yaw

        :returns: coordinates of a random index on centerline, yaw
        :rtype: np array, float
        """
        rand_idx = np.random.randint(0, len(self.random_poses))
        pos = self.random_poses[rand_idx]
        print(f'setting random location to: {pos}')
        coords = {'x': pos[0], 'y': pos[1], 'z': pos[2]}
        rot = {'yaw': pos[3], 'pitch': 0.0, 'roll': 0.0}
        return coords, rot

    def _waypoints(self, goal='center', ct=3, step=8):
        """Return position of goal
        """
        l = len(self.centerline_arr)
        idxs = [self.nearest_idx+i*step for i in range(ct)]
        if goal=='center':
            return np.asarray([self.centerline_arr[idx % l] for idx in idxs])
        else:
            raise NotImplementedError

