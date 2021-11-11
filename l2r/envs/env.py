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
import random
import time

import gym
import matplotlib.path as mplPath
import numpy as np
from gym.spaces import Box, Dict
from scipy.spatial import KDTree

import envs.utils as utils
from baselines.reward import CustomReward
from core.controller import SimulatorController
from envs.reward import GranTurismo
from core.tracker import ProgressTracker
from racetracks.mapping import level_2_trackmap

import ipdb as pdb

# Simulator Lag Delay
MEDIUM_DELAY = 3
TIMEOUT_DELAY = 30
LAUNCHING_DELAY = 15

# Restart simulator container every so often
SIM_RESET = 60 * 20

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

N_EPISODE_LAPS = 1
N_SEGMENTS = 10

# Pose observation space boundaries
MIN_OBS_ARR = [
    -1., -1., -1.,                   # steering, gear, mode
    -200., -200., -10.,              # velocity
    -100., -100., -100.,             # acceleration
    -1., -1., -5.,                   # angular velocity
    -6.2832, -6.2832, -6.2832,       # yaw, pitch, roll
    -2000., 2000., 2000.,            # location coordinates in the format (y, x, z)
    -2000., -2000., -2000., -2000.,  # rpm (per wheel)
    -1., -1., -1., -1.,              # brake (per wheel)
    -1., -1., -1300., -1300.]        # torq (per wheel)

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

# Racetrack IDs
RACETRACKS = {
    'VegasNorthRoad': 0,
    'Thruxton': 1,
    'AngleseyNational': 2,
}

class RacingEnv(gym.Env):
    """A reinforcement learning environment for autonomous racing.

    :param int max_timesteps: maximimum number of timesteps per episode
    :param dict controller_kwargs: keyword args for the simulator controller
    :param dict reward_kwargs: keyword args the reward policy
    :param dict action_if_kwargs: keyword args for the action interface
    :param dict camera_if_kwargs: keyword args for the camera interface
    :param dict pose_if_kwargs: keyword args for the position receive interface
    :param str reward_pol: reward policy to use, defaults to GranTurismo
    :param int not_moving_timeout: terminate if not moving for this many
      timesteps
    :param bool provide_waypoints: flag to provide ground-truth, future
      waypoints on the track in the info returned from **step()**
    :param float obs_delay: time delay between action and observation
   """

    def __init__(self, max_timesteps, controller_kwargs, reward_kwargs,
                 action_if_kwargs, camera_if_kwargs, pose_if_kwargs, sensors,
                 reward_pol='default', obs_delay=0.10, segm_if_kwargs=False,
                 birdseye_if_kwargs=False, birdseye_segm_if_kwargs=False,
                 not_moving_timeout=20, zone=False, provide_waypoints=False):

        self.controller = SimulatorController(**controller_kwargs)
        self.action_if = utils.ActionInterface(**action_if_kwargs)
        self.pose_if = utils.PoseInterface(**pose_if_kwargs)

        camera_sensor_name = [c for c in sensors if "Camera" in c][0]

        self.cameras = [(camera_sensor_name,
            utils.CameraInterface(**camera_if_kwargs))]

        if segm_if_kwargs:
            self.cameras.append(('CameraFrontSegm',
                utils.CameraInterface(**segm_if_kwargs)))

        if birdseye_if_kwargs:
            self.cameras.append(('CameraBirdsEye',
                utils.CameraInterface(**birdseye_if_kwargs)))

        if birdseye_segm_if_kwargs:
            self.cameras.append(('CameraBirdsEyeSegm',
                utils.CameraInterface(**birdseye_segm_if_kwargs)))

        self.reward = GranTurismo(**reward_kwargs) if reward_pol == 'default' \
            else CustomReward(**reward_kwargs)
        self.max_timesteps = max_timesteps
        self.not_moving_timeout = not_moving_timeout
        self.observation_delay = obs_delay
        self.provide_waypoints = provide_waypoints
        self.last_restart = time.time()
        self.zone = zone

        # openAI gym compliance - action space
        self.action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float64)

    def make(self, level, multimodal, camera_params, sensors, driver_params,
             segm_params=False, birdseye_params=False,
             birdseye_segm_params=False, vehicle_params=None,
             multi_agent=False, remake=False):
        """Unlike many environments, make does not start the simulator process.
        It does, however, configure the simulator's settings. The simulator
        process must be running prior to calling this method otherwise an error
        will occur when trying to establish a connection with the simulator.

        :param str level: the desired racetrack map
        :param bool multimodal: if false, then the agent is 'visual only'
          and only receives pixel values, if true, the agent also has access
          to additional sensor data
        :param dict camera_params: camera parameters to set
        :param list sensors: sensors to enable for the simulation
        :param dict driver_params: driver parameters to modify
        :param dict vehicle_params: vehicle parameters to set
        :param multi_agent: not currently supported
        :param bool remake: if remaking, reset the camera interface
        """
        camera_sensor_name = [c for c in sensors if "Camera" in c][0]
        self.camera_dims = {camera_sensor_name: # 'CameraFrontalRGB'
            {'width': camera_params['Width'],
             'height': camera_params['Height']}}

        if segm_params:
            self.camera_dims['CameraFrontSegm'] = \
                {'width': segm_params['Width'],
                 'height': segm_params['Height']}

        if birdseye_params:
            self.camera_dims['CameraBirdsEye'] = \
                {'width': birdseye_params['Width'],
                 'height': birdseye_params['Height']}

        if birdseye_segm_params:
            self.camera_dims['CameraBirdsEyeSegm'] = \
                {'width': birdseye_segm_params['Width'],
                 'height': birdseye_segm_params['Height']}
        
        if type(level) == str:
            self.level = level
            self.levels = None
            self.active_level = level
        elif type(level) == list:
            self.levels = level
            self.active_level = random.choice(self.levels)
        else:
            pdb.set_trace()
            raise NotImplemented

        self.sensors = sensors
        self.multimodal = multimodal
        self.multi_agent = multi_agent
        self.vehicle_params = vehicle_params
        self.camera_params = camera_params
        self.driver_params = driver_params

        self.controller.set_level(self.active_level)
        self.controller.set_api_udp()        
        self._load_map()
        #pdb.set_trace()

        for sensor in sensors:
            self.controller.enable_sensor(sensor)

        self.controller.set_sensor_params(sensor='ArrivalVehicleDriver',
                                          params=driver_params)

        for name, params in self.camera_dims.items():
            self.controller.set_sensor_params(sensor=name, params=params)

        if remake:
            for (_, cam) in self.cameras:
                cam.reconnect()
        else:
            self.pose_if.start()
            for (name, cam) in self.cameras:
                cam.start(img_dims=(self.camera_dims[name]['width'],
                                    self.camera_dims[name]['height'], 3))

    def _restart_simulator(self):
        """Periodically need to restart the container for long runtimes
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
            multi_agent=self.multi_agent,
            remake=True
        )

    def _check_restart(self, done):
        """Check if we should restart the simulator
        """
        if not done or not self.controller.start_container:
            return

        if time.time() - self.last_restart > SIM_RESET:
            self.last_restart = time.time()
            self._restart_simulator()

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
        self.action_if.act(action)
        _observation = self._observe()
        _data, _imgs = _observation
        observation = _observation if self.multimodal else _imgs
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

    def reset(self, level=None, random_pos=False, segment_pos=True):
        """Resets the vehicle to start position. A small time delay is used
        allow for the simulator to reset.

        :param str level: if specified, will set the simulator to this level,
          otherwise set to a random track
        :param bool random_pos: true/false for random starting position on the
          track
        :return: an intial observation as in the *step* method
        :rtype: see **step()** method
        """
        #new_level = level if level else random.choice(self.levels)
        if level:
            new_level = level
        elif self.levels:
            new_level = random.choice(self.levels)
        else:
            new_level = self.level
            
        if new_level is self.active_level:
            self.controller.reset_level()
        else:
            self.active_level = new_level
            self.controller.set_level(self.active_level)
            self._load_map()

        self.nearest_idx, info = None, {}

        # give the simulator time to reset
        time.sleep(MEDIUM_DELAY)

        self.training, self.evaluation = False, True

        # randomly initialize starting location
        p = np.random.uniform()
        # with prob 1/(1+n) use the default start location. 
        if (random_pos) & (p > 2/(1+len(self.random_poses))) :
            coords, rot = self.random_start_location()
            self.controller.set_location(coords, rot)
            time.sleep(MEDIUM_DELAY)

        elif segment_pos and self.evaluation:
            coords, rot = self.next_segment_start_location()
            self.controller.set_location(coords, rot)
            time.sleep(MEDIUM_DELAY)
        else:
            pass

        # reset simulator sensors
        self.controller.set_mode_ai()

        if self.vehicle_params:
            self.controller.set_vehicle_params(self.vehicle_params)

        for sensor in self.sensors:
            self.controller.enable_sensor(sensor)

        self.reward.reset()
        self.pose_if.reset()

        for (_, cam) in self.cameras:
            cam.reset()

        # no delay is causing issues with the initial starting index
        #time.sleep(MEDIUM_DELAY)
        self.poll_simulator(new_level, random_pos)

        _observation = self._observe()
        _data, _img = _observation
        observation = _observation if self.multimodal else _img
        self.tracker.reset(start_idx=self.nearest_idx, segmentwise=segment_pos)

        if self.provide_waypoints:
            info['waypoints'] = self._waypoints()
            info['track_idx'] = self.nearest_idx
            return observation, info

        return observation

    def poll_simulator(self, level, random_pos):
        """Poll the simulator until it receives an action
        """
        action = (1.0, 0) # steering, acceleration

        while True:
            self.action_if.act(action)
            pose = self.pose_if.get_data()
            time.sleep(0.1)
            if abs(pose[0]) > 0.05:
                break

    def render(self):
        """Not implmeneted. The simulator, by default, provides a graphical
        interface, but can also be run on a server.
        """
        #raise NotImplementedError
        return self.imgs

    @property
    def multimodal(self):
        """Getter method for the multimodal property. Changing this value will
        cause the environment's observation space to change. If true, the
        environment returns observations as a dictionary with keys:
        ['sensors', 'img'], otherwise, just the image is returned.

        :return: true if the environment is set to multimodal, false otherwise
        :rtype: bool
        """
        return self._multimodal

    @multimodal.setter
    def multimodal(self, value):
        """Setter method for the multimodal property.

        :param bool value: value to self the multimodal property to. true sets
          the environment to multimodal and makes the observation space a
          dictionary of the camera images and the sensor data. false is visual
          only features.
        """
        if not isinstance(value, bool):
            raise TypeError('Multimodal property must be of type: bool')

        self._multimodal = value
        _spaces = {}

        for (name, cam) in self.cameras:
            _shape = (self.camera_dims[name]['width'],
                      self.camera_dims[name]['height'], 3)
            _spaces[name] = Box(low=0, high=255, shape=_shape, dtype=np.uint8)

        if self._multimodal:
            _spaces['sensors'] = Box(low=np.array(MIN_OBS_ARR),
                                     high=np.array(MAX_OBS_ARR))
        self.observation_space = Dict(_spaces)
        print(self.observation_space)

    def _observe(self):
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
        pose = self.pose_if.get_data()
        self.imgs = [cam.get_data() for (_, cam) in self.cameras]

        yaw = pose[12]
        bp = pose[22:25]
        a = pose[6:9]

        # provide racetrack ID in the observation
        pose[2] = RACETRACKS[self.active_level]

        # convert to local coordinate system
        x, y, z = pose[16], pose[15], pose[17]
        enu_x, enu_y, enu_z = self.geo_location.convert_to_ENU((x, y, z))
        pose[16], pose[15], pose[17] = enu_x, enu_y, enu_z

        self.nearest_idx = self.kdtree.query(np.asarray([enu_x, enu_y]))[1]
        self.tracker.update(self.nearest_idx, enu_x, enu_y, enu_z, yaw, a, bp)

        #pdb.set_trace()

        return (pose, self.imgs)

    def _is_complete(self, observation):
        """Determine if the episode is complete. Termination conditions include
        car out-of-bounds, 3-laps successfully complete, not-moving-timeout,
        and max timesteps reached
        """
        return self.tracker.is_complete()

    def _load_map(self):
        """Loads the racetrack map from a data file. The map is parsed into
        numerous arrays and matplotlib Path objects representing the inside
        and outside track boundaries along with the centerline.

        :param str level: the racetrack name
        """
        map_file, self.random_poses = level_2_trackmap(self.active_level)

        with open(os.path.join(pathlib.Path().absolute(), map_file), 'r') as f:
            self.original_map = json.load(f)
            self.ref_point = self.original_map["ReferencePoint"]

        _out = np.asarray(self.original_map['Outside'])
        _in = np.asarray(self.original_map['Inside'])

        self.outside_arr = _out if _out.shape[-1] == 2 else _out[:, :-1]
        self.inside_arr = _in if _in.shape[-1] == 2 else _in[:, :-1]
        self.centerline_arr = np.asarray(self.original_map['Centre'])

        self.centre_path = mplPath.Path(self.centerline_arr)
        self.outside_path = mplPath.Path(self.outside_arr)
        self.inside_path = mplPath.Path(self.inside_arr)

        self.geo_location = utils.GeoLocation(self.ref_point)
        self.n_indices = len(self.centerline_arr)
        self.kdtree = KDTree(self.centerline_arr)

        self.local_segment_idxs = self.poses_to_local_segment_idxs()

        #pdb.set_trace()

        self.tracker = ProgressTracker(
            n_indices=len(self.centerline_arr),
            obs_delay=self.observation_delay,
            max_timesteps=self.max_timesteps,
            inner_track=self.inside_path,
            outer_track=self.outside_path,
            not_moving_ct=self.not_moving_timeout,
            centerline=self.centerline_arr,
            car_dims=CAR_DIMS,
            n_episode_laps=N_EPISODE_LAPS,
            n_segments=N_SEGMENTS
        )

        self.reward.set_track(
            inside_path=self.inside_path,
            outside_path=self.outside_path,
            centre_path=self.centre_path,
            car_dims=CAR_DIMS
        )

        self.segment_coords = self.tracker.get_segment_coords(self.centerline_arr, self.tracker.segment_idxs)

        pdb.set_trace()
        pass


    def record_manually(self, output_dir, fname='thruxton', num_imgs=5000,
                        sleep_time=0.03):
        """Record observations, including images, to an output directory. This
        is useful for collecting images from the environment. This method does
        not use the an agent to take environment steps; instead, it just
        listens for observations while a user manually drives the car in the
        simulator.

        :param str output_dir: path of the output directory
        :param str fname: file name for output
        :param int num_imgs: number of images to record
        :param float sleep_time: time to sleep between images, in seconds
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

        print('Complete')

    def random_start_location(self):
        """Randomly selects an index on the centerline of the track and
        returns the ENU coordinates of the selected index along with the yaw of
        the centerline at that point.

        :returns: coordinates of a random index on centerline, yaw
        :rtype: np array, float
        """
        rand_idx = np.random.randint(0, len(self.random_poses))
        pos = self.random_poses[rand_idx]
        print(f'setting random location to: {pos}')
        coords = {'x': pos[0], 'y': pos[1], 'z': pos[2]}
        rot = {'yaw': pos[3], 'pitch': 0.0, 'roll': 0.0}
        return coords, rot

    
    def next_segment_start_location(self):
        print(f"Spawning to next segment start location: curr_segment: {self.tracker.current_segment}; respawns: {self.tracker.respawns}")

        next_segment_idx = self.tracker.respawns

        next_segment_idx = next_segment_idx % 11

        try:
            pos = self.random_poses[next_segment_idx]
            #pos = [0]*4
            #pos[0] = self.tracker.segment_coords['first'][next_segment_idx][0] # x
            #pos[1] = self.tracker.segment_coords['first'][next_segment_idx][1] # y

            #dy = pos[1]-self.tracker.segment_coords['second'][next_segment_idx][1]
            #dx = pos[0]-self.tracker.segment_coords['second'][next_segment_idx][0]
            #
            #pos[2] = 61.3 # z # TODO: different for each track
            #pos[3] = np.arctan(dx/-dy) # yaw, radians
            
        except:
            pdb.set_trace()
            pass

        #pdb.set_trace()

        coords = {'x': pos[0], 'y': pos[1], 'z': pos[2]}
        rot = {'yaw': pos[3], 'pitch': 0.0, 'roll': 0.0}

        self.tracker.current_segment += 1

        return coords, rot
    
    def poses_to_local_segment_idxs(self):
        
        segment_idxs = []
        for (x,y,z,yaw) in self.random_poses:
            enu_x, enu_y, enu_z = self.geo_location.convert_to_ENU((x, y, z))
            idx = self.kdtree.query(np.asarray([enu_x, enu_y]))[1]
            segment_idxs.append(idx)

        return segment_idxs


    def _waypoints(self, goal='center', ct=3, step=8):
        """Return position of goal
        """
        num = len(self.centerline_arr)
        idxs = [self.nearest_idx + i * step for i in range(ct)]
        if goal == 'center':
            return np.asarray([self.centerline_arr[idx % num] for idx in idxs])
        else:
            raise NotImplementedError
