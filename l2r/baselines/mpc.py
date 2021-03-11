# ========================================================================= #
# Filename:                                                                 #
#    MPC.py                                                                 #
#                                                                           #
# Description:                                                              # 
#    an agent used a MPC controller                                         #
# ========================================================================= #

import json
import os
import pathlib

import numpy as np
import torch

from baselines.components.mpc_utils import BikeModel, MPC
from core.templates import AbstractAgent
from envs.env import RacingEnv
from racetracks.mapping import level_2_trackmap

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

# State variables
DT = 0.1  # [s] time tick
N_STATE = 4 # [x, y, v, yaw]
VELOCITY_TARGET = 25.0 # [m/s]

class MPCAgent(AbstractAgent):
    """Agent that selects actions using an MPC controller
    """
    def __init__(self, mpc_kwargs):
        self.num_episodes = mpc_kwargs["num_episodes"]
        self.car = BikeModel(dt=DT, init_params=np.array([WB, 1, 1]))
        self.controller = MPC(cost_weight=torch.FloatTensor([1.0, 1.0, 0.5, 0.5, 0.01, 0.01]), cost_type='reference')

    def race(self):
        """Race following the environment's step until completion framework
        """
        for e in range(self.num_episodes):
            print('='*10+f' Episode {e+1} of {self.num_episodes} '+'='*10)
            ep_reward, ep_timestep = 0, 0
            state, info = self.env.reset()
            done = False

            while not done:
                x, y, v, yaw = MPCAgent.unpack_state(state)
                x0 = [x, y, v, yaw]  # current state
                idx = info['track_idx']
                xref = self.get_xref(idx)
                
                ## Using my model
                _, u_opt = self.controller.forward(self.car,
                        torch.tensor(x0).unsqueeze(0), 
                        torch.tensor(xref).transpose(0, 1).unsqueeze(1))

                u = u_opt[0].squeeze().detach().numpy()
                ai, di = u

                print(f'acceleration: {ai}, steer: {di}')

                #action = self.select_action(state, info)

                # u_opt is a sequence of actions, along the predicted trajectory, 
                # so we'll take the first one to give to the environment
                obs, reward, done, info = self.env.step([di, ai])
                ep_reward += reward
                ep_timestep += 1

            print(f'Completed episode with total reward: {ep_reward}')
            print(f'Episode info: {info}\n')

    def get_xref(self, idx, step_size=6, n_targets=6):
    	"""Get targets.

    	:param idx: index of the raceline the agent is nearest to
    	:type idx: int
    	:return: array of shape (4, n_targets)
    	:rtype: numpy array
    	"""
    	target_idxs = [(idx+step_size*t) % self.raceline_length for t in range(1, n_targets+1)]
    	target_x = [self.race_x[i] for i in target_idxs]
    	target_y = [self.race_y[i] for i in target_idxs]
    	target_yaw = [self.race_yaw[i] for i in target_idxs]
    	target_v = [VELOCITY_TARGET] * n_targets
    	return np.array([target_x, target_y, target_yaw, target_v], dtype=np.float32)

    @staticmethod
    def unpack_state(state):
    	(state, _) = state
    	x = state[16]
    	y = state[15]
    	v = (state[4]**2 + state[3]**2 + state[5]**2)**0.5
    	yaw = state[12]
    	return x, y, v, yaw

    def select_action(self):
    	pass

    def create_env(self, env_kwargs, sim_kwargs):
        """Instantiate a racing environment

        :param env_kwargs: environment keyword arguments
        :type env_kwargs: dict
        :param sim_kwargs: simulator setting keyword arguments
        :type sim_kwargs: dict
        """
        self.env = RacingEnv(
                max_timesteps=env_kwargs['max_timesteps'],
                obs_delay=env_kwargs['obs_delay'],
                not_moving_timeout=env_kwargs['not_moving_timeout'],
                controller_kwargs=env_kwargs['controller_kwargs'],
                reward_pol=env_kwargs['reward_pol'],
                reward_kwargs=env_kwargs['reward_kwargs'],
                action_if_kwargs=env_kwargs['action_if_kwargs'],
                camera_if_kwargs=env_kwargs['camera_if_kwargs'],
                pose_if_kwargs=env_kwargs['pose_if_kwargs'],
                provide_waypoints=env_kwargs['provide_waypoints'],
                logger_kwargs=env_kwargs['pose_if_kwargs']
        )

        self.env.make(
                level=sim_kwargs['racetrack'],
                multimodal=env_kwargs['multimodal'],
                driver_params=sim_kwargs['driver_params'],
                camera_params=sim_kwargs['camera_params'],
                sensors=sim_kwargs['active_sensors']
        )

        self.load_track(sim_kwargs['racetrack'])


    def load_track(self, track_name='VegasNorthRoad'):
        """Load trace track

        :param track_name: 'VegasNorthRoad' or 'Thruxton'
        :type track_name: str
        """
        map_file, _ = level_2_trackmap(track_name)

        with open(os.path.join(pathlib.Path().absolute(), map_file), 'r') as f:
            original_map = json.load(f)

        raceline = np.asarray(original_map['Racing'], dtype=np.float32).T
        self.race_x = raceline[0]
        self.race_y = raceline[1]
        self.race_yaw = np.asarray(original_map['RacingPsi'], dtype=np.float32)
        self.raceline_length = self.race_x.shape[0]
