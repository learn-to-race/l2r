# ========================================================================= #
# Filename:                                                                 #
#    MPC.py                                                                 #
#                                                                           #
# Description:                                                              # 
#    an agent used a MPC controller                                         #
# ========================================================================= #

from core.templates import AbstractAgent
from envs.env import RacingEnv

class MPCAgent(AbstractAgent):
	"""MPC controller
	"""
	def __init__(self, num_episodes):
		self.num_episodes = num_episodes
		# TODO - create controller

	def race(self):
		"""Race using an MPC controller
		"""
		for e in range(self.num_episodes):
			print('='*10+f' Episode {e+1} of {self.num_episodes} '+'='*10)
			ep_reward, ep_timestep = 0, 0
			state, done, info = self.env.reset(), False, {'waypoints': None}

			while not done:
				action = self.select_action(state, info)
				state, reward, done, info = self.env.step(action)
				ep_reward += reward
				ep_timestep += 1

			print(f'Completed episode with total reward: {ep_reward}')
			print(f'Episode info: {info}\n')


	def select_action(self, state, info):
		"""Select an action using a MPC controller

		:return: random action to take
		:rtype: numpy array
		"""
		(state, _) = state
		vx, vy, vz = state[4], state[3], state[5] # directional velocity
		v = (vx**2 + vy**2 + vz**2)**0.5          # magnitude of velocity
		e, n = state[16], state[15]               # current (x,y)
		waypoints = info['waypoints']             # list of 3 future waypoints -> (x,y) pairs
		yaw = state[12]

		if waypoints:
			print(waypoints)

		# select action [steer, accel], hardcoded for now...
		# bounds at (-1., 1.) for both
		action = [0.0, 0.5]
		return action

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
