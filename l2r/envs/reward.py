# ========================================================================= #
# Filename:                                                                 #
#    reward.py                                                              #
#                                                                           #
# Description:                                                              # 
#    Dense incentive policy based off Super-Human Performance in Gran       #
#    Turismo Sport Using Deep Reinforcement Learning.                       #
#    https://arxiv.org/abs/2008.07971                                       #
# ========================================================================= #

import numpy as np

from core.templates import AbstractReward

VELOCITY_IDX_LOW=3
VELOCITY_IDX_HIGH=6

class GranTurismo(AbstractReward):
	"""This is the default reward for the environment. It is our interpretation
	of the work: Super-Human Performance in Gran Turismo Sport Using Deep
	Reinforcement Learning (https://arxiv.org/abs/2008.07971). Rewarding the
	agent only for completing a lap is too sparse for learning. Instead, this
	is a dense reward function that rewards the agent for progressing down the
	track and penalizes the agent for going out-of-bounds.

	:param oob_penalty: penalty factor for going out-of-bounds where the total
	  penalty is this factor times the velocity of the vehicle
	:type oob_penalty: float, optional
	:param min_oob_penalty: minimum penalty for going out-of-bounds
	:type min_oob_penalty: float, optional
	"""
	def __init__(self, oob_penalty=5.0, min_oob_penalty=25.0):
		self.oob_penalty = oob_penalty
		self.min_oob_penalty = min_oob_penalty

	def get_reward(self, state, oob_flag=False):
		"""The reward for the given state is the sum of its progress reward
		and the penalty for going out-of-bounds.

		:param state: the current state of the environment
		:type state: varies
		:param oob_flag: true if out-of-bounds, otherwise false
		:type oob_flag: boolean, optional
		"""
		(pose_data, race_idx) = state
		velocity = np.linalg.norm(pose_data[VELOCITY_IDX_LOW:VELOCITY_IDX_HIGH])
		oob_reward = self._reward_oob(velocity, oob_flag)
		progress_reward = self._reward_progress(race_idx)
		return oob_reward + progress_reward

	def reset(self):
		"""Reset cached index representing the position on the track.
		"""
		self.prior_idx = None

	def _reward_oob(self, velocity, oob_flag):
		"""Determine the reward for going out-of-bounds.

		:param velocity: magnitude of the velocity of the vehicle
		:type velocity: float
		:param oob_flag: true if out-of-bounds, otherwise false
		:type oob_flag: boolean
		:return: reward for going out-of-bounds
		:rtype: float
		"""
		if not oob_flag:
			return 0.0

		return min(-1.0*self.min_oob_penalty, -1.0*self.oob_penalty * velocity)

	def _reward_progress(self, race_idx):
		"""Reward for progressing down the track. This is simply a reward of 1
		for each index the vehicle has progressed since the previous step.

		:param race_idx: nearest index on the centerline of the racetrack
		:type race_idx: int
		:return: reward for progressing down the track
		:rtype: float
		"""
		if not self.prior_idx:
			self.prior_idx = race_idx
			return 0.0

		rwd = race_idx - self.prior_idx

		""" Check if vehicle has crossed back to index 0 """
		if rwd < (self.n_centre/-2.0):
			rwd = race_idx + (self.n_centre - self.prior_idx)

		self.prior_idx = race_idx
		return float(rwd)

