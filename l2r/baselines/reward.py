# ========================================================================= #
# Filename:                                                                 #
#    reward.py                                                              #
#                                                                           #
# Description:                                                              #
#    Sample reward policy which rewards being near centerline               #
# ========================================================================= #

import numpy as np
from core.templates import AbstractReward

VELOCITY_IDX_LOW = 3
VELOCITY_IDX_HIGH = 6
NORTH_IDX = 15
EAST_IDX = 16


class BaseGranTurismo(AbstractReward):
    """This is the default reward for the environment. It is our interpretation
    of the work: Super-Human Performance in Gran Turismo Sport Using Deep
    Reinforcement Learning (https://arxiv.org/abs/2008.07971). Rewarding the
    agent only for completing a lap is too sparse for learning. Instead, this
    is a dense reward function that rewards the agent for progressing down the
    track and penalizes the agent for going out-of-bounds.

    :param float oob_penalty: penalty factor for going out-of-bounds where the
      total penalty is this factor times the velocity of the vehicle
    :param float min_oob_penalty: minimum penalty for going out-of-bounds
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
        v = np.linalg.norm(pose_data[VELOCITY_IDX_LOW:VELOCITY_IDX_HIGH])
        oob_reward = self._reward_oob(v, oob_flag)
        progress_reward = self._reward_progress(race_idx)
        return oob_reward + progress_reward

    def reset(self):
        """Reset cached index representing the position on the track.
        """
        self.prior_idx = None

    def _reward_oob(self, velocity, oob_flag):
        """Determine the reward for going out-of-bounds.

        :param float velocity: magnitude of the velocity of the vehicle
        :param bool oob_flag: true if out-of-bounds, otherwise false
        :return: reward for going out-of-bounds
        :rtype: float
        """
        if not oob_flag:
            return 0.0

        return min(-1.0 * self.min_oob_penalty, -
                   1.0 * self.oob_penalty * velocity)

    def _reward_progress(self, race_idx):
        """Reward for progressing down the track. This is simply a reward of 1
        for each index the vehicle has progressed since the previous step.

        :param int race_idx: nearest index on the centerline of the racetrack
        :return: reward for progressing down the track
        :rtype: float
        """
        if not self.prior_idx:
            self.prior_idx = race_idx
            return 0.0

        rwd = race_idx - self.prior_idx

        """ Check if vehicle has crossed back to index 0 """
        if rwd < (self.n_centre / -2.0):
            rwd = race_idx + (self.n_centre - self.prior_idx)

        self.prior_idx = race_idx
        return float(rwd)


class CustomGranTurismoReward(BaseGranTurismo):
    """This is a modfication of the default GranTurismo reward policy which is
    our interpretation of the work: Super-Human Performance in Gran Turismo
    Sport Using Deep Reinforcement Learning (https://arxiv.org/abs/2008.07971).
    Here we reward the agent for progressing down the track and staying near
    the centerline of the track and punish it for going out-of-bounds.

    :param oob_penalty: penalty factor for going out-of-bounds where the total
      penalty is this factor times the velocity of the vehicle
    :type oob_penalty: float, optional
    :param min_oob_penalty: minimum penalty for going out-of-bounds
    :type min_oob_penalty: float, optional
    :param centerline_bonus: bonus reward for being near the centerline
    :type centerline_bonus: float, optional
    :param dist_threshold: max distance from centerline to receive the bonus
    :type dist_threshold: float, optional
    """

    def __init__(
            self,
            oob_penalty=5.0,
            min_oob_penalty=25.0,
            centerline_bonus=1.0,
            dist_threshold=1.0):
        self.oob_penalty = oob_penalty
        self.min_oob_penalty = min_oob_penalty
        self.c_bonus = centerline_bonus
        self.d_thresh = dist_threshold

    def get_reward(self, state, oob_flag=False):
        """The reward for the given state is the sum of its progress reward
        and the penalty for going out-of-bounds.

        :param state: the current state of the environment
        :type state: varies
        :param oob_flag: true if out-of-bounds, otherwise false
        :type oob_flag: boolean, optional
        """
        (pose_data, race_idx) = state
        velocity = np.linalg.norm(
            pose_data[VELOCITY_IDX_LOW:VELOCITY_IDX_HIGH])
        loc = np.array([pose_data[EAST_IDX], pose_data[NORTH_IDX]])
        oob_reward = self._reward_oob(velocity, oob_flag)
        progress_reward = self._reward_progress(race_idx)
        bonus = self._reward_centerline(race_idx, loc, progress_reward)
        return oob_reward + progress_reward + bonus

    def _reward_centerline(self, race_idx, loc, progress_reward):
        """Provide bonus reward if near the centerline if there has been
        progress from the last time step.

        :param race_idx: index on the track that the vehicle is nearest to
        :type race_idx: int
        :param loc: location of the vehicle, (East, North)
        :type loc: tuple of floats
        :param progress_reward: reward for progressing down the centerline
        :type progress_reward: float
        :return: centerline bonus reward
        :rtype: float
        """
        if not progress_reward > 0:
            return 0.0

        dist = np.linalg.norm(loc - self.centre_path[race_idx])
        return self.c_bonus if dist < self.d_thresh else 0.0
