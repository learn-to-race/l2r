# ========================================================================= #
# Filename:                                                                 #
#    random.py                                                              #
#                                                                           #
# Description:                                                              #
#    an agent that randomly chooses actions                                 #
# ========================================================================= #

import os, random
from core.templates import AbstractAgent
from envs.env import RacingEnv

class RandomActionAgent(AbstractAgent):
    """Reinforcement learning agent that simply chooses random actions.

    :param dict training_kwargs: training keyword arguments
    """

    def __init__(self, training_kwargs):
        self.num_episodes = training_kwargs['num_episodes']
        self.seed = random.randint(0, 9999)
        self.save_path = False
        if 'save_path' in training_kwargs:
            self.save_path = training_kwargs['save_path']

    def race(self):
        """Demonstrative agent method."""

        for e in range(self.num_episodes):
            print('='*10 + f' Episode {e+1} of {self.num_episodes} ' + '='*10)
            ep_reward, ep_timestep = 0, 0
            state, done = self.env.reset(), False

            while not done:
                action = self.select_action()
                state, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_timestep += 1

            if self.save_path:
                save_dir = os.path.join(self.save_path, f'seed_{self.seed}')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, f'episode{e}.txt'), 'w') as f:
                    f.write(f'total reward: {reward}')

            print(f'Completed episode with total reward: {ep_reward}')
            print(f'Episode info: {info}\n')

    def select_action(self):
        """Select a random action from the action space.

        :return: random action to take
        :rtype: numpy array
        """
        return self.env.action_space.sample()

    def create_env(self, env_kwargs, sim_kwargs):
        """Instantiate a racing environment

        :param dict env_kwargs: environment keyword arguments
        :param dict sim_kwargs: simulator setting keyword arguments
        """

        self.env = RacingEnv(env_kwargs, sim_kwargs)
        self.env.make()

        print('Environment created with observation space: ')
        for k, v in self.env.observation_space.spaces.items():
            print(f'\t{k}: {v}')
