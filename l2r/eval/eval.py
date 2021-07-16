"""
Description:
    This script is used to evaluate agents. Modifying methods of the Evaluator
    class which are marked as 'Do not modify' will result in an exception
    during evaluation, and no score will be provided.

Invocation:
    This script will be invoked as follows during evaluation.

    $ python3 \
      --config conf.yaml \
      --duration 3600 \
      --episodes 3 \
      --racetrack VegasNorthRoad \
      eval.py

Loading Data:
    TODO

Custom Reward Policies:
    TODO

Environment Wrappers:
    TODO
"""

import argparse
import time

from envs.env import RacingEnv

class IllegalEvaluation(Exception):
    pass

class Evaluator(object):
    """Evaluator class which consists of a 1-hour pre-evaluation phase
    followed by an evaluation phase.

    :param dict config: yaml configuation which must be compatible with this
      class's create_env method
    :param int pre_eval_time: duration of pre-evaluation phase, in seconds
    :param int eval_episodes: number of evaluation episodes
    """
    def __init__(self, config, pre_eval_time, eval_episodes):
        """Do not modify.
        """
        self.conf = config
        self.pre_eval_time = pre_eval_time
        self.eval_episodes = eval_episodes
        self.start_time = False

    def load_agent(self, **kwargs):
        """Modify this method.

        Load your agent, policy weights, etc.
        """
        raise NotImplementedError

    def pre_evaluate(self):
        """Modify this method.

        WARNING: Users that use an environment wrapper are responsible for
        ensuring that their agent begins evaluation within 1-hour of calling
        this method.
        """
        self._start_pre_eval() # do not modify this line

        done = False
        state = self.env.reset()

        while True:
            while not done:
                action = self.agent.select_action(state)
                state, reward, done, info = self.env.step(action)

                # Users are responsible for keeping time under 1-hour
                if time.time() - self.start_time > (self.pre_eval_time-10.0):
                    # Do not forget to set agent to evaluation mode
                    self.agent.eval()
                    return

            self._check_duration()

    def evaluate(self):
        """Do not modify.
        """
        self.env.eval()

        for ep in range(self.eval_episodes):
            done = False
            state = self.env.reset()
            while not done:
                state, reward, done, info = self.env.step(action)
            self._record_metrics(ep, info['metrics'])

    def _record_metrics(self, episode, metrics):
        """Do not modify.
        """
        print(f'\nCompleted evaluation episode {episode+1} with metrics:')
        for k, v in metrics.items():
            print(f'{k}: {v}')

    def _start_pre_eval(self):
        """Do not modify
        """
        self.env.train()
        self.start_time = time.time()

    def _check_duration(self):
        """Do not modify.
        """
        if time.time() - self.start_time > PRE_EVALUATION_TIME:
            raise IllegalEvaluation('Time exceeded')

    def create_env(self, eval_track):
        """Do not modify.

        Your configuration yaml file must contain the keys below.
        """
        try:
            env_kwargs = self.config['env_kwargs']
            sim_kwargs = self.config['sim_kwargs']
        except KeyError:
            raise KeyError('Invalid configuration file')

        self.env = RacingEnv(
            max_timesteps=env_kwargs['max_timesteps'],
            obs_delay=env_kwargs['obs_delay'],
            not_moving_timeout=env_kwargs['not_moving_timeout'],
            controller_kwargs=env_kwargs['controller_kwargs'],
            reward_pol=env_kwargs['reward_pol'],
            reward_kwargs=env_kwargs['reward_kwargs'],
            action_if_kwargs=env_kwargs['action_if_kwargs'],
            pose_if_kwargs=env_kwargs['pose_if_kwargs'],
            cameras=env_kwargs['cameras']
        )

        self.env.make(
            level=eval_track,
            multimodal=env_kwargs['multimodal'],
            driver_params=sim_kwargs['driver_params']
        )

if __name__ == '__main__':
    """Do not modify"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--duration', type=int, default=3600,
                        help='Pre-evaluation duration, in seconds')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Configuration file')
    parser.add_argument('--racetrack', type=str, default='VegasNorthRoad',
                        help='Racetrack to evaluate agent on')
    args = parser.parse_args()

    evaluator = Evaluator(config=args.config, pre_eval_time=args.duration,
                          eval_episodes=args.episodes)

    evaluator.create_env(args.racetrack)
    evaluator.load_agent()
    evaluator.pre_evaluate()
    evaluator.evaluate()
