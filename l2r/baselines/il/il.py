# ========================================================================= #
# Filename:                                                                 #
#    random.py                                                              #
#                                                                           #
# Description:                                                              # 
#    an agent that randomly chooses actions                                 #
# ========================================================================= #
import torch.nn as nn
import torch.optim as optim

from core.templates import AbstractAgent
from envs.env import RacingEnv
from baselines.il.building_blocks import Conv, Branching, FC, Join
from baselines.il.il_model import CILModel

class ILAgent(AbstractAgent):
    """Reinforcement learning agent that simply chooses random actions.

    :param training_kwargs: training keyword arguments
    :type training_kwargs: dict
    """
    def __init__(self, model_params, training_kwargs):
        self.num_episodes = training_kwargs['num_episodes']
        self.model = CILModel(model_params)    
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_kwargs['learning_rate'])
        self.mseLoss = nn.MSELoss()
    
    def select_action(self, x, a):
        """Select an action
        """
        out = self.model(x, a)
        return out

    ## Referencing: https://github.com/felipecode/coiltraine/blob/29060ab5fd2ea5531686e72c621aaaca3b23f4fb/coil_core/train.py
    def il_train(self, data_loader, n_epochs = 100, eval_every = 10):
        for i in range(n_epochs):
            for data, target in data_loader:
                # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
                self.model.zero_grad()
                
                ##TODO: Match I/O
                out = self.model(torch.squeeze(data['rgb'].cuda()),
                                dataset.extract_inputs(data).cuda())
                
                loss = self.mseLoss(out, target)
                loss.backward()
                self.optimizer.step()
            if (i+1)%eval_every == 0:
                self.eval()

    def eval(self):
		"""evaluate the agent
		"""
		for e in range(self.num_episodes):
			print('='*10+f' Episode {e+1} of {self.num_episodes} '+'='*10)
			ep_reward, ep_timestep = 0, 0
			state, done = self.env.reset(), False

			while not done:
				action = self.select_action()
				state, reward, done, info = self.env.step(action)
				ep_reward += reward
				ep_timestep += 1

			print(f'Completed episode with total reward: {ep_reward}')
			print(f'Episode info: {info}\n')

    
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
            logger_kwargs=env_kwargs['pose_if_kwargs']
        )

        self.env.make(
            level=sim_kwargs['racetrack'],
            multimodal=env_kwargs['multimodal'],
            driver_params=sim_kwargs['driver_params'],
            camera_params=sim_kwargs['camera_params'],
            sensors=sim_kwargs['active_sensors']
        )
