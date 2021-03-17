# ========================================================================= #
# Filename:                                                                 #
#    random.py                                                              #
#                                                                           #
# Description:                                                              # 
#    an agent that randomly chooses actions                                 #
# ========================================================================= #
import torch
import torch.nn as nn
import torch.optim as optim

import pdb as pdb

from core.templates import AbstractAgent
from envs.env import RacingEnv

from baselines.il.il_model import CILModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

class ILAgent(AbstractAgent):
    """Reinforcement learning agent that simply chooses random actions.

    :param training_kwargs: training keyword arguments
    :type training_kwargs: dict
    """
    def __init__(self, model_params, training_kwargs):
        self.num_episodes = training_kwargs['num_episodes']
        
        self.model = CILModel(model_params)    
        # self.model = self.model.to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=training_kwargs['learning_rate'])
        self.mseLoss = nn.MSELoss()
        self.model = self.model.to(DEVICE)
        self.save_path = training_kwargs['save_path']

    def select_action(self, x, a):
        """Select an action
        """
        out = self.model(x, a)
        return out

    def il_train(self, data_loader, **il_kwargs):

        n_epochs = il_kwargs['n_epochs']
        eval_every = il_kwargs['eval_every']

        for i in range(n_epochs):
            for imgs, sensors, target in data_loader:
                '''
                Input for NN:
                    imgs: n x 3 x H x W
                    sensors: n x Dim 
                Target: n x 2 
                '''

                imgs, sensors, target = imgs.type(torch.FloatTensor).to(DEVICE), \
                        sensors.to(DEVICE), target.to(DEVICE) 
                
                imgs = imgs.transpose(2, 3) # B x 3 x 512 x 384

                pdb.set_trace()

                # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
                self.model.zero_grad()
                
                ##TODO: Match I/O
                out = self.model(imgs, sensors)
                
                loss = self.mseLoss(out, target)
                loss.backward()
                self.optimizer.step()
            
            if (i+1)%eval_every == 0:
                print("Eval / save")
                self.eval()
                self.save_model(i)

    def eval(self):
        """
        evaluate the agent
        """
        print("Model evaluation")

        model_cpu = self.model.cpu()

        for e in range(self.num_episodes):
            print('='*10+f' Episode {e+1} of {self.num_episodes} '+'='*10)
            ep_reward, ep_timestep, best_ep_reward = 0, 0, 0
            obs = self.env.reset()
            obs, reward, done, info = self.env.step([0, 1])

            while not done:
                (sensor, img) = obs
                img = torch.FloatTensor(img).unsqueeze(0).transpose(1, 3) # 1 x 3 x 512 x 384 
                action = model_cpu(img, torch.FloatTensor(sensor).unsqueeze(0))
                action = torch.clamp(action, -1, 1)
                obs, reward, done, info = self.env.step(action.squeeze(0).detach().numpy())
                ep_reward += reward
                ep_timestep += 1
            
            # Save if best (or periodically)
                if (ep_reward > best_ep_reward and ep_reward > 250):
                    print(f'New best episode reward of {round(ep_reward,1)}!')
                    best_ep_reward = ep_reward
                    path_name = f'{self.save_path}il_episode_{e}_best.pt'
                    torch.save(self.model.state_dict(), path_name)

            print(f'Completed episode with total reward: {ep_reward}')
            print(f'Episode info: {info}\n')


    def save_model(self, e):
        path_name = f'{self.save_path}il_episode_{e}.pt'
        torch.save(self.model.state_dict(), path_name)

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
