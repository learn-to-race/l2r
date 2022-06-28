import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from baselines.core import *


class MLPActorCritic(nn.Module):

    def __init__(
            self,
            observation_space,
            action_space,
            hidden_sizes=(
                256,
                256),
            activation=nn.ReLU,
            latent_dims=None,
            device='cpu'):
        super().__init__()

        obs_dim = observation_space.shape[0] if latent_dims is None else latent_dims
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.device = device
        self.to(device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy() if self.device == 'cpu' else a.cpu().numpy()
