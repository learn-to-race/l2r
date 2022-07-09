"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.

For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version

Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
import queue
import threading
from copy import deepcopy

import torch
import numpy as np
from torch.optim import Adam

from core.templates import AbstractAgent
from baselines.network_architecture.sac import ActorCritic
from cv.vae import VAE
from common.utils import RecordExperience
from baselines.rl.simplebuffer import ReplayBuffer
from constants import DEVICE

#seed = np.random.randint(255)
# torch.manual_seed(seed)
# np.random.seed(seed)


class SACAgent(AbstractAgent):
    """
    Soft Actor-Critic (SAC)


    Args:
        env : an OpenAI gym compliant reinforcement learning environment

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        total_steps (int): Total timesteps to be executed in the environment

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        num_updates (int): Number of gradient steps to take per update.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        encoder_path (str): Path to image encoder

        im_w (int): width of observation image in pixels

        im_h (int): height of observation image in pixels

        latent_dims (int): size of the flattened latent space

        save_path (str): path to save model checkpoints

    """

    def __init__(self, env, agent_kwargs, loggers=tuple(), save_episodes=True, save_batch_size=256,
                 atol=1e-3, store_from_safe=False,
                 t_start=0):  # Use when loading from a checkpoint

        # create the environment
        self.env, self.test_env = env, env

        # This is important: it allows child classes (that extend this one) to "push up" information
        # that this parent class should log
        self.metadata = {}
        self.record = {'transition_actor': ''}

        self.save_episodes = save_episodes
        self.episode_num = 0
        self.best_ret = 0

        # Create environment
        self.cfg = agent_kwargs
        self.atol = atol
        self.store_from_safe = store_from_safe
        self.file_logger, self.tb_logger = loggers

        self.pi_scheduler = None

        if self.cfg['record_experience']:
            self.save_queue = queue.Queue()
            self.save_batch_size = save_batch_size
            self.record_experience = RecordExperience(
                self.cfg['record_dir'],
                self.cfg['track_name'],
                self.cfg['experiment_name'],
                self.file_logger,
                self)
            self.save_thread = threading.Thread(
                target=self.record_experience.save_thread)
            self.save_thread.start()

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # self.act_limit = self.env.action_space.high[0]

        assert self.cfg['use_encoder_type'] in ['vae'], \
            "Specified encoder type must be in ['vae']"
        speed_hiddens = self.cfg[self.cfg['use_encoder_type']]['speed_hiddens']
        self.feat_dim = self.cfg[self.cfg['use_encoder_type']
                                 ]['latent_dims'] + 1
        self.obs_dim = self.cfg[self.cfg['use_encoder_type']]['latent_dims'] + \
            speed_hiddens[-1] if self.cfg['encoder_switch'] else self.env.observation_space.shape

        self.act_dim = self.env.action_space.shape[0]

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.feat_dim,
                                          act_dim=self.act_dim,
                                          size=self.cfg['replay_size'])

        '''
        ## transform image
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        '''
        self.actor_critic = ActorCritic(self.obs_dim,
                                        self.env.action_space,
                                        self.cfg,
                                        latent_dims=self.obs_dim,
                                        device=DEVICE)

        self.t_start = t_start
        if self.cfg['checkpoint'] and self.cfg['load_checkpoint']:
            self.actor_critic.load_state_dict(
                torch.load(self.cfg['checkpoint']))
            self.episode_num = int(
                self.cfg['checkpoint'].split('.')[-2].split('_')[-1])
            self.file_logger(
                f"Loaded checkpoint {self.cfg['checkpoint']} at episode {self.episode_num}")

        self.actor_critic_target = deepcopy(self.actor_critic)
        self.best_pct = 0

        self.pi_optimizer = Adam(
            self.actor_critic.policy.parameters(),
            lr=self.agent.cfg['lr'])
        self.q_optimizer = Adam(
            self.actor_critic.q_params,
            lr=self.agent.cfg['lr'])
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, 1, gamma=0.5)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.actor_critic.q1(o, a)
        q2 = self.actor_critic.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critic.pi(o2)

            # Target Q-values
            q1_pi_targ = self.actor_critic_target.q1(o2, a2)
            q2_pi_targ = self.actor_critic_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.cfg['gamma'] * \
                (1 - d) * (q_pi_targ - self.cfg['alpha'] * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = (self.replay_buffer.weights * (q1 - backup)**2).mean()
        loss_q2 = (self.replay_buffer.weights * (q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.actor_critic.pi(o)
        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.cfg['alpha'] * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, data):
        #print('Using the update in SAC')
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.actor_critic.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.actor_critic.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                    self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new
                # tensors.
                p_targ.data.mul_(self.cfg['polyak'])
                p_targ.data.add_((1 - self.cfg['polyak']) * p.data)

    def select_action(self, t, obs, state=None, deterministic=False):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > self.cfg['start_steps']:
            a = self.actor_critic.act(obs.to(DEVICE), deterministic)
            a = a  # numpy array...
            self.record['transition_actor'] = 'learner'
        else:
            a = self.env.action_space.sample()
            self.record['transition_actor'] = 'random'
        return a
