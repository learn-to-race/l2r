"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.

For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version

Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
import os
from copy import deepcopy

import cv2
import numpy as np
import torch
from torch.optim import Adam

import baselines.core as core

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs.detach().cpu().numpy()
        self.obs2_buf[self.ptr] = next_obs.detach().cpu().numpy()
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for k, v in batch.items()}


def sac(env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        total_steps=1_000_000, replay_size=int(1e6), gamma=0.99, polyak=0.995,
        lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, update_after=1000,
        num_updates=1, num_test_episodes=10, max_ep_len=1000, logger_kwargs=dict(),
        save_freq=1, encoder_path=None, im_w=None, im_h=None, latent_dims=None,
        save_path=None, checkpoint=None, save_episodes=None, inference_only=False):
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

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load image encoder
    if encoder_path:
        vae = torch.load(encoder_path)
        vae.to(DEVICE)

    # Create run log file
    with open(f'{save_path}run_log.txt', 'w') as f:
        f.write('*' * 32 + '   Run Log   ' + '*' * 32 + '\n')

    # Create environment
    env, test_env = env, env
    obs_dim = latent_dims if encoder_path else env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    if checkpoint:
        ac = torch.load(checkpoint)
        ep_num = int(''.join(filter(str.isdigit, os.path.split(checkpoint)[-1])))
    else:
        ac = actor_critic(obs_dim, env.action_space,
                          latent_dims=latent_dims, device=DEVICE, **ac_kwargs)
        ep_num = 0

    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(o, deterministic)

    def _step(a, test=False):
        o, r, d, info = test_env.step(a) if test else env.step(a)
        return _encode(o), r, d, info

    def _reset(test=False):
        o = test_env.reset() if test else env.reset()
        return _encode(o)

    def _encode(o):
        o = torch.as_tensor(cv2.resize(o, (im_w, im_h)),
                            dtype=torch.float32, device=DEVICE)
        o = o / 255
        o, _, _ = vae.encode(o.view(1, 3, im_w, im_h))
        return o.squeeze()

    def log(msg):
        print(msg)
        with open(f'{save_path}run_log.txt', 'a') as f:
            f.write(msg + '\n')

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = _reset(test=True), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                a = get_action(o, True)
                o, r, d, info = _step(a, test=True)
                ep_ret += r
                ep_len += 1

            print(f'[eval episode] {info}')

    if inference_only:
        test_agent()

    else:
        # Prepare for interaction with environment
        # start_time = time.time()
        best_ret, ep_ret, ep_len = 0, 0, 0
        o = _reset()

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > start_steps:
                a = get_action(o)
            else:
                a = env.action_space.sample()

            # Step the env
            o2, r, d, info = _step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                ep_num += 1
                msg = f'[Ep {ep_num}] {info}'
                log(msg)

                # Update
                if t >= update_after:
                    for j in range(num_updates):
                        batch = replay_buffer.sample_batch(batch_size)
                        update(data=batch)

                # Save if best (or periodically)
                if (ep_ret > best_ret and ep_ret > 250):
                    print(f'New best episode reward of {round(ep_ret,1)}!')
                    best_ret = ep_ret
                    path_name = f'{save_path}sac_episode_{ep_num}.pt'
                    torch.save(ac, path_name)

                elif save_episodes and ep_num in save_episodes:
                    print('Periodically saving')
                    path_name = f'{save_path}sac_episode_{ep_num}.pt'
                    torch.save(ac, path_name)

                # Reset
                o, ep_ret, ep_len = _reset(), 0, 0
