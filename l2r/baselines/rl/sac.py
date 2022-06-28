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
from network_architecture.sac import ActorCritic
from cv.vae import VAE
from common.utils import RecordExperience
from baselines.rl.simplebuffer import ReplayBuffer

DEVICE = torch.device('cuda') if torch.cuda.is_available() else "cpu"

#seed = np.random.randint(255)
# torch.manual_seed(seed)
# np.random.seed(seed)

class VAEBackbone():
    def __init__(self, cfg):
        # vision encoder
        if cfg['use_encoder_type'] == 'vae':
            self.model = VAE(
                im_c=cfg['vae']['im_c'],
                im_h=cfg['vae']['im_h'],
                im_w=cfg['vae']['im_w'],
                z_dim=cfg['vae']['latent_dims']
            )
            self.model.load_state_dict(
                torch.load(
                    cfg['vae']['vae_chkpt_statedict'],
                    map_location=DEVICE))
        else:
            raise NotImplementedError

        self.model.to(DEVICE)
    
    def encode(self, o):
        state, img = o

        if self.cfg['use_encoder_type'] == 'vae':
            img_embed = self.model.encode_raw(np.array(img), DEVICE)[0][0]
            speed = torch.tensor((state[4]**2 +
                                  state[3]**2 +
                                  state[5]**2)**0.5).float().reshape(1, -
                                                                     1).to(DEVICE)
            out = torch.cat([img_embed.unsqueeze(0), speed],
                            dim=-1).squeeze(0)  # torch.Size([33])
            self.using_speed = 1
        else:
            raise NotImplementedError

        assert not torch.sum(torch.isnan(out)), "found a nan value"
        out[torch.isnan(out)] = 0

        return out


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
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
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






class SACRunner():

    def __init__(self, agent, env, encoder):
        self.agent = agent
        self.env = env
        self.vision_encoder = encoder

    def train(self):

        # List of parameters for both Q-networks (save this for convenience)
        self.agent.q_params = itertools.chain(
            self.agent.actor_critic.q1.parameters(),
            self.agent.actor_critic.q2.parameters())

        # Set up optimizers for policy and q-function
        self.agent.pi_optimizer = Adam(
            self.agent.actor_critic.policy.parameters(),
            lr=self.agent.cfg['lr'])
        self.agent.q_optimizer = Adam(self.agent.q_params, lr=self.agent.cfg['lr'])
        self.agent.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.agent.pi_optimizer, 1, gamma=0.5)

        # Freeze target networks with respect to optimizers (only update via
        # polyak averaging)
        for p in self.agent.actor_critic_target.parameters():
            p.requires_grad = False

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

        # Prepare for interaction with environment
        # start_time = time.time()
        best_ret, ep_ret, ep_len = 0, 0, 0
        camera, feat, state = self.agent._reset()
        camera, feat, state, r, d, info = self.env._step([0, 1])
        feat = self.vision_encoder(feat)

        experience = []
        speed_dim = 1 if self.agent.using_speed else 0
        assert len(feat) == self.agent.cfg[self.agent.cfg['use_encoder_type']]['latent_dims'] + \
            speed_dim, "'o' has unexpected dimension or is a tuple"

        t_start = self.agent.t_start
        # Main loop: collect experience in env and update/log each epoch
        for t in range(self.agent.t_start, self.agent.cfg['total_steps']):

            a = self.agent.select_action(t, feat, state)

            # Step the env
            camera2, feat2, state2, r, d, info = self.env._step(a)
            feat2 = self.vision_encoder(feat2)
            # Check that the camera is turned on
            assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

            # Prevents the agent from getting stuck by sampling random actions
            # self.agent.atol for SafeRandom and SPAR are set to -1 so that this
            # condition does not activate
            if np.allclose(state2[15:16], state[15:16],
                           atol=self.agent.atol, rtol=0):
                
                self.agent.file_logger("Sampling random action to get unstuck")
                a = self.agent.env.action_space.sample()

                # Step the env
                camera2, feat2, state2, r, d, info = self.env._step(a)
                feat2 = self.vision_encoder(feat2)
                ep_len += 1

            state = state2
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.agent.cfg['max_ep_len'] else d

            # Store experience to replay buffer
            if (not np.allclose(state2[15:16],
                                state[15:16],
                                atol=3e-1,
                                rtol=0)) | (r != 0):
                self.agent.replay_buffer.store(feat, a, r, feat2, d)
            else:
                print('Skip')

            if self.agent.cfg['record_experience']:
                self.agent.recording = {
                    'step': t,
                    'nearest_idx': self.agent.env.nearest_idx,
                    'camera': camera,
                    'feature': feat.detach().cpu().numpy(),
                    'state': state,
                    'action_taken': a,
                    'next_camera': camera2,
                    'next_feature': feat2.detach().cpu().numpy(),
                    'next_state': state2,
                    'reward': r,
                    'episode': self.agent.episode_num,
                    'stage': 'training',
                    'done': d,
                    'transition_actor': self.agent.record['transition_actor'],
                    'metadata': info}

                experience.append(self.agent.recording)

                # quickly pass data to save thread
                # if len(experience) == self.agent.save_batch_size:
                #    self.agent.save_queue.put(experience)
                #    experience = []

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            feat = feat2
            state = state2  # in case we, later, wish to store the state in the replay as well
            camera = camera2  # in case we, later, wish to store the state in the replay as well

            # Update handling
            if (t >= self.agent.cfg['update_after']) & (
                    t % self.agent.cfg['update_every'] == 0):
                for j in range(self.agent.cfg['update_every']):
                    batch = self.agent.replay_buffer.sample_batch(
                        self.agent.cfg['batch_size'])
                    self.agent.update(data=batch)

            if ((t + 1) % self.agent.cfg['eval_every'] == 0):
                # eval on test environment
                #val_returns = self.agent.eval(t // self.agent.cfg['eval_every'])
                # Reset
                camera, feat, state = self.env._reset()
                ep_ret, ep_len, self.agent.metadata, experience = 0, 0, {}, []
                t_start = t + 1
                camera, feat, state2, r, d, info = self.env._step([0, 1])
                feat = self.vision_encoder(feat)

            # End of trajectory handling
            if d or (ep_len == self.agent.cfg['max_ep_len']):
                self.agent.metadata['info'] = info
                self.agent.episode_num += 1
                msg = f'[Ep {self.agent.episode_num }] {self.agent.metadata}'
                self.agent.file_logger(msg)

                self.agent.tb_logger.add_scalar(
                    'train/episodic_return', ep_ret, self.agent.episode_num)
                #self.agent.tb_logger.add_scalar('train/ep_pct_complete', self.agent.metadata['info']['metrics']['pct_complete'], self.agent.episode_num)
                self.agent.tb_logger.add_scalar(
                    'train/ep_total_time',
                    self.agent.metadata['info']['metrics']['total_time'],
                    self.agent.episode_num)
                self.agent.tb_logger.add_scalar(
                    'train/ep_total_distance',
                    self.agent.metadata['info']['metrics']['total_distance'],
                    self.agent.episode_num)
                self.agent.tb_logger.add_scalar(
                    'train/ep_avg_speed',
                    self.agent.metadata['info']['metrics']['average_speed_kph'],
                    self.agent.episode_num)
                self.agent.tb_logger.add_scalar(
                    'train/ep_avg_disp_err',
                    self.agent.metadata['info']['metrics']['average_displacement_error'],
                    self.agent.episode_num)
                self.agent.tb_logger.add_scalar(
                    'train/ep_traj_efficiency',
                    self.agent.metadata['info']['metrics']['trajectory_efficiency'],
                    self.agent.episode_num)
                self.agent.tb_logger.add_scalar(
                    'train/ep_traj_admissibility',
                    self.agent.metadata['info']['metrics']['trajectory_admissibility'],
                    self.agent.episode_num)
                self.agent.tb_logger.add_scalar(
                    'train/movement_smoothness',
                    self.agent.metadata['info']['metrics']['movement_smoothness'],
                    self.agent.episode_num)
                self.agent.tb_logger.add_scalar(
                    'train/ep_n_steps', t - t_start, self.agent.episode_num)

                # Quickly dump recently-completed episode's experience to the multithread queue,
                # as long as the episode resulted in "success"
                # and self.agent.metadata['info']['success']:
                if self.agent.cfg['record_experience']:
                    self.agent.file_logger("Writing experience")
                    self.agent.save_queue.put(experience)

                # Reset
                camera, feat, state = self.env._reset()
                ep_ret, ep_len, self.agent.metadata, experience = 0, 0, {}, []
                t_start = t + 1
                camera, feat, state2, r, d, info = self.env._step([0, 1])
                feat = self.vision_encoder(feat)

    def eval(self, n_eps):
        print('Evaluation:')
        val_ep_rets = []

        # Not implemented for logging multiple test episodes
        assert self.cfg['num_test_episodes'] == 1

        for j in range(self.cfg['num_test_episodes']):
            camera, features, state = self.env._reset(test=True)
            d, ep_ret, ep_len, n_val_steps, self.metadata = False, 0, 0, 0, {}
            camera, features, state2, r, d, info = self.env._step(
                [0, 1], test=True)
            features = self.vision_encoder(features)
            experience, t = [], 0

            while (not d) & (ep_len <= self.cfg['max_ep_len']):
                # Take deterministic actions at test time
                a = self.select_action(1e6, features, state, True)
                camera2, features2, state2, r, d, info = self.env._step(
                    a, test=True)
                features2 = self.vision_encoder(features2)

                # Check that the camera is turned on
                assert (np.mean(camera2) > 0) & (np.mean(camera2) < 255)

                ep_ret += r
                ep_len += 1
                n_val_steps += 1

                # Prevent the agent from being stuck
                if np.allclose(state2[15:16], state[15:16],
                               atol=self.atol, rtol=0):
                    self.file_logger("Sampling random action to get unstuck")
                    a = self.test_env.action_space.sample()
                    # Step the env
                    camera2, features2, state2, r, d, info = self.env._step(a)
                    features2 = self.vision_encoder(features2)
                    
                    ep_len += 1

                if self.cfg['record_experience']:
                    self.recording = {
                        'step': t,
                        'nearest_idx': self.test_env.nearest_idx,
                        'camera': camera,
                        'feature': features.detach().cpu().numpy(),
                        'state': state,
                        'action_taken': a,
                        'next_camera': camera2,
                        'next_feature': features2.detach().cpu().numpy(),
                        'next_state': state2,
                        'reward': r,
                        'episode': self.episode_num,
                        'stage': 'evaluation',
                        'done': d,
                        'transition_actor': self.record['transition_actor'],
                        'metadata': info}

                    experience.append(self.recording)

                features = features2
                camera = camera2
                state = state2
                t += 1

            self.file_logger(f'[eval episode] {info}')

            val_ep_rets.append(ep_ret)
            self.metadata['info'] = info

            self.tb_logger.add_scalar('val/episodic_return', ep_ret, n_eps)
            self.tb_logger.add_scalar('val/ep_n_steps', n_val_steps, n_eps)
            # The metrics are not calculated if the environment is manually
            # terminated.
            try:
                self.tb_logger.add_scalar(
                    'val/ep_pct_complete', info['metrics']['pct_complete'], n_eps)
                self.tb_logger.add_scalar(
                    'val/ep_total_time', info['metrics']['total_time'], n_eps)
                self.tb_logger.add_scalar(
                    'val/ep_total_distance',
                    info['metrics']['total_distance'],
                    n_eps)
                self.tb_logger.add_scalar(
                    'val/ep_avg_speed', info['metrics']['average_speed_kph'], n_eps)
                self.tb_logger.add_scalar(
                    'val/ep_avg_disp_err',
                    info['metrics']['average_displacement_error'],
                    n_eps)
                self.tb_logger.add_scalar(
                    'val/ep_traj_efficiency',
                    info['metrics']['trajectory_efficiency'],
                    n_eps)
                self.tb_logger.add_scalar(
                    'val/ep_traj_admissibility',
                    info['metrics']['trajectory_admissibility'],
                    n_eps)
                self.tb_logger.add_scalar(
                    'val/movement_smoothness',
                    info['metrics']['movement_smoothness'],
                    n_eps)
            except BaseException:
                pass

            # TODO: Find a better way: requires knowledge of child class API :(
            if 'safety_info' in self.metadata:
                self.tb_logger.add_scalar(
                    'val/ep_interventions',
                    self.metadata['safety_info']['ep_interventions'],
                    n_eps)

            # Quickly dump recently-completed episode's experience to the multithread queue,
            # as long as the episode resulted in "success"
            # and self.metadata['info']['success']:
            if self.cfg['record_experience']:
                self.file_logger("writing experience")
                self.save_queue.put(experience)

        # Save if best (or periodically)
        if (ep_ret > self.best_ret):  # and ep_ret > 100):
            path_name = f"{self.cfg['save_path']}/best_{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            self.file_logger(
                f'New best episode reward of {round(ep_ret, 1)}! Saving: {path_name}')
            self.best_ret = ep_ret
            torch.save(self.actor_critic.state_dict(), path_name)
            path_name = f"{self.cfg['save_path']}/best_{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            try:
                # Try to save Safety Actor-Critic, if present
                torch.save(self.safety_actor_critic.state_dict(), path_name)
            except BaseException:
                pass

        elif self.save_episodes and (n_eps + 1 % self.cfg['save_freq'] == 0):
            path_name = f"{self.cfg['save_path']}/{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            self.file_logger(
                f"Periodic save (save_freq of {self.cfg['save_freq']}) to {path_name}")
            torch.save(self.actor_critic.state_dict(), path_name)
            path_name = f"{self.cfg['save_path']}/{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            try:
                # Try to save Safety Actor-Critic, if present
                torch.save(self.safety_actor_critic.state_dict(), path_name)
            except BaseException:
                pass

        if self.best_pct < info['metrics']['pct_complete']:
            for cutoff in [93, 100]:
                if (self.best_pct < cutoff) & (
                        info['metrics']['pct_complete'] >= cutoff):
                    self.pi_scheduler.step()
            self.best_pct = info['metrics']['pct_complete']

        return val_ep_rets


class EnvWrapper():
    def __init__(self, test_env, train_env):
        self.test_env = test_env
        self.train_env = train_env
    
    def _step(self, a, test=False):
        o, r, d, info = self.test_env.step(a) if test else self.train_env.step(a)
        return o[1], o, o[0], r, d, info

    def _reset(self, test=False):
        camera = 0
        while (np.mean(camera) == 0) | (np.mean(camera) == 255):
            obs = self.test_env.reset(random_pos=False) \
                if test else self.env.reset(random_pos=True)
            (state, camera), _ = obs
        return camera, (state, camera), state
