import torch
import numpy as np
from torch.optim import Adam
import itertools


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
                
                info['metrics']['episodic_return'] = ep_ret
                info['metrics']['ep_n_steps'] = t - t_start
                self.agent.metadata['info'] = info
                self.agent.episode_num += 1
                msg = f'[Ep {self.agent.episode_num }] {self.agent.metadata}'
                self.agent.file_logger(msg)


                self.agent.tb_logger.log(info['metrics'], self.agent.episode_num)

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
            info['metrics']['episodic_return'] = ep_ret
            info['metrics']['ep_n_steps'] = n_val_steps
            self.metadata['info'] = info

            self.tb_logger.add_scalar('val/episodic_return', ep_ret, n_eps)
            self.tb_logger.add_scalar('val/ep_n_steps', n_val_steps, n_eps)
            # The metrics are not calculated if the environment is manually
            # terminated.
            try:
                self.tb_logger.log(info['metrics'], n_eps)
                if 'safety_info' in self.metadata:
                    self.tb_logger.log(self.metadata['safety_info'], n_eps)
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

