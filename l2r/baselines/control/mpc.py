# ========================================================================= #
# Filename:                                                                 #
#    mpc.py                                                                 #
#                                                                           #
# Description:                                                              #
#    an agent using a MPC controller                                        #
# ========================================================================= #

import numpy as np
import torch
import matplotlib.pyplot as plt

from baselines.control.mpc_utils import BikeModel, MPC
from core.templates import AbstractAgent
from envs.env import RacingEnv
from racetracks.mapping import level_2_trackmap

if float(torch.__version__[0:3]) > 1.4:
    raise Exception('MPC agent requires torch version of 1.4 or lower')

# Vehicle parameters
WB = 2.7  # [m]

# State variables
DT = 0.1  # [s] time tick
N_STATE = 4  # [x, y, v, yaw]

# Controller
T = 6
dl = 0.25


class MPCAgent(AbstractAgent):
    """Agent that selects actions using an MPC controller

    :param dict mpc_kwargs: keyword arguments
    """

    def __init__(self, mpc_kwargs):
        self.vt = mpc_kwargs['velocity_target']
        self.step_size = mpc_kwargs['step_size']
        self.plot = mpc_kwargs['plot']
        self.num_episodes = mpc_kwargs['num_episodes']
        self.save_path = mpc_kwargs['save_path']
        self.save_transitions = mpc_kwargs['save_transitions']

        self.car = BikeModel(dt=DT,
                             init_params=np.array([WB, 10, 6]))
        self.controller = MPC(T=T,
                              cost_weight=torch.FloatTensor([1, 1, 1, 16, 0.1, 1]),
                              cost_type='reference')

    def race(self):
        """Race following the environment's step until completion framework
        """
        plt.plot(self.race_x, self.race_y, 'k--')
        if self.save_transitions:
            self._imgs = []
            self._multimodal = []
            self._actions = []

        for e in range(self.num_episodes):
            print('=' * 10 + f' Episode {e+1} of {self.num_episodes} ' + '=' * 10)
            ep_reward, ep_timestep = 0, 0
            obs, info = self.env.reset()
            obs, reward, done, info = self.env.step([0, 1])

            while not done:
                x, y, v, yaw = MPCAgent.unpack_state(obs)

                idx = info['track_idx']
                xref, to2pi_flag = self.get_xref(idx, yaw)

                if to2pi_flag:
                    if yaw < 0:
                        yaw += 2 * np.pi
                x0 = [x, y, v, yaw]  # current state

                print(f'State: @{idx}, loc=({x}, {y}), v={v}, yaw={yaw*180/np.pi}')
                print(f'Target: loc=({xref[0, 0]}, {xref[1, 0]}), v={xref[2, 0]}, yaw={xref[3, 0]*180/np.pi}')

                if self.plot:
                    plt.plot([x], [y], 'bo', markersize=4)
                    l_arr = 10
                    plt.arrow(x, y, l_arr * np.cos(yaw), l_arr * np.sin(yaw),
                              color='r',)  # head_width = 0.5,
                    plt.plot(xref[0, 0], xref[1, 0], 'o')

                # action selection
                _, u_opt = self.controller.forward(
                    self.car,
                    torch.tensor(x0).unsqueeze(0),
                    torch.tensor(xref).transpose(0, 1).unsqueeze(1))

                u = u_opt[0].squeeze().detach().numpy()
                ai, di = u

                # u_opt is a sequence of actions, along the predicted trajectory,
                # so we'll take the first one to give to the environment
                obs, reward, done, info = self.env.step([di, ai])
                ep_reward += reward
                ep_timestep += 1

                if self.save_transitions:
                    (data, img) = obs
                    self._imgs.append(img)
                    self._multimodal.append(data)
                    self._actions.append(np.array([di, ai]))

                    if done:
                        for i in range(len(self._imgs)):
                            filename = f'{self.save_path}/transitions_{i}'
                            np.savez_compressed(
                                filename,
                                img=self._imgs[i],
                                multimodal_data=self._multimodal[i],
                                action=self._actions[i]
                            )

            print(f'Completed episode with total reward: {ep_reward}')
            print(f'Episode info: {info}\n')

    def get_xref(self, idx, yaw, n_targets=T):
        """Get targets.

        :param int idx: index of the raceline the agent is nearest to
        :param float yaw: current vehicle heading
        :return: array of shape (4, n_targets)
        :rtype: numpy array
        """
        # TODO: Interval between waypoint have to change as well
        target_idxs = [(idx + self.step_size * t) %
                       self.raceline_length for t in range(1, 1 + n_targets)]
        target_x = [self.race_x[i] for i in target_idxs]
        target_y = [self.race_y[i] for i in target_idxs]

        target_yaw = np.array([self.race_yaw[i] for i in target_idxs])
        if np.any((target_yaw > 5 / 6 * np.pi) | (target_yaw < -5 / 6 * np.pi)):
            to2pi_flag = True
            target_yaw[target_yaw < 0] += 2 * np.pi
        else:
            to2pi_flag = False

        cdy = abs(target_yaw[-1] - yaw)
        mdy = abs(target_yaw[0] - target_yaw[-3])
        tdy = abs(target_yaw[0] - target_yaw[-1])
        vt = self.vt

        # crude velocity profile
        if tdy > 0.1 or cdy > 0.1:
            vt = self.vt * 0.9
        if tdy > 0.2 or cdy > 0.2:
            vt = self.vt * 0.7
        if tdy > 0.4 or cdy > 0.4:
            vt = self.vt * 0.5
        if tdy > 1.0 or mdy > 1.0:
            vt = self.vt * 0.3

        target_v = [vt] * n_targets

        return np.array([target_x, target_y, target_v, target_yaw], dtype=np.float32), to2pi_flag

    @staticmethod
    def unpack_state(state):
        (state, _) = state
        x = state[16]
        y = state[15]
        v = (state[4]**2 + state[3]**2 + state[5]**2)**0.5
        yaw = np.pi / 2 - state[12]
        # Ensure the yaw is within [-pi, pi)
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        return x, y, v, yaw

    def select_action(self):
        pass

    def create_env(self, env_kwargs, sim_kwargs):
        """Instantiate a racing environment

        :param dict env_kwargs: environment keyword arguments
        :param dict sim_kwargs: simulator setting keyword arguments
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
            provide_waypoints=env_kwargs['provide_waypoints'],
            logger_kwargs=env_kwargs['pose_if_kwargs']
        )

        self.env.make(
            level=sim_kwargs['racetrack'],
            multimodal=env_kwargs['multimodal'],
            driver_params=sim_kwargs['driver_params'],
            camera_params=sim_kwargs['camera_params'],
            sensors=sim_kwargs['active_sensors']
        )

        self.load_track(sim_kwargs['racetrack'])

    def load_track(self, track_name='VegasNorthRoad'):
        """Load trace track

        :param str track_name: 'VegasNorthRoad' or 'Thruxton'
        """
        map_file, _ = level_2_trackmap(track_name)

        # with open(os.path.join(pathlib.Path().absolute(), map_file), 'r') as f:
        #     original_map = json.load(f)

        # np.asarray(original_map['Racing'], dtype=np.float32).T
        raceline = self.env.centerline_arr
        self.race_x = raceline[:, 0]
        self.race_y = raceline[:, 1]
        self.raceline_length = self.race_x.shape[0]

        X_diff = np.concatenate([self.race_x[1:] - self.race_x[:-1],
                                 [self.race_x[0] - self.race_x[-1]]])
        Y_diff = np.concatenate([self.race_y[1:] - self.race_y[:-1],
                                 [self.race_y[0] - self.race_y[-1]]])
        self.race_yaw = np.arctan(Y_diff / X_diff)  # (L-1, n)
        self.race_yaw[X_diff < 0] += np.pi
        # Ensure the yaw is within [-pi, pi)
        self.race_yaw = (self.race_yaw + np.pi) % (2 * np.pi) - np.pi
        # pdb.set_trace()
        # self.race_yaw = np.asarray(original_map['RacingPsi'], dtype=np.float32)
