# ========================================================================= #
# Filename:                                                                 #
#    MPC.py                                                                 #
#                                                                           #
# Description:                                                              # 
#    an agent used a MPC controller                                         #
# ========================================================================= #

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods

import ipdb as pdb

from core.templates import AbstractAgent
from envs.env import RacingEnv

##################################
# env vars
##################################

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

##################################
##################################

def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False


def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))
        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref

def update_state(state, a, delta):

    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state. v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state. v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


class MPCAgent(AbstractAgent):
    """MPC controller
    """
    def __init__(self, init_package, num_episodes):

        self.num_episodes = init_package["num_episodes"]
        self.cx = init_package['cx']
        self.cy = init_package['cy']
        self.cyaw = init_package['cyaw']
        self.ck = init_package['ck']
        self.sp = init_package['sp']
        self.dl = init_package['dl']
        self.state = init_package['initial_state']

        self.car = bike_model.BikeModel(dt = DT, init_params = np.array([WB, 1, 1]))
        self.controller = bike_model.MPC(cost_weight = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.01, 0.01]))
        self.goal = [self.cx[-1], self.cy[-1]]

        self.state = initial_state

        # initial yaw compensation
        if self.state.yaw - self.cyaw[0] >= math.pi:
            self.state.yaw -= math.pi * 2.0
        elif self.state.yaw - self.cyaw[0] <= -math.pi:
            self.state.yaw += math.pi * 2.0

        time = 0.0
        self.x = [self.state.x]
        self.y = [self.state.y]
        self.yaw = [self.state.yaw]
        self.v = [self.state.v]
        self.t = [0.0]
        self.d = [0.0]
        self.a = [0.0]
        
        self.target_ind, _ = calc_nearest_index(self.state, self.cx, self.cy, self.cyaw, 0)

        self.ox, self.oy, self.odelta, self.oa = None, None, None, None

        self.cyaw = smooth_yaw(self.cyaw)


    def race(self):
        """Race using an MPC controller
        """
        for e in range(self.num_episodes):
            print('='*10+f' Episode {e+1} of {self.num_episodes} '+'='*10)
            ep_reward, ep_timestep = 0, 0
            state, done, info = self.env.reset(), False, {'waypoints': None}

            while not done:
                self.xref, self.target_ind, self.dref = calc_ref_trajectory(self.state, 
                        self.cx, 
                        self.cy, 
                        self.cyaw, 
                        self.ck, 
                        self.sp, 
                        self.dl, 
                        self.target_ind)
                
                self.x0 = [self.state.x, state.y, state.v, state.yaw]  # current state
                
                ## Using my model
                self.x_opt, self.u_opt = controller.forward(self.car, 
                        torch.tensor(self.x0).unsqueeze(0), 
                        torch.tensor(self.xref).transpose(0, 1).unsqueeze(1))
                
                #if odelta is not None:
                #    di, ai = odelta[0], oa[0]
                self.u = self.u_opt[0].squeeze().detach().numpy()
                
                self.ai, self.di = self.u
                
                self.state = update_state(self.state, self.ai, self.di)
                self.time = self.time + DT

                self.x.append(self.state.x)
                self.y.append(self.state.y)
                self.yaw.append(self.state.yaw)
                self.v.append(self.state.v)
                self.t.append(self.time)
                self.d.append(self.di)
                self.a.append(self.ai)

                #action = self.select_action(state, info)

                # u_opt is a sequence of actions, along the predicted trajectory, 
                # so we'll take the first one to give to the environment
                self.state, reward, done, info = self.env.step(self.u)
                ep_reward += reward
                ep_timestep += 1

                #if check_goal(state, goal, target_ind, len(cx)):
                #    print("Goal")
                #    break

                #if show_animation:  # pragma: no cover
                #    plt.cla()
                #    # for stopping simulation with the esc key.
                #    plt.gcf().canvas.mpl_connect('key_release_event',
                #            lambda event: [exit(0) if event.key == 'escape' else None])
                #    if ox is not None:
                #        plt.plot(ox, oy, "xr", label="MPC")
                #    plt.plot(cx, cy, "-r", label="course")
                #    plt.plot(x, y, "ob", label="trajectory")
                #    plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
                #    plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                #    plot_car(state.x, state.y, state.yaw, steer=di)
                #    plt.axis("equal")
                #    plt.grid(True)
                #    plt.title("Time[s]:" + str(round(time, 2))
                #              + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
                #    plt.pause(0.0001)

            print(f'Completed episode with total reward: {ep_reward}')
            print(f'Episode info: {info}\n')

        return self.t, self.x, self.y, self.yaw, self.v, self.d, self.a

    def select_action(self):
    	pass

    #def select_action(self, state, info):
    #    """Select an action using a MPC controller

    #    :return: random action to take
    #    :rtype: numpy array
    #    """
    #    (state, _) = state
    #    vx, vy, vz = state[4], state[3], state[5] # directional velocity
    #    v = (vx**2 + vy**2 + vz**2)**0.5          # magnitude of velocity
    #    e, n = state[16], state[15]               # current (x,y)
    #    waypoints = info['waypoints']             # list of 3 future waypoints -> (x,y) pairs
    #    yaw = state[12]

    #    if waypoints:
    #        print(waypoints)

    #    A, B = self.bike_model(state)
    #    x_lqr, u_lqr = self.controller(state, A, B)

    #    # select action [steer, accel], hardcoded for now...
    #    # bounds at (-1., 1.) for both
    #    #action = [0.0, 0.5]
    #    return u_lqr #action

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
