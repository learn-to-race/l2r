# ========================================================================= #
# Filename:                                                                 #
#    runner.py                                                              #
#                                                                           #
# Description:                                                              # 
#    Convenience script to load parameters and train a model.               #
# ========================================================================= #
import json
import math
import os
import sys

from ruamel.yaml import YAML

from baselines.mpc import mpc
from baselines.models.vae import *
from envs.env import RacingEnv

from baselines.components.cubic_spline_planner import calc_spline_course

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed

def get_straight_course(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = calc_spline_course(ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


if __name__ == "__main__":
    """
    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """

    # load configuration file
    yaml = YAML()
    params = yaml.load(open(sys.argv[1]))

    env_kwargs = params['env_kwargs']
    sim_kwargs = params['sim_kwargs']
    mpc_kwargs = params['mpc_kwargs']

    # create the environment
    env = RacingEnv(
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

    env.make(
        level=sim_kwargs['racetrack'],
        multimodal=env_kwargs['multimodal'],
        driver_params=sim_kwargs['driver_params'],
        camera_params=sim_kwargs['camera_params'],
        sensors=sim_kwargs['active_sensors'],
    )

    # create results directory
    save_path = mpc_kwargs['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+'params.json', 'w') as f:
        json = json.dumps(params)
        f.write(json)

    dl = 1.0  # course tick

    cx, cy, cyaw, ck = get_straight_course(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    init_package = {
        "num_episodes": 200,
        "cx": cx,
        "cy": cy,
        "cyaw": cyaw,
        "ck": ck,
        "sp": sp,
        "dl": dl,
        "initial_state": initial_state
    }

    # train an agent
    mpc(env=env, **mpc_kwargs)
    