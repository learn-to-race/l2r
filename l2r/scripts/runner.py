# ========================================================================= #
# Filename:                                                                 #
#    runner.py                                                              #
#                                                                           #
# Description:                                                              #
#    Convenience script to load parameters and train a model.               #
# ========================================================================= #
import json
import os
import sys

from ruamel.yaml import YAML

from baselines.sac import sac
from envs.env import RacingEnv

if __name__ == "__main__":

    # load configuration file
    yaml = YAML()
    params = yaml.load(open(sys.argv[1]))

    env_kwargs = params['env_kwargs']
    sim_kwargs = params['sim_kwargs']
    sac_kwargs = params['sac_kwargs']

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
    save_path = sac_kwargs['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + 'params.json', 'w') as f:
        json = json.dumps(params)
        f.write(json)

    # train an agent
    sac(env=env, **sac_kwargs)
