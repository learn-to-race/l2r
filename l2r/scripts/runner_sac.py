# ========================================================================= #
# Filename:                                                                 #
#    runner_sac.py                                                          #
#                                                                           #
# Description:                                                              #
#    Convenience script to load parameters and train an sac agent           #
# ========================================================================= #
import json
import os
import sys

from ruamel.yaml import YAML

from baselines.rl.sac import sac
from envs.env import RacingEnv

if __name__ == "__main__":

    # load configuration file
    yaml = YAML()
    agent_params = yaml.load(open(sys.argv[1]))
    agent_kwargs = agent_params['agent_kwargs']

    sys_params = yaml.load(open(f"{(sys.argv[1]).split['/'][0]}/params-env.yaml"))
    env_kwargs = sys_params['env_kwargs']
    sim_kwargs = sys_params['sim_kwargs']

    # create the environment
    env = RacingEnv(env_kwargs, sim_kwargs)
    env.make()

    # create results directory
    save_path = agent_kwargs['model_save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + 'params.json', 'w') as f:
        json = json.dumps([agent_params, sys_params])
        f.write(json)

    # train an agent
    sac(env=env, **agent_kwargs)
