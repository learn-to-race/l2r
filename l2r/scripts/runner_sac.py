# ========================================================================= #
# Filename:                                                                 #
#    runner_sac.py                                                          #
#                                                                           #
# Description:                                                              #
#    Convenience script to load parameters and train an sac agent           #
# ========================================================================= #
import json, os, sys

from ruamel.yaml import YAML
from datetime import date, datetime, timezone

l2r_path = os.path.abspath(os.path.join(''))
if l2r_path not in sys.path: sys.path.append(l2r_path)
import ipdb as pdb

from baselines.rl.sac import SACAgent
from common.utils import setup_logging
from envs.env import RacingEnv

if __name__ == "__main__":

    # load configuration file
    yaml = YAML()
    agent_params = yaml.load(open(sys.argv[1]))
    agent_kwargs = agent_params['agent_kwargs']

    sys_params = yaml.load(open(f"{sys.argv[1].split('/')[0]}/params-env.yaml"))
    env_kwargs = sys_params['env_kwargs']
    sim_kwargs = sys_params['sim_kwargs']

    # overrrides
    env_kwargs['action_if_kwargs']['max_accel'] = 6.
    env_kwargs['action_if_kwargs']['min_accel'] = -2
    env_kwargs['eval_mode'] = False

    # create results directory
    save_path = agent_kwargs['model_save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + 'params.json', 'w') as f:
        json = json.dumps([agent_params, sys_params])
        f.write(json)

    loggers = setup_logging(save_path, agent_kwargs['experiment_name'], True)

    # create the environment               
    env = RacingEnv(env_kwargs, sim_kwargs)
    env.make()                             

    # deploy an agent
    agent = SACAgent(env, agent_kwargs, loggers=loggers)
    agent.sac_train()

