# ========================================================================= #
# Filename:                                                                 #
#    runner_random.py                                                       #
#                                                                           #
# Description:                                                              #
#    Convenience script to load parameters and train a model.               #
# ========================================================================= #

import sys
import ipdb as pdb
from ruamel.yaml import YAML

from baselines.random import RandomActionAgent

if __name__ == "__main__":

    # load configuration file
    yaml = YAML()
    agent_params = yaml.load(open(sys.argv[1]))
    agent_kwargs = agent_params['agent_kwargs']

    sys_params = yaml.load(open(f"{sys.argv[1].split('/')[0]}/params-env.yaml"))
    env_kwargs = sys_params['env_kwargs']
    sim_kwargs = sys_params['sim_kwargs']

    # instantiate and run agent
    agent = RandomActionAgent(agent_kwargs)
    agent.create_env(env_kwargs, sim_kwargs)
    agent.race()
