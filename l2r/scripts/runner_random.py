# ========================================================================= #
# Filename:                                                                 #
#    runner_random.py                                                       #
#                                                                           #
# Description:                                                              #
#    Convenience script to load parameters and train a model.               #
# ========================================================================= #

import sys
from ruamel.yaml import YAML

from baselines.random import RandomActionAgent

if __name__ == "__main__":

    # load configuration file
    yaml = YAML()
    params = yaml.load(open(sys.argv[1]))

    env_kwargs = params['env_kwargs']
    sim_kwargs = params['sim_kwargs']
    training_kwargs = params['training_kwargs']

    # instantiate and train agent
    agent = RandomActionAgent(training_kwargs)
    agent.create_env(env_kwargs, sim_kwargs)
    agent.race()
