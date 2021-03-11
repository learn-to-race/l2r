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

from baselines.mpc import MPCAgent

if __name__ == "__main__":

    # load configuration file
    yaml = YAML()
    params = yaml.load(open(sys.argv[1]))

    env_kwargs = params['env_kwargs']
    sim_kwargs = params['sim_kwargs']
    mpc_kwargs = params['mpc_kwargs']

    # create results directory
    save_path = mpc_kwargs['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+'params.json', 'w') as f:
        data = json.dumps(params)
        f.write(data)

    agent = MPCAgent(mpc_kwargs)
    agent.create_env(env_kwargs, sim_kwargs)
    agent.race()

    