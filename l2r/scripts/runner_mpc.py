# ========================================================================= #
# Filename:                                                                 #
#    runner_mpc.py                                                          #
#                                                                           #
# Description:                                                              #
#    Convenience script to load parameters and run a MPC agent              #
# ========================================================================= #

import sys

from ruamel.yaml import YAML

from baselines.control.mpc import MPCAgent

sys.path.insert(0, "../")

if __name__ == "__main__":

    # load configuration file
    yaml = YAML()
    params = yaml.load(open(sys.argv[1]))

    env_kwargs = params['env_kwargs']
    sim_kwargs = params['sim_kwargs']
    mpc_kwargs = params['mpc_kwargs']

    # create and run agent
    agent = MPCAgent(mpc_kwargs)
    agent.create_env(env_kwargs, sim_kwargs)
    agent.race()
