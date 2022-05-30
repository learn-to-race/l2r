# ========================================================================= #
# Filename:                                                                 #
#    runner_mpc.py                                                          #
#                                                                           #
# Description:                                                              #
#    Convenience script to load parameters and run a MPC agent              #
# ========================================================================= #

import sys
from ruamel.yaml import YAML
import ipdb as pdb

# pdb.set_trace()

from baselines.control.mpc import MPCAgent

sys.path.insert(0, "../")


if __name__ == "__main__":

    # load configuration file
    yaml = YAML()
    agent_params = yaml.load(open(sys.argv[1]))
    agent_kwargs = agent_params['agent_kwargs']

    sys_params = yaml.load(
        open(f"{sys.argv[1].split('/')[0]}/params-env.yaml"))
    env_kwargs = sys_params['env_kwargs']
    sim_kwargs = sys_params['sim_kwargs']

    # overrrides
    env_kwargs['action_if_kwargs']['max_accel'] = 4.
    env_kwargs['action_if_kwargs']['min_accel'] = -6.
    env_kwargs['action_if_kwargs']['max_steer'] = 1.0
    env_kwargs['action_if_kwargs']['min_steer'] = -1.0
    env_kwargs['obs_delay'] = 0.
    env_kwargs['provide_waypoints'] = True

    # create and run agent
    agent = MPCAgent(agent_kwargs)
    agent.create_env(env_kwargs, sim_kwargs)
    agent.race()
