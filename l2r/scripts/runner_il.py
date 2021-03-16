# ========================================================================= #
# Filename:                                                                 #
#    runner.py                                                              #
#                                                                           #
# Description:                                                              # 
#    Convenience script to load parameters and train a model.               #
# ========================================================================= #


#import yaml
import os, sys, argparse
from torch.utils.data import DataLoader
from ruamel.yaml import YAML

from baselines.il.data.il_dataset import ILDataset
from baselines.il import ILAgent

def main(args):
    assert args.exp is not None, "FATAL: "+__file__+": Specify an experiment config file:\npython scripts/runner_il.py configs/params_il.yaml" 

    # load configuration file
    yaml = YAML()
    params = yaml.load(open(args.exp))

    print("######### Configuration #########")
    print(yaml.dump(params, default_flow_style=False))
    print("#################################")

    env_kwargs = params['env_kwargs']
    sim_kwargs = params['sim_kwargs']
    il_kwargs = params['il_kwargs']

    train = ILDataset(il_kwargs['DATASET']['LOCATION'], 
            il_kwargs['DATASET']['NAME'], 
            il_kwargs['DATASET']['SPLIT']['TRAIN'])

    val = ILDataset(il_kwargs['DATASET']['LOCATION'], 
            il_kwargs['DATASET']['NAME'], 
            il_kwargs['DATASET']['SPLIT']['VAL'])

    train = DataLoader(train, params['il_kwargs']['TRAIN_BS'], 
                       num_workers=params['il_kwargs']['CPU'], 
                       pin_memory=params['il_kwargs']['PIN'],
                       shuffle=True)

    val = DataLoader(val, batch_size=params['il_kwargs']['VAL_BS'], 
                     num_workers=params['il_kwargs']['CPU'], 
                     pin_memory=params['il_kwargs']['PIN'],
                     shuffle=False)

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
    save_path = il_kwargs['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+'params.json', 'w') as f:
        json = json.dumps(params)
        f.write(json)

    # instantiate agent
    agent = ILAgent(training_kwargs)
    agent.create_env(env_kwargs, sim_kwargs)

    if il_kwargs['il_train_first']:

        # train agent with imitation learning
        agent.train(env=env, **il_kwargs)
    
    # run agent on the track
    agent.run(env=env, **il_kwargs)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-e', '--exp', type=str)
    args = argparser.parse_args()
    main(args)
