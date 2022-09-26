import logging
import socket
from l2r import build_env
from l2r import RacingEnv

L2R_HOST = "0.0.0.0"
ARRIVAL_SIM_HOST = "0.0.0.0"


def race_n_episodes(env: RacingEnv, num_episodes: int = 5):
    """Complete an episode in the environment"""

    for ep in range(num_episodes):
        logging.info(f"Episode {ep+1} of {num_episodes}")

        obs = env.reset()
        total_reward = 0

        while True:
            obs, reward, done, info = env.step(action=[0.00, 1.00])
            total_reward += reward
            if done:
                logging.info(f"Completed episode with reward: {total_reward}")
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Build environment
    env = build_env(
        levels=["Thruxton"],
        controller_kwargs={"ip": ARRIVAL_SIM_HOST},
        camera_cfg=[
            {
                "name": "CameraFrontRGB",
                "Addr": "tcp://0.0.0.0:8008",
                "Width": 256,
                "Height": 192,
                "sim_addr": f"tcp://{ARRIVAL_SIM_HOST}:18008",
            }
        ],
        action_cfg={"ip": ARRIVAL_SIM_HOST, "port": 17077},
        pose_cfg={"port": 17078},
        env_ip=L2R_HOST,
    )

    race_n_episodes(env=env)
