import gym


class EnvWrapper(gym.Wrapper):
    """A sample wrapper that may be particularly useful for pre-processing
    observations from the environment, such as encoding images
    """

    def __init__(self, env, encoder, device):
        super().__init__(env)
        self.encoder = encoder
        self.device = device
        self.encoder.eval()

    def reset(self):
        obs = super().reset()
        return self.observation(obs)

    def step(self, action):
        obs, rew, done, _ = super().step(action)
        return self.observation(obs), rew, done, {}

    def observation(self, observation):
        img = observation["CameraFrontRGB"]
        return self.encoder.encode_raw(img[None], self.device)[0]
