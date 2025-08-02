import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OrientationEnv(gym.Env):
    """A simple environment for teaching an RL agent to correct image orientation."""

    def __init__(self, true_angle=0):
        super(OrientationEnv, self).__init__()
        self.true_angle = true_angle
        # Define a discrete action space: 360 actions, one for each degree
        self.action_space = spaces.Discrete(360)
        # Define the observation space: the current angle of the image
        self.observation_space = spaces.Box(low=0, high=359, shape=(1,), dtype=np.float32)
        # Initialize the current angle
        self.current_angle = 0

    def step(self, action):
        """
        Take a step in the environment.
        The action is the predicted angle.
        """
        # The agent's action is the predicted angle
        predicted_angle = action
        # Calculate the reward
        # The reward is higher the closer the predicted angle is to the true angle
        reward = 1.0 / (1.0 + abs(predicted_angle - self.true_angle))
        # The episode is done after one step
        done = True
        # There is no additional info to return
        info = {}
        # The observation is the true angle
        observation = np.array([self.true_angle], dtype=np.float32)
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        # We don't need to do anything here, as the state is always the same
        return np.array([self.true_angle], dtype=np.float32), {}

    def render(self, mode='human'):
        """
        Render the environment.
        """
        pass
