import gym
from gym import spaces
import numpy as np


# Create a custom environment for OpenAI Gym
class ForestGrowthEnv(gym.Env):
    def __init__(self):
        super(ForestGrowthEnv, self).__init__()
        # Define action space - Here, we can assume no actions for simplification
        self.action_space = spaces.Discrete(1)
        # Define state space
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        # Initialize state
        self.state = 100  # Starting timber volume
        # Set up other variables
        self.step_size = 5  # years per step

    def step(self, action):
        # Simulate time step
        growth_factor = 1.02  # assuming a 2% growth per period under normal conditions
        weather_impact = np.random.choice([0.9, 1.1], p=[0.1, 0.9])  # 10% chance of bad weather reducing volume by
        self.state *= growth_factor * weather_impact
        reward = self.state  # Reward is the timber volume
        done = False  # Simulation can continue indefinitely
        return np.array([self.state]), reward, done, {}


def reset(self):
    self.state = 100
    return np.array([self.state])


def render(self, mode='human'):
    print(f'Current forest timber volume: {self.state}')


# Example of creating and testing the environment
env = ForestGrowthEnv()
env.reset()
for _ in range(10):
    obs, reward, done, info = env.step(0)
    env.render()
    print(f'Reward: {reward}')