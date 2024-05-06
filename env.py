import gym
from gym import spaces
import numpy as np


# Create a custom environment for OpenAI Gym
class ForestGrowthEnv(gym.Env):
    def __init__(self):
        super(ForestGrowthEnv, self).__init__()
        # % Thinning
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
        # State space
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        # Initialize state
        self.state = 0  # Starting timber volume
        # Set up other variables
        self.step_size = 5  # years per step

    def step(self, action):
        # Action is the thinning percentage, which can reduce the growth factor
        base_growth_factor = 1.02  # Base growth without thinning
        thinning_effect = 1 - action[0] * 0.1  # Assuming 10% reduction in growth per 10% thinning
        growth_factor = base_growth_factor * thinning_effect

        # 10% chance of bad weather reducing volume by 10%
        weather_impact = np.random.choice([0.9, 1.1], p=[0.1, 0.9])
        self.state *= growth_factor * weather_impact
        reward = self.state  # Reward is the timber volume
        done = False  # Simulation can continue indefinitely
        return np.array([self.state]), reward, done, {}

    def reset(self):
        self.state = 100
        return np.array([self.state])

    def render(self, mode='human'):
        print(f'Current forest timber volume: {self.state}')
