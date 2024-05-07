import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class ThinningAction:
    spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)


class Forest:
    def __init__(self):
        self.last_action = None
        self.area = 10
        self.reset()

    def reset(self, seed=None):
        random.seed(seed)
        
    def perform_action(self, thinning_action:ThinningAction) -> float:
        self.last_action = thinning_action
        
        