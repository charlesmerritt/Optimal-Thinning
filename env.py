import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ForestGrowthEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    # Forest growth is nonlinear, and often shows diminishing returns as stands become mature or overcrowded.
    # Growth rate changes based on innumerable factors, only some of which can be modelled here.
    # These factors can include things like weather, species, soil, competition, and more.
    # We will focus on competition and weather as a starting point for this project.
    def __init__(self, render_mode=None):
        super(ForestGrowthEnv, self).__init__()
        # Actions: percentage of area to thin (0% to 100%)
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)

        # Observation space: [Timber Volume (V), Density (D), Forest Stand Area (A)]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([np.inf, np.inf, np.inf]),
                                            dtype=np.float32)

        # Constants for growth calculations
        self.r = 0.03  # intrinsic growth rate
        self.K0 = 1000  # baseline carrying capacity
        self.alpha = 0.01  # sensitivity to density
        self.step_size = 5  # Years between thinning decisions

        # Initial state
        self.np_random, _ = gym.utils.seeding.np_random(None)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        V, D, A = self.state

        # Calculate growth over the step size period
        for _ in range(self.step_size):
            K = self.K0 * np.exp(-self.alpha * D) * A
            growth = self.r * V * (1 - V / K)
            V += growth

        # Thinning action
        thinning = action[0]
        removed_volume = V * thinning
        V -= removed_volume
        D *= (1 - thinning)  # Density reduces proportionally

        self.state = np.array([V, D, A], dtype=np.float32)
        reward = removed_volume
        terminated = V <= 0 or D <= 0
        if not isinstance(terminated, bool):
            terminated = bool(terminated)

        # SB3 Requirements
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None):
        # Seed the environment (optional)
        if seed is not None:
            self.seed(seed)

        # Reset the state of the environment to an initial state
        V = self.np_random.uniform(50, 150)  # initial volume randomized
        D = self.np_random.uniform(25, 75)   # initial density randomized
        A = 10  # forest area in hectares constant for simplicity
        self.state = np.array([V, D, A], dtype=np.float32)
        # Auxilliary info, SB3 requirement
        info = {}
        return self.state, info

    def render(self, mode='console'):
        if self.render_mode == 'console':
            # Simple text-based rendering
            print(f"Volume: {self.state[0]}, Density: {self.state[1]}, Area: {self.state[2]}")
        elif self.render_mode == 'human':
            # Potentially more complex graphical rendering
            pass
