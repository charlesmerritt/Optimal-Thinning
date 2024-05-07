import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register

register(
    id='ForestGrowthEnv',
    entry_point='env:ForestGrowthEnv',
)


class ForestGrowthEnv(gym.Env):
    metadata = {'render_modes': ['console']}

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

        self.last_action = None
        self.current_step = 0
        self.total_reward = 0

        # Constants for growth calculations
        self.r = 0.03  # intrinsic growth rate
        self.K0 = 1200  # baseline carrying capacity
        self.alpha = 0.004  # sensitivity to density
        self.step_size = 5  # Years between thinning decisions

        # Initial state
        self.state = None
        self.np_random, _ = gym.utils.seeding.np_random(None)

        # Render
        self.render_mode = render_mode

    def step(self, action):
        V, D, A = self.state
        print(f"Initial Volume V: {V}, Density D: {D}, Area A: {A}")

        # Calculate growth over the step size period
        for _ in range(self.step_size):
            K = self.K0 * np.exp(-self.alpha * D) * A
            growth = self.r * V * (1 - V / K)
            V += growth
            print(f"Updated Volume V after growth: {V}")

        # Thinning action
        thinning = action[0]
        removed_volume = V * thinning
        V -= removed_volume
        D *= (1 - thinning)  # Density reduces proportionally

        print(f"Action (thinning): {thinning}, Removed Volume: {removed_volume}")
        print(f"Volume V after thinning: {V}")

        self.state = np.array([V, D, A], dtype=np.float32)
        reward = removed_volume
        self.total_reward += reward

        terminated = V <= 0 or D <= 0
        if not isinstance(terminated, bool):
            terminated = bool(terminated)

        self.last_action = action
        self.current_step += 1

        # SB3 Requirements
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        print("RESET")
        # Seed the environment (optional)


        # Reset the state of the environment to an initial state
        V = self.np_random.uniform(50, 150)  # initial volume randomized
        D = self.np_random.uniform(25, 75)  # initial density randomized
        A = 10  # forest area in hectares constant for simplicity
        self.state = np.array([V, D, A], dtype=np.float32)

        # Auxiliary info, SB3 requirement
        info = {}
        self.current_step = 0
        self.total_reward = 0
        return self.state, info

    def render(self, mode='console'):
        if self.render_mode == 'console':
            # Simple text-based rendering
            print(f"Step: {self.current_step}, Action taken: {self.last_action}, Current State: Volume: "
                  f"{self.state[0]:.3f}, Density: {self.state[1]:.3f}")
            print(f"Total Reward: {self.total_reward:.3f}")
        elif self.render_mode == 'human':
            # Potentially more complex graphical rendering
            pass
