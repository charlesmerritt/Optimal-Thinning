import numpy as np
import gym
from env import OptimalThinningEnv

def policy_iteration(env, gamma=0.99, theta=1e-5):
    def one_step_lookahead(state, V, gamma):
        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            env.state = state
            next_state, reward, done, _ = env.step(action)
            action_values[action] = reward + gamma * V[next_state]
        return action_values

    V = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n, dtype=int)
    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for state in range(env.observation_space.n):
                v = V[state]
                V[state] = np.max(one_step_lookahead(state, V, gamma))
                delta = max(delta, np.abs(v - V[state]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for state in range(env.observation_space.n):
            old_action = policy[state]
            action_values = one_step_lookahead(state, V, gamma)
            new_action = np.argmax(action_values)
            policy[state] = new_action
            if old_action != new_action:
                policy_stable = False

        if policy_stable:
            break

    return policy, V

if __name__ == "__main__":
    env = OptimalThinningEnv(max_trees=10, max_steps=100)
    policy, value = policy_iteration(env)
    print(f"Optimal Policy: {policy}")
    print(f"Value Function: {value}")
