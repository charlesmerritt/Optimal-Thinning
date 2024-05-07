from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import ForestGrowthEnv
import gymnasium as gym

if __name__ == '__main__':

    # env = gym.make("CartPole-v1", render_mode="human")
    env = ForestGrowthEnv(render_mode='console')
    env.reset(seed=42)
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

    env.close()
