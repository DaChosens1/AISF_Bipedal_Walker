import gymnasium as gym
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    return gym.make("BipedalWalker-v3", render_mode="human")

env = DummyVecEnv([make_env])
env = VecNormalize.load("../models/vec_normalize.pkl", env)
env.training = False
env.norm_reward = False 

model = TD3.load("../models/td3_walker", env=env)

obs = env.reset()
episode_reward = 0.0
episode_count = 0

print(f"--- Starting Evaluation ---")

while episode_count < 5: 
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    
    episode_reward += rewards[0]
    
    if dones[0]:
        episode_count += 1
        print(f"Episode {episode_count} Finished! Score: {episode_reward:.2f}")
        
        episode_reward = 0.0 