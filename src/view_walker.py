import gymnasium as gym
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1. Setup the environment exactly like training
def make_env():
    return gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")

env = DummyVecEnv([make_env])
env = VecNormalize.load("vec_normalize.pkl", env)
env.training = False
env.norm_reward = False 

model = TD3.load("td3_walker", env=env)

obs = env.reset()
episode_reward = 0.0
episode_count = 0

print(f"--- Starting Evaluation ---")

while episode_count < 5:  # Watch 5 episodes
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    
    # In a VecEnv, 'rewards' is an array. We take the first element [0].
    episode_reward += rewards[0]
    
    # Check if the episode ended
    if dones[0]:
        episode_count += 1
        print(f"Episode {episode_count} Finished! Score: {episode_reward:.2f}")
        
        # Reset tracker for next episode
        episode_reward = 0.0 
        # Note: VecEnv resets automatically, so no env.reset() needed here