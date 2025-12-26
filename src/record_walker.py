import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
import os

def record_walker(model_path, stats_path="../models/vec_normalize.pkl", video_folder="./best_videos"):
    env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3", render_mode="rgb_array")])
    
    env = VecNormalize.load(stats_path, env)
    env.training = False     
    env.norm_reward = False  
    
    env = VecVideoRecorder(
        env, 
        video_folder, 
        record_video_trigger=lambda x: x == 0, 
        video_length=1600,                     
        name_prefix="final_eval"
    )
    
    model = TD3.load(model_path, env=env)
    
    obs = env.reset()
    for _ in range(1600):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break
            
    env.close()
    print(f"Video saved to {video_folder}")

if __name__ == "__main__":
    record_walker("../models/td3_walker.zip")