import gymnasium as gym
from stable_baselines3 import PPO, TD3
from gymnasium.wrappers import RecordVideo

def record_walker(model_path, video_folder="./best_videos"):
    # Create env with rgb_array mode for recording
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    
    # Wrap the env to record every episode in this run
    env = RecordVideo(env, video_folder=video_folder, name_prefix="final_eval")
    
    model = TD3.load(model_path)
    
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    env.close()
    print(f"Video saved to {video_folder}")

# Call the function
record_walker("td3_walker")