import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the environment
# We use 'rgb_array' so we can record videos later
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

# 3. Initialize the PPO Agent
# We use the default 'MlpPolicy' (Multi-Layer Perceptron) as requested
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./logs/ppo_walker_experiments/",
    device="cpu"  # Macs handle this well on CPU
)

# 4. Start Training
# We will train for 200,000 steps for the baseline.
print("Starting training... check TensorBoard to see progress.")
model.learn(
    total_timesteps=5e5, 
    progress_bar=True
)

# 5. Save the final model
model.save("ppo_bipedal_walker_baseline")
print("Baseline training complete!")
