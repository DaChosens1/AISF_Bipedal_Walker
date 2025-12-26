from stable_baselines3 import SAC
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor # Add this import

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

# 1. Environment with Normalization
env = gym.make("BipedalWalker-v3")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
# SAC loves normalized observations but keep reward normalization OFF initially
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

# 2. Create the evaluation callback
# This will test the agent every 10,000 steps and stop if it hits the threshold
eval_callback = EvalCallback(
    env, 
    callback_on_new_best=callback_on_best, 
    verbose=1, 
    best_model_save_path='./logs/best_model/',
    log_path='./logs/results/', 
    eval_freq=10000
)

# 3. The SAC Model
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=1_000_000,  # SAC needs a big memory to store past walks
    learning_starts=1000,    # Collects random data first to seed the memory
    batch_size=256,         # SAC usually likes larger batches than PPO
    tau=0.005,              # How fast the "target network" updates (stability)
    gamma=0.99,             # Discount factor
    learning_rate=3e-4,     # Standard starting point for SAC
    train_freq=1,           # Update the model every 1 step
    gradient_steps=1,       # How many gradient steps per update
    ent_coef="auto",        # SECRET SAUCE: SAC will tune its own entropy!
    tensorboard_log="./logs/sac_walker_experiments/"
)

model.learn(total_timesteps=1_000_000, log_interval=10, progress_bar=True, callback=eval_callback)

model.save("sac_walker")
env.save("sac_env.pkl")