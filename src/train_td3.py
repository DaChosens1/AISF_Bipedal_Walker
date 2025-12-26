import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor # Add this import

# 1. Environment with Normalization
env = gym.make("BipedalWalker-v3")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 2. Action Noise (CRITICAL for TD3)
# We add Gaussian noise with mean 0 and std dev 0.1 to encourage exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 3. Early Stopping Callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000)

# 4. TD3 Model
model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,   # Standard TD3 exploration
    verbose=1,
    buffer_size=1_000_000,      # Large memory for walker history
    learning_rate=1e-3,         # Slightly higher than SAC usually
    batch_size=256,             # Standard batch size
    tau=0.005,                  # Target network update rate
    gamma=0.99,                 # Discount factor
    policy_delay=2,             # Updates the actor every 2 critic steps
    target_policy_noise=0.2,    # Smoothing noise for target actions
    target_noise_clip=0.5,      # Limits the smoothing noise
    policy_kwargs=dict(net_arch=[400, 300]), # The "Gold Standard" architecture
    tensorboard_log="./logs/td3_walker/"
)

model.learn(total_timesteps=1_000_000, progress_bar=True, callback=eval_callback)

model.save("td3_walker")
env.save("vec_normalize.pkl")