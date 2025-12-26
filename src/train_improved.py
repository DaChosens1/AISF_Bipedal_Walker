import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor # Add this import

# from gymnasium.wrappers import ClipAction

def make_env():
    env = gym.make("BipedalWalker-v3")
    
    env = Monitor(env)  # This is the "secret sauce" for TensorBoard metrics
    
    # env = ClipAction(env) # Ensures all actions are strictly between -1 and 1

    return env

# 1. Wrap the environment for Normalization
env = DummyVecEnv([make_env])
# norm_obs=True scales the inputs; norm_reward=True scales the rewards
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

# 3. Initialize PPO with tuned hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    use_sde=True,
    sde_sample_freq=16,
    n_steps=2048,         # Double the data per update (default is 2048)
    batch_size=64,       # More stable gradients
    n_epochs=20,          # More time to think about the data
    learning_rate=2e-4,   # Lower learning rate to prevent "wild" changes
    clip_range=0.2,
    ent_coef=0.001,        # Slight entropy bonus to keep exploring recovery moves
    tensorboard_log="./logs/ppo_walker_experiments/"
)

# 4. Train for a longer duration (500k to 1M steps is usually needed for walking)
model.learn(total_timesteps=1e6, progress_bar=True)

# 5. CRITICAL: Save the normalization stats along with the model
model.save("ppo_walker_improved6")
env.save("vec_normalize6.pkl")
