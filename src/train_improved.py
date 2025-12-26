import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor # Add this import

def make_env():
    env = gym.make("BipedalWalker-v3")
    
    env = Monitor(env)

    return env

env = DummyVecEnv([make_env])

env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)


model = PPO(
    "MlpPolicy",
    env,
    use_sde=True,
    sde_sample_freq=16,
    n_steps=2048,        
    batch_size=64,      
    n_epochs=20,          
    learning_rate=2e-4,  
    clip_range=0.2,
    ent_coef=0.001,      
    tensorboard_log="./logs/ppo_walker_experiments/"
)

model.learn(total_timesteps=1e6, progress_bar=True)

model.save("ppo_walker_improved6")
env.save("vec_normalize6.pkl")
