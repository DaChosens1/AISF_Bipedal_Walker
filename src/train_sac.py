from stable_baselines3 import SAC
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

env = gym.make("BipedalWalker-v3")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

eval_callback = EvalCallback(
    env, 
    callback_on_new_best=callback_on_best, 
    verbose=1, 
    best_model_save_path='./logs/best_model/',
    log_path='./logs/results/', 
    eval_freq=10000
)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=1_000_000, 
    learning_starts=1000,  
    batch_size=256,        
    tau=0.005,             
    gamma=0.99,            
    learning_rate=3e-4,    
    train_freq=1,          
    gradient_steps=1,     
    ent_coef="auto",      
    tensorboard_log="./logs/sac_walker_experiments/"
)

model.learn(total_timesteps=1_000_000, log_interval=10, progress_bar=True, callback=eval_callback)

model.save("sac_walker")
env.save("sac_env.pkl")