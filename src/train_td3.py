import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

env = gym.make("BipedalWalker-v3")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000) # eval_freq could be heigher like 50k

model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    buffer_size=1_000_000,    
    learning_rate=1e-3,       
    batch_size=256,           
    tau=0.005,                
    gamma=0.99,               
    policy_delay=2,           
    target_policy_noise=0.2,  
    target_noise_clip=0.5,    
    policy_kwargs=dict(net_arch=[400, 300]),
    tensorboard_log="./logs/td3_walker/"
)

model.learn(total_timesteps=1_000_000, progress_bar=True, callback=eval_callback)

model.save("td3_walker")
env.save("vec_normalize.pkl")