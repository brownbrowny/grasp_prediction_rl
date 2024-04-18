# Imports
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import torch

import os
import numpy as np

from src.networks.point_net import GraspInputExtractor
from src.envs.GraspEnv import GraspEnv
from src.envs.GraspEnvNoPC import GraspEnvNoPC

import gc

gc.collect()
torch.cuda.empty_cache()

# test training loop function
def test_training_loop(env, episodes=5):
    print(env.observation_space.sample())

    for episode in range(1, episodes+1):
        state = env.reset()        
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        print(f'Episode: {episode} - Reward: {reward}')
        if truncated or done:
            break
        
def model_memory_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()  # numel gives the total number of elements, element_size gives size of each element in bytes
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    total_size = param_size + buffer_size
    return total_size / (1024 ** 2)  # Convert from bytes to megabytes
        
        
# create the environment
env = GraspEnvNoPC()

#test_training_loop(env, 20)

log_path = os.path.join('Training', 'Logs')

policy_kwargs = dict(
    features_extractor_class=GraspInputExtractor,
    features_extractor_kwargs=dict(num_points=256, features_dim=128),
)

env_kwargs = dict(
    pc_shape=(3, 256)
)

vec_env = make_vec_env(GraspEnv, n_envs=8, seed=0, env_kwargs=env_kwargs)

#check_env(envs, warn=True)

#model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_path)

model = A2C("MultiInputPolicy", vec_env, verbose=1,gamma=0, tensorboard_log=log_path)
# model.save("test_model")

print(model.policy)
#model.learn(2_000_000, progress_bar=True)

# # try:
# #     pass
# #     model.learn(total_timesteps=50_000, progress_bar=True)
# # except Exception as e:
# #     print(e)
# #     print(torch.cuda.memory_summary(device=None, abbreviated=False))