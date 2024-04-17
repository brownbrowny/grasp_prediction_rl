# Imports
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

import torch

import os
import numpy as np

from src.networks.point_net import GraspInputExtractor
from src.envs.GraspEnv import GraspEnv

import gc

gc.collect()
torch.cuda.empty_cache()

# test training loop function
def test_training_loop(env, episodes=5):
    print(env.observation_space.sample())

    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            if truncated:
                break
        print('Episode:{} Reward:{}'.format(episode, reward))
        
def model_memory_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()  # numel gives the total number of elements, element_size gives size of each element in bytes
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    total_size = param_size + buffer_size
    return total_size / (1024 ** 2)  # Convert from bytes to megabytes

def profile_memory(model, input_size):
    # Create a random input tensor of the specified size
    input_tensor = torch.randn(input_size).cuda()

    # Forward pass
    with torch.cuda.memory_stats() as mem_stats:
        output = model(input_tensor)
        loss = output.mean()  # assuming a simple operation to compute loss
        loss.backward()

    # Print memory statistics
    for stat in ["allocated_bytes.all.peak", "reserved_bytes.all.peak"]:
        print(f"{stat}: {mem_stats[stat] / (1024 ** 2)} MB")  # Convert bytes to MB
        
        
# create the environment
env = GraspEnv()


#test_training_loop(env)

log_path = os.path.join('Training', 'Logs')

policy_kwargs = dict(
    features_extractor_class=GraspInputExtractor,
    features_extractor_kwargs=dict(features_dim=1027),
)

#envs = gym.make_vec("GraspEnv-v0", num_envs=3)

#check_env(envs, warn=True)

model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=3, tensorboard_log=log_path)
# model.save("test_model")

# #print(model.policy)

try:
    model.learn(20000)
except Exception as e:
    print(e)
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    
    model.save("test_model")
    
    #print(f"Model memory size: {model_memory_size(model)} MB")
    profile_memory(model, (1, 3, 4096))  # Adjust input size as needed