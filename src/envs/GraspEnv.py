# Imports
import gymnasium as gym
import numpy as np

import os

global point_cloud

# helper functions

# function for loading the point cloud as numpy array and adding a batch dimension
# by copying the point cloud batch_size times
def load_point_cloud():
    # get point cloud from default point_clouds directory relative to working directory
    cwd = os.getcwd()   
    absolute_path = os.path.join(cwd, 'point_clouds', 'point_clouds_ready', 'box_point_cloud_256.npy')
    
    point_cloud_np = np.load(absolute_path).astype(np.float32)
    point_cloud_np = np.transpose(point_cloud_np, (1, 0)) # reshape to (3, 4096)
    
    # raise error if point cloud is not of shape (3, n)
    if point_cloud_np.shape[0] != 3:
        raise ValueError("Invalid shape for point cloud. Expected shape: (3, n), Got shape: {}".format(point_cloud_np.shape))
    
    # raise error if point cloud has less than 128 points
    if point_cloud_np.shape[1] < 128:
        raise ValueError("Invalid number of points in point cloud. Expected at least 128 points, Got: {}".format(point_cloud_np.shape[1]))
    
    # point_cloud = np.expand_dims(point_cloud, axis=0)
    # point_cloud_batch = np.repeat(point_cloud, batch_size, axis=0)
    return point_cloud_np

def get_point_cloud():
    #print(f'Getting point cloud with shape {point_cloud.shape}')
    return point_cloud


class GraspEnv(gym.Env):
    def __init__(self, pc_shape=(3, 4096)):
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = gym.spaces.Box(0, 1, shape=(3,), dtype=np.float32)
        super().__init__()
        
        self.vector_size = 3
        
        if pc_shape[0] == 3:
            self.pc_shape = pc_shape
        elif pc_shape[1] == 3:
            raise ValueError("Invalid shape for point cloud. Expected shape: (3, n), Got shape: {}".format(pc_shape))
        
        self.point_cloud = load_point_cloud()
        
        self.observation_space = gym.spaces.Dict(
            spaces={
                "pose_vector": gym.spaces.Box(-1, 1, (self.vector_size,), dtype=np.float32),
                "point_cloud": gym.spaces.Box(-1, 1, self.pc_shape, dtype=np.float32),
            }
        )
        
        # 1 action including (x, y, z) floats
        self.action_space = gym.spaces.Box(0, 1, shape=(3,), dtype=np.float32)

        # rewards
        self._distance_to_target = 10
        self._max_reward = 100
        self._max_distance = 1.74   # can occur when target is [0, 0, 0] and action [1, 1, 1]
        
        self.max_steps = 100
        self.current_step = 0


    # translates the environment’s state into an observation
    def _get_obs(self):
        return {
            "pose_vector": [0.45, 0.48, 0.042],
            "point_cloud": self.point_cloud
        }

    # provide the manhattan distance between the agent and the target
    def _get_info(self):
        return {
            "distance": self._reward_distance
        }

    # contains most of the logic of the environment. It accepts an action, 
    # computes the state of the environment after applying that action and 
    # returns the 5-tuple (observation, reward, terminated, truncated, info). 
    # See gymnasium.Env.step(). Once the new state of the environment has been 
    # computed, we can check whether it is a terminal state and we set done accordingly.
    def step(self, action):
        terminated, truncated = False, False
        self.current_step += 1

        # episode is done when distance between agent and target is less than 0.1
        observation = self._get_obs()
        reward, terminated = self.reward(action, observation['pose_vector'])
        
        info = self._get_info()

        # The # episode is done when the maximum number of steps is reached
        if self.current_step >= self.max_steps:
            truncated = True
            self.current_step = 0

        # cast reward to float
        reward = float(reward)
        
        # Return step information
        return observation, reward, terminated, truncated, info
    

    # will be called to initiate a new episode. You may assume that the step method will not 
    # be called before reset has been called. Moreover, reset should be called whenever 
    # a done signal has been issued.
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        if seed is not None:
            super().reset(seed=seed)

        # reset step
        self.current_step = 0
        self._reward_distance = 200
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def reward(self, action, target):
        # Calculate distance between target and action
        self._distance_to_target = np.abs(np.linalg.norm(target - action, ord=1))
        
        # remaps the distance e.g.:
        # 0m    = 100 - (0 / 1.74) * 100 = 100 --> max reward
        # 1m    = 100 - (1 / 1.74) * 100 = 100 - 57.47 = 42.53
        # 1.74m = 100 - (1.74 / 1.74) * 100 = 0 --> min reward
        reward = self._max_reward - (self._distance_to_target / self._max_distance) * self._max_reward
        
        # Termination condition
        terminated = self._distance_to_target < 0.001
        return reward, terminated