# Imports
import gymnasium as gym
import numpy as np

from src.envs.utils import load_point_cloud, compute_delauny_triangulation
from src.envs.rewards import reward_check_point_inside_point_cloud, reward_check_distance_to_point_cloud


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
        
        self.delauny = compute_delauny_triangulation(self.point_cloud)
        
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
        self._max_distance_reward = 1
        self._max_distance = 6   # can occur when target is [-1, -1, -1] and action [1, 1, 1]
        
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

        ######################################################################
        # REWARD LOGIC
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
        
        # compute new delauny triangulation as point cloud might have changed
        self.delauny = compute_delauny_triangulation(self.point_cloud)

        return observation, info
    
    def reward(self, action, target):
        total_reward = 0
        
        total_reward += reward_check_distance_to_point_cloud(action, target, self._max_distance_reward, self._max_distance)
        
        # if the action is inside the point cloud, it is not a valid action
        if reward_check_point_inside_point_cloud(self.delauny, action):
            total_reward = 0
        
        if self._distance_to_target < 0.01:
            terminated = True
        
        return total_reward, terminated