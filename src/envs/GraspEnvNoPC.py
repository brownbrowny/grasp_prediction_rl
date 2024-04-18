# Imports
import gymnasium as gym
import numpy as np

class GraspEnvNoPC(gym.Env):
    def __init__(self):     
        super().__init__()
        
        self.vector_size = 3        
        
        self.observation_space = gym.spaces.Box(-1, 1, (self.vector_size,), dtype=np.float32)
        
        # 1 action including (x, y, z) floats
        self.action_space = gym.spaces.Box(0, 1, shape=(3,), dtype=np.float32)

        self._distance_to_target = 10
        self._max_reward = 100
        self._max_distance = 1.74   # can occur when target is [0, 0, 0] and action [1, 1, 1]
        
        self.max_steps = 100
        self.current_step = 0


    # translates the environment’s state into an observation
    def _get_obs(self):
        return [0.45, 0.48, 0.042]

    # provide the manhattan distance between the agent and the target
    def _get_info(self):
        return {
            "distance": self._distance_to_target
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
        reward, terminated = self.reward(action, observation)
        
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
        self._distance_to_target = 200
        
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