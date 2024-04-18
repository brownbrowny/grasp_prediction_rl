import numpy as np

from src.envs.utils import is_point_inside_surface, calc_3d_distance

def reward_check_point_inside_point_cloud(delauny, action):
    '''
    Reward function that checks if the action is inside the point cloud
    :param delauny: scipy.spatial.Delaunay object containing the triangulation
    :param action: np.ndarray of shape (3,) containing the action as a point
    '''
    if is_point_inside_surface(action, delauny):
        return 1
    else:
        return 0
    
    
def reward_check_distance_to_point_cloud(action_pose, target_pose, max_reward=1, max_distance=6.0):
    '''
    Reward function that retrieves the distance between the action and the target
    then maps this distance to a proportional reward
    e.g.
    0m    = 100 - (0 / 1.74) * 100 = 100 --> max reward
    1m    = 100 - (1 / 1.74) * 100 = 100 - 57.47 = 42.53
    1.74m = 100 - (1.74 / 1.74) * 100 = 0 --> min reward
    
    :param action_pose: np.ndarray of shape (3,) containing the action's pose
    :param target_pose: np.ndarray of shape (3,) containing the target's pose
    :return: float indicating the 3d distance between the action and the target
    '''
    distance = calc_3d_distance(action_pose, target_pose)    
    reward = max_reward - (distance / max_distance) * max_reward
    # reward must be between 0 and max_reward
    return reward
