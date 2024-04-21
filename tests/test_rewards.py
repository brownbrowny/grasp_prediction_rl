from src.envs.utils import load_point_cloud, plot_delauny_3d, compute_delauny_triangulation, is_point_inside_surface
from src.envs.rewards import reward_check_point_inside_point_cloud, reward_check_distance_to_point_cloud
import numpy as np
import open3d as o3d

def test_point_within_point_cloud():

    # load the point cloud
    point_cloud_np = load_point_cloud()
    
    # perform delauny triangulation
    delauny = compute_delauny_triangulation(point_cloud_np)
                
    # create 100 random points
    num_points = 100
    random_points = np.random.choice(point_cloud_np.shape[1], size=num_points, replace=True)
    noise = np.random.normal(0, 0.01, size=(3, num_points))
    random_points_with_noise = point_cloud_np[:, random_points] + noise
    
    # check if each random point is inside the surface
    inside_indices = []
    outside_indices = []
    for i in range(num_points):
        if is_point_inside_surface(random_points_with_noise[:, i], delauny):
            inside_indices.append(i)
        else:
            outside_indices.append(i)
    
    # visualize the point cloud and the points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np.T)
    point_cloud.paint_uniform_color([0.0, 0.0, 1.0])  # Set point cloud color to blue
    
    inside_points = random_points_with_noise[:, inside_indices]
    outside_points = random_points_with_noise[:, outside_indices]
    
    inside_o3d = o3d.geometry.PointCloud()
    inside_o3d.points = o3d.utility.Vector3dVector(inside_points.T)
    inside_o3d.paint_uniform_color([1.0, 0.0, 0.0])  # Set inside points color to red
    
    outside_o3d = o3d.geometry.PointCloud()
    outside_o3d.points = o3d.utility.Vector3dVector(outside_points.T)
    outside_o3d.paint_uniform_color([0.0, 1.0, 0.0])  # Set outside points color to green
    
    # create coordinate axes and visualize
    coordinate_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([point_cloud, inside_o3d, outside_o3d, coordinate_axes])
    
def test_reward_check_point_inside_point_cloud():
    # load the point cloud
    point_cloud_np = load_point_cloud()
    
    # perform delauny triangulation
    delauny = compute_delauny_triangulation(point_cloud_np)
    
    # create a point that is inside the point cloud
    inside_point = np.array([0, 0, 0])
    reward = reward_check_point_inside_point_cloud(delauny, inside_point)
    assert reward == 1, f'Expected return: 0, Got reward: {reward}'
    
    # create a point that is outside the point cloud
    outside_point = np.array([1.5, 1.5, 1.5])
    reward = reward_check_point_inside_point_cloud(delauny, outside_point)
    assert reward == 0, f'Expected reward: 1, Got reward: {reward}'
    
    print('test_reward_point_within_point_cloud() --- All tests passed!\n')
    
    
def test_reward_check_distance_to_point_cloud():
    ##### ATTENTION #####
    # The maximum distance between two points in the point cloud is 6,
    # this is ensured by leaving action and target coordinates in the range [0, 1]
    # therefore any distances greater than this can not occur with the test setup
    # and are not tested
    
    # max reward is 100
    # max distance is 6
    max_reward = 1
    max_distance = 6
    
    #######################
    # create two points
    action_pose = np.array([0, 0, 0])
    target_pose = np.array([1, 1, 1])
    
    # calculate the distance between the two points
    distance = np.abs(np.linalg.norm(target_pose - action_pose, ord=1))
    
    # calculate the reward
    reward = reward_check_distance_to_point_cloud(action_pose, target_pose, max_reward, max_distance)
    
    # calculate the expected reward
    expected_reward = max_reward - (distance / max_distance) * max_reward
    
    assert reward == expected_reward, f'Expected reward: {expected_reward}, Got reward: {reward}'
    
    #######################
    # create two points
    action_pose = np.array([1, 1, 1])
    target_pose = np.array([0, 0, 0.05])
    
    # calculate the distance between the two points
    distance = np.abs(np.linalg.norm(target_pose - action_pose, ord=1))
    
    # calculate the reward
    reward = reward_check_distance_to_point_cloud(action_pose, target_pose, max_reward, max_distance)
    
    # calculate the expected reward
    expected_reward = max_reward - (distance / max_distance) * max_reward
    
    assert reward == expected_reward, f'Expected reward: {expected_reward}, Got reward: {reward}'
    
    #######################
    # create two points
    action_pose = np.array([0.02, 0.06, -0.02])
    target_pose = np.array([0.02, 0.06, -0.02])
    
    # calculate the distance between the two points
    distance = np.abs(np.linalg.norm(target_pose - action_pose, ord=1))
    
    # calculate the reward
    reward = reward_check_distance_to_point_cloud(action_pose, target_pose, max_reward, max_distance)
    
    # calculate the expected reward
    expected_reward = max_reward - (distance / max_distance) * max_reward
    
    assert reward == expected_reward, f'Expected reward: {expected_reward}, Got reward: {reward}'
    
    print('test_reward_check_distance_to_point_cloud() --- All tests passed!\n')
    
    
def test_find_highest_reward():
    '''
    The highest reward is the point which is closest to the target but not within point cloud.
    So basically all points on the surface of the point cloud are candidates for the highest reward.
    Let's find out which one is closest.
    '''
    point_cloud_np = load_point_cloud(pc_shape=(3, 4096))
    
    # use all points on the surface of the point cloud as actions and calculate the reward for each point
    # create an empty list to store action-reward pairs
    action_reward_pairs = []

    # loop through each point in the point cloud
    for i in range(point_cloud_np.shape[1]):
        action = point_cloud_np[:, i]
        reward = reward_check_distance_to_point_cloud(action, target_pose=[0, 0, 0], max_reward=1, max_distance=6)
        
        # append the action-reward pair to the list
        action_reward_pairs.append((action, reward))

    # sort the action-reward pairs based on the reward in descending order
    action_reward_pairs.sort(key=lambda x: x[1], reverse=True)

    # visualize the point cloud and the points with highest rewards
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np.T)
    point_cloud.paint_uniform_color([0.0, 0.0, 1.0])  # Set point cloud color to blue
    
    target_point_o3d = o3d.geometry.PointCloud()
    target_point_o3d.points = o3d.utility.Vector3dVector([[0, 0, 0]])
    target_point_o3d.paint_uniform_color([1.0, 0.0, 0.0])  # Set target point color to red
    
    highest_reward_points = [pair[0] for pair in action_reward_pairs[:5]]
    print(f'Rewards: {[pair[1] for pair in action_reward_pairs[:5]]}')
    highest_reward_o3d = o3d.geometry.PointCloud()
    highest_reward_o3d.points = o3d.utility.Vector3dVector(highest_reward_points)
    highest_reward_o3d.paint_uniform_color([1.0, 0.0, 0.0])  # Set highest reward points color to blue
    
    # create coordinate axes and visualize
    coordinate_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([point_cloud, target_point_o3d, highest_reward_o3d])
    

#test_point_within_point_cloud()
test_reward_check_point_inside_point_cloud()
test_reward_check_distance_to_point_cloud()
test_find_highest_reward()