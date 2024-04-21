from src.envs.utils import load_point_cloud, is_point_inside_surface, translate_point_cloud
import numpy as np
import open3d as o3d

def test_translate_point_cloud():
    point_cloud_np = load_point_cloud((3, 256))
    
    ################ Test 1: No translation - center: [0, 0, 0] ################
    # find geometric center of the point cloud
    center = np.mean(point_cloud_np, axis=1)
    print(f'Test1: Center of point cloud: {center}')
    
    # check if the center is close to [0, 0, 0]
    assert np.allclose(center, [0, 0, 0], atol=0.01), f'Expected center: [0, 0, 0], Got center: {center}'
    
    ################ Test 2: Translation Vector - [0.45, 0.48, 0.042] ################
    translation_vector = np.array([0.45, 0.48, 0.042])
    
    # translate the point cloud
    translated_point_cloud = translate_point_cloud(point_cloud_np, translation_vector)
    
    # find geometric center of the translated point cloud
    center = np.mean(translated_point_cloud, axis=1)
    print(f'Test2: Center of translated point cloud: {center}')
    assert np.allclose(center, translation_vector, atol=0.01), f'Expected center: {translation_vector}, Got center: {center}'
    
    ################ Test 3: Translation Vector - [-0.45, -0.48, -0.042] ################
    translation_vector = np.array([-0.45, -0.48, -0.042])
    
    # translate the point cloud
    translated_point_cloud = translate_point_cloud(point_cloud_np, translation_vector)
    
    # find geometric center of the translated point cloud
    center = np.mean(translated_point_cloud, axis=1)
    print(f'Test3: Center of translated point cloud: {center}')
    assert np.allclose(center, translation_vector, atol=0.01), f'Expected center: {translation_vector}, Got center: {center}'    
    
test_translate_point_cloud()