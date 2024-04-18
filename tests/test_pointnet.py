import numpy as np
import torch
import open3d as o3d
import os

from src.networks.point_net import GraspInputExtractor

# get point cloud from default point_clouds directory relative to working directory
cwd = os.getcwd()   
os.path.join(cwd, 'point_clouds', 'point_clouds_ready', 'box_point_cloud.npy')

# load numpy array from disk
point_cloud_np = np.load('point_clouds/point_clouds_ready/box_point_cloud.npy').astype(np.float32)
    

print(f'Point Cloud Shape: {point_cloud_np.shape}')
# (4096x3)

# batch size
batch_size = 32

# print reshape to (3, 4096) and copy it batch_size times
point_cloud = np.transpose(point_cloud_np, (1, 0))
point_cloud = np.expand_dims(point_cloud, axis=0)
point_cloud_batch = np.repeat(point_cloud, batch_size, axis=0)

# pose vector
pose_vector = np.array([0.45, 0.48, 0.042])
# copy it batch_size times to match point cloud as (32, 3)
pose_vector_batch = np.repeat([pose_vector], batch_size, axis=0)
    
# convert both to torch tensors and put into a dictionary
test_data = {
    'point_cloud': torch.tensor(point_cloud_batch, dtype=torch.float32),
    'pose_vector': torch.tensor(pose_vector_batch, dtype=torch.float32)
}

print(f'point cloud shape: {test_data["point_cloud"].shape}')
print(f'pose_vector shape: {test_data["pose_vector"].shape}')


# # Call the model with mock data
model = GraspInputExtractor(test_data, debug=True)
output_features, indices = model(test_data)
print(f'Output Features Shape: {output_features.shape}')
print(f'Indices Shape: {indices.shape}')

# create a visualization of the original point cloud (needs to be converted to a point cloud)
# and the critical points with a different color
# Convert indices to a NumPy array if they are not already
indices = indices[1, :].numpy()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
# Initialize colors for all points. Default color is blue.
colors = np.zeros(point_cloud_np.shape)
colors[:] = [0, 0, 1]  # Blue

# Set emphasized points to red
colors[indices] = [1, 0, 0]  # Red

# Assign colors to the point cloud
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])