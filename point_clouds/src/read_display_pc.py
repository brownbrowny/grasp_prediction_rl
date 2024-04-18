import open3d as o3d
import numpy as np
import open3d.visualization as vis
from utils import mesh_to_point_cloud_sampled_with_visualization, mesh_to_point_cloud_sampled, downsample_point_cloud_to_target
import os

# Get the absolute path by joining the relative path with the current working directory

# mesh_path = "point_clouds/meshes/box_with_texture_aligned.obj"

# absolute_mesh_path = os.path.join(os.getcwd(), mesh_path)

# downsampled_point_cloud, _ = downsample_point_cloud_to_target(absolute_mesh_path, 256)

# # convert to numpy array and save to disk
# pc_np = np.asarray(downsampled_point_cloud.points)

# save_path = "point_clouds/point_clouds/point_cloud_256.obj"
# absolute_save_path = os.path.join(os.getcwd(), save_path)

# # save the resulting numpy array to disk
# np.save(save_path, pc_np)


cwd = os.getcwd()   
absolute_path = os.path.join(cwd, 'point_clouds', 'point_clouds_ready', 'box_point_cloud_256.npy')

pc_np = np.load(absolute_path).astype(np.float32)

print(pc_np.shape)

# convert the numpy array back to an open3d point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pc_np)

# visualize the point cloud
vis.draw_geometries([point_cloud])