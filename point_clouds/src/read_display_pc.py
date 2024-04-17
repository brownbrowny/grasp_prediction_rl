import open3d as o3d
import numpy as np
import open3d.visualization as vis
from utils import mesh_to_point_cloud_sampled_with_visualization, mesh_to_point_cloud_sampled, downsample_point_cloud_to_target


mesh_path = "obj_files/box_with_texture_aligned.obj"

downsampled_point_cloud, _ = downsample_point_cloud_to_target(mesh_path, 4096)

# # convert to numpy array and save to disk
# pc_np = np.asarray(downsampled_point_cloud.points)

# # # save the resulting numpy array to disk
# np.save("point_clouds/my_function_result.npy", pc_np)

# # load the numpy array from disk
# pc_np = np.load("point_clouds/my_function_result.npy")

# print(pc_np.shape)

# # convert the numpy array back to an open3d point cloud
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(pc_np)

# visualize the point cloud
vis.draw_geometries([downsampled_point_cloud])