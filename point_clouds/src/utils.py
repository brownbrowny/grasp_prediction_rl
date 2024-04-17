import numpy as np
import open3d as o3d




def downsample_point_cloud_to_target(mesh_path, max_points):
    """
    Lädt ein Mesh von einem gegebenen Pfad, wandelt es in eine Punktwolke um,
    und tastet es ab, um eine maximale Anzahl von Punkten zu erreichen.

    Parameter:
    - mesh_path: Der Pfad zur Mesh-Datei.
    - max_points: Die maximale Anzahl von Punkten in der heruntergetasteten Punktwolke.

    Gibt die heruntergetastete Punktwolke und die verwendete Voxel-Größe zurück.
    """
    # Mesh laden und in eine Punktwolke umwandeln
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = mesh.vertices
    if mesh.has_vertex_normals():
        point_cloud.normals = mesh.vertex_normals
    if mesh.has_vertex_colors():
        point_cloud.colors = mesh.vertex_colors

    # Bounding Box und ihre Ausdehnung berechnen
    bbox = point_cloud.get_axis_aligned_bounding_box()
    volume = np.prod(bbox.get_max_bound() - bbox.get_min_bound())

    # Zielvolumen pro Punkt berechnen
    target_volume_per_point = volume / max_points

    # Voxel-Größe als Kubikwurzel des Zielvolumens pro Punkt
    voxel_size = np.cbrt(target_volume_per_point)
    print(f"Anzahl der Punkte des ursprünglichen Meshes: {len(point_cloud.points)}")
    print(f"Volumen der Bounding Box: {volume}")
    print(f"Zielvolumen pro Punkt: {target_volume_per_point}")
    print(f"Voxel-Größe für das Heruntertasten: {voxel_size}")

    # Punktwolke heruntertasten
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    

    voxel_size = 0.01
    i = 0
    # Iterative Anpassung der Voxelgröße
    while len(downsampled_point_cloud.points) != max_points:
        if len(downsampled_point_cloud.points) > max_points:
            voxel_size *= 1.01  # Erhöhen Sie die Voxelgröße, wenn es zu viele Punkte gibt
        else:
            voxel_size *= 0.99  # Verringern Sie die Voxelgröße, wenn es zu wenige Punkte gibt
        downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        i += 1
    
    print(f"Anzahl der Iterationen: {i}")
    print(f"Anzahl der Punkte in der heruntergetasteten Punktwolke: {len(downsampled_point_cloud.points)}")
      
    return downsampled_point_cloud, voxel_size


def mesh_to_point_cloud_sampled(mesh_path, amount_points):
    """
    Convert a mesh to a point cloud with an exact number of points.
    
    Parameters:
    - mesh_path: str, the file path to the mesh.
    - amount_points: int, the desired number of points in the sampled point cloud.
    
    Returns:
    - point_cloud: open3d.geometry.PointCloud, the resulting point cloud with exactly amount_points.
    """
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Convert the mesh to a point cloud
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=amount_points)
    
    # If the Poisson disk sampling does not return the exact number of points,
    # we use a voxel downsampling followed by a uniform sampling to adjust.
    if len(point_cloud.points) != amount_points:
        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / (2 * amount_points) ** (1/3)
        voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        
        if len(voxel_down_pcd.points) > amount_points:
            # If still more points, randomly remove points
            indices = np.random.choice(len(voxel_down_pcd.points), amount_points, replace=False)
            sampled_points = np.asarray(voxel_down_pcd.points)[indices]
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
        elif len(voxel_down_pcd.points) < amount_points:
            # If less points, add by duplicating existing points (not ideal but ensures count)
            additional_indices = np.random.choice(len(voxel_down_pcd.points), amount_points - len(voxel_down_pcd.points), replace=True)
            all_indices = np.arange(len(voxel_down_pcd.points)).tolist() + additional_indices.tolist()
            sampled_points = np.asarray(voxel_down_pcd.points)[all_indices]
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
            
    return point_cloud


def mesh_to_point_cloud_sampled_with_visualization(mesh_path, amount_points):
    """
    Convert a mesh to a point cloud with an exact number of points and visualize both
    the original and the downsampled point clouds in different colors.
    
    Parameters:
    - mesh_path: str, the file path to the mesh.
    - amount_points: int, the desired number of points in the sampled point cloud.
    """
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Convert the mesh to a point cloud for visualization
    original_point_cloud = o3d.geometry.PointCloud()
    original_point_cloud.points = mesh.vertices

    # Downsample or upsample the point cloud to get the exact amount of points
    point_cloud = mesh_to_point_cloud_sampled(mesh_path, amount_points)
    
    # Set the colors of the original and downsampled point clouds for visualization
    original_point_cloud.paint_uniform_color([0, 0, 1])  # Blue
    point_cloud.paint_uniform_color([1, 0, 0])  # Red
    
    # Visualize the original and downsampled point clouds
    o3d.visualization.draw_geometries([original_point_cloud, point_cloud], 
                                      window_name="Original (Blue) vs. Downsampled (Red) Point Cloud",
                                      width=800,
                                      height=600)