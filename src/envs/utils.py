import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

def load_point_cloud(pc_shape=(3, 256)):
    '''
    Loads a point cloud from a .npy file in the default point_clouds directory relative to the working directory.
    The point cloud is reshaped to (3, n) and validated to ensure it has the correct shape and sufficient points.
    
    :raises ValueError: If the point cloud is not of shape (3, n) or has less than 128 points.
    :return: np.ndarray of shape (3, n) containing the loaded and validated point cloud.
    '''
    cwd = os.getcwd()   
    
    if pc_shape[1] == 4096:
        file_name = 'box_point_cloud_4096.npy'
    elif pc_shape[1] == 256:
        file_name = 'box_point_cloud_256.npy'
    else:
        raise ValueError("Invalid shape for point cloud. Expected shape: (3, 4096) or (3, 256), Got shape: {}".format(pc_shape))
    
    absolute_path = os.path.join(cwd, 'point_clouds', 'point_clouds_ready', file_name)
    
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

def translate_point_cloud(point_cloud, translation):
    '''
    Translates a point cloud by a given translation vector
    
    :param point_cloud: np.ndarray of shape (3, n) containing the point cloud
    :param translation: np.ndarray of shape (3,) containing the translation vector
    :return: np.ndarray of shape (3, n) containing the translated point cloud
    '''
    return point_cloud + translation[:, np.newaxis]

def compute_delauny_triangulation(point_cloud):
    '''
    Computes the delauny triangulation of a point cloud
    '''
    return Delaunay(point_cloud.T)


def is_point_inside_surface(point, delauny):
    '''
    Determines if a point is inside one of the tetraedrons of the delauny triangulation
    
    :param point: np.ndarray of shape (3,) containing the point to check
    :param delauny: scipy.spatial.Delaunay object containing the triangulation
    :return: bool indicating if the point is inside the surface
    '''
    simplex_index = delauny.find_simplex(point)
    return simplex_index >= -0

def calc_3d_distance(point1, point2):
    '''
    Calculates the absolute 3D distance between two points
    
    :param point1: np.ndarray of shape (3,) containing the first point
    :param point2: np.ndarray of shape (3,) containing the second point
    :return: float indicating the absolute 3D distance between the two points
    '''
    return np.abs(np.linalg.norm(point2 - point1, ord=1))
    

def plot_delauny_3d(point_cloud):
    delauny = Delaunay(point_cloud.T)
    fig = plt.figure()
    ax = fig.aplot_delauny_3d_mayavidd_subplot(111, projection='3d')
    
    # draw points
    ax.scatter(point_cloud[0,:], point_cloud[1,:], point_cloud[2,:], color='blue')

    # draw connections between points
    for simplex in delauny.simplices:
        for i in range(4):
            for j in range(i+1, 4):
                p1 = point_cloud[:,simplex[i]]
                p2 = point_cloud[:,simplex[j]]
                ax.plot3D(*zip(p1, p2), color='red')
                
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
                
def reconstruct_surface(point_cloud_np):
    '''
    Conducts a surface reconstruction of the point cloud using the ConvexHull algorithm
    
    : param point_cloud: np.ndarray of shape (3, n) containing the point cloud
    : return: np.ndarray of shape (3, m) containing the reconstructed surface
    '''
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np.T)
    
    # compute the convex hull as a mesh
    hull, _ = point_cloud_o3d.compute_convex_hull()
    hull_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(hull.vertices), triangles=o3d.utility.Vector3iVector(hull.triangles))
    
    return hull_mesh