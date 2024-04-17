from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import open3d as o3d

import gymnasium as gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict


# ============================================================================
# Combined Feature Extractor Network
# Uses the Point Net Backbone to extract features from the input point cloud and then
# concatenates the features with the pose vector
class GraspInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1027, debug: bool = False):
     
        super().__init__(observation_space, features_dim)  # 4096 (pc) + 3 (position vector)
                
        self.point_cloud_extractor = PointNetBackbone(num_points=4096, num_global_feats=1024, local_feat=False)
        self.flatten_layer = nn.Flatten()
        self.debug = debug
        
        self._features_dim = features_dim 
    
    # forward() automatically unpacks dictionaries which are passed to the model
    def forward(self, observations: TensorDict) -> torch.Tensor:        
        
        assert 'point_cloud' in observations, 'No point cloud in observation space'
        assert 'pose_vector' in observations, 'No position vector in observation space'
        assert len(observations) == 2, 'More than two observations provided'
        
        encoded_tensor_list = []
        
        point_cloud = observations["point_cloud"]
        pose_vector = observations["pose_vector"]
        
        features, critical_indexes = self.point_cloud_extractor(point_cloud)
        
        # Flatten the features and pose vector
        encoded_tensor_list.append(self.flatten_layer(features))
        encoded_tensor_list.append(self.flatten_layer(pose_vector))
        
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        if self.debug:
            print(f'pose_vector cloud shape: {encoded_tensor_list[1].shape}')
            print(f'point cloud shape: {encoded_tensor_list[0].shape}')
            return torch.cat(encoded_tensor_list, dim=1), critical_indexes
        else:
            return torch.cat(encoded_tensor_list, dim=1)
    
    
    # @torch.no_grad()
    # def preprocess(self, observations: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        
    #     point_cloud = observations['point_cloud']
    #     pose_vector = observations['pose_vector']
        
    #     # check if the points in x are centered, center it
    #     centroid = np.mean(point_cloud, axis=1)
    #     if not np.allclose(centroid, np.zeros(3), atol=1e-6):
    #         print('Centering Point Cloud!')
    #         point_cloud -= centroid
        
    #     # check if the point cloud is normalized
    #     norms = np.linalg.norm(point_cloud, axis=1)  # Calculate the Euclidean norm for each point
    #     max_distance = np.max(norms)
    #     if not max_distance <= 1:
    #         print("Normalizing Point Cloud!")
    #         point_cloud /= max_distance
            
            
    #     # check if pose vector is normalized
    #     if not np.all((pose_vector >= -1) & (pose_vector <= 1)):
    #         print("Normalizing Pose Vector!")
    #         # Calculate the maximum absolute value for scaling
    #         max_abs_val = np.max(np.abs(pose_vector))
    #         # Normalize values to the range [-1, 1]
    #         pose_vector = pose_vector / max_abs_val
            
    #     # Convert to PyTorch tensors
    #     point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
    #     pose_vector = torch.tensor(pose_vector, dtype=torch.float32)
        
    #     return point_cloud, pose_vector
    


# ============================================================================
# T-net (Spatial Transformer Network)
# serves as a preprocessing layer by learning a transformation matrix for the input point cloud
# as pcs are not sorted, the system must be robust against spatial translation and rotations
# points which are close to each other in the point cloud must also be processed in that manner
# if they were e.g. slightly rotated or translated
class Tnet(nn.Module):
    ''' T-Net learns a Transformation matrix with a specified dimension '''
    def __init__(self, dim, num_points=4096):
        super(Tnet, self).__init__()

        # dimensions for transform matrix
        self.dim = dim 

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)
        

    def forward(self, x):
        bs = x.shape[0]

        # pass through shared MLP layers (conv1d)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        # max pool over num points
        x = self.max_pool(x).view(bs, -1)
        
        # pass through MLP
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)

        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden

        return x


# ============================================================================
# Point Net Backbone (main Architecture)
class PointNetBackbone(nn.Module):
    '''
    This is the main portion of Point Net before the classification and segmentation heads.
    The main function of this network is to obtain the local and global point features, 
    which can then be passed to each of the heads to perform either classification or
    segmentation. The forward pass through the backbone includes both T-nets and their 
    transformations, the shared MLPs, and the max pool layer to obtain the global features.

    The forward function either returns the global or combined (local and global features)
    along with the critical point index locations and the feature transformation matrix. The
    feature transformation matrix is used for a regularization term that will help it become
    orthogonal. (i.e. a rigid body transformation is an orthogonal transform and we would like
    to maintain orthogonality in high dimensional space). "An orthogonal transformations preserves
    the lengths of vectors and angles between them"
    ''' 
    def __init__(self, num_points=4096, num_global_feats=1024, local_feat=True):
        ''' Initializers:
                num_points - number of points in point cloud
                num_global_feats - number of Global Features for the main 
                                   Max Pooling layer
                local_feat - if True, forward() returns the concatenation 
                             of the local and global features
            '''
        super(PointNetBackbone, self).__init__()

        # if true concat local and global features
        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat

        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.tnet2 = Tnet(dim=64, num_points=num_points)

        # shared MLP 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)
        
        # batch norms for both shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

    
    def forward(self, x):
        # get batch size
        bs = x.shape[0]
        
        # pass through first Tnet to get transform matrix
        A_input = self.tnet1(x)

        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        # pass through first shared MLP
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        
        # get feature transform
        A_feat = self.tnet2(x)

        # perform second transformation across each (64 dim) feature in the batch
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)

        # store local point features for segmentation head
        local_features = x.clone()

        # pass through second MLP
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        # get global feature vector and critical indexes
        global_features, critical_indexes = self.max_pool(x)
        # global_features = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        if self.local_feat:
            features = torch.cat((local_features, 
                                  global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), 
                                  dim=1)

            # return features, critical_indexes, A_feat
            return features, critical_indexes

        else:
            # return global_features, critical_indexes, A_feat
            return global_features, critical_indexes


# ============================================================================
# Classification Head
class PointNetClassHead(nn.Module):
    '''' Classification Head '''
    def __init__(self, num_points=4096, num_global_feats=1024, k=2):
        super(PointNetClassHead, self).__init__()

        # get the backbone (only need global features for classification)
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=False)

        # MLP for classification
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, k)

        # batchnorm for the first 2 linear layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # The paper states that batch norm was only added to the layer 
        # before the classication layer, but another version adds dropout  
        # to the first 2 layers
        self.dropout = nn.Dropout(p=0.3)
        

    def forward(self, x):
        # get global features
        x, crit_idxs, A_feat = self.backbone(x) 

        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)

        # return logits
        return x, crit_idxs, A_feat


# ============================================================================
# Segmentation Head
class PointNetSegHead(nn.Module):
    ''' Segmentation Head '''
    def __init__(self, num_points=4096, num_global_feats=1024, m=2):
        super(PointNetSegHead, self).__init__()

        self.num_points = num_points
        self.m = m

        # get the backbone 
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=True)

        # shared MLP
        num_features = num_global_feats + 64 # local and global features
        self.conv1 = nn.Conv1d(num_features, 512, kernel_size=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, m, kernel_size=1)

        # batch norms for shared MLP
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)


    def forward(self, x):
        
        # get combined features
        x, crit_idxs, A_feat = self.backbone(x) 

        # pass through shared MLP
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1)
        
        return x, crit_idxs, A_feat

    
# ============================================================================
# Test 
def main():
    
    # load numpy array from disk
    point_cloud_np = np.load('point_clouds/box_point_cloud.npy')
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
    
if __name__ == '__main__':
    main()
