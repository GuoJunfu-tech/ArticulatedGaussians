import open3d as o3d
import numpy as np
import torch

from utils.knn_utils import knn
from utils.loss_utils import arap_loss


if __name__ == "__main__":
    n = 50000
    num_knn = 10

    xyz = torch.tensor([[0, 0, 0], [0, 0.1e-3, 0], [1e-3, 0, 0], [1e-3, 1e-3, 0]])
    new_xyz = torch.tensor([[0, 0, 0], [0, 5e-2, 0], [5e-2, 0, 0], [5e-2, 3e-2, 0]])

    neighbor_sq_dist, neighbor_indices = knn(xyz.numpy(), 2)
    print(neighbor_sq_dist)
    weight = np.exp(-2000, neighbor_sq_dist)
    print(neighbor_sq_dist)
    print(weight)
    dist = np.sqrt(neighbor_sq_dist)
    neighbor_weight = torch.tensor(weight).float().to(xyz.device)
    neighbor_dist = torch.tensor(dist).float().to(xyz.device)

    loss = arap_loss(new_xyz, neighbor_indices, neighbor_dist, neighbor_weight)
    print(loss)
