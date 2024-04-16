import open3d as o3d
import numpy as np

from utils.knn_utils import knn


def test_knn(n, num_knn):
    pts = np.random.rand(n, 3)
    neighbor_sq_dist, neighbor_indices = knn(pts, num_knn)
    print(neighbor_sq_dist.shape)
    print(neighbor_indices.shape)


if __name__ == "__main__":
    n = 50000
    num_knn = 10
    test_knn(n, num_knn)
