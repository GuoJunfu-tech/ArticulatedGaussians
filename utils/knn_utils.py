import open3d as o3d
import numpy as np


def knn(xyz, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])

    return np.array(sq_dists), np.array(indices)


def construct_tree(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    return pcd, pcd_tree
