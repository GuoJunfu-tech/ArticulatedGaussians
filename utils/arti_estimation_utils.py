import math
import numpy as np
import open3d as o3d
import torch

from typing import Tuple


def find_nearest_point_pairs(pcd, target_pcd, num_pairs):
    target_pcd_tree = o3d.geometry.KDTreeFlann(target_pcd)

    point_pairs = []
    distances = []
    for point in np.asarray(pcd.points):
        _, idx, _ = target_pcd_tree.search_knn_vector_3d(point, 1)
        nearest_point = np.asarray(target_pcd.points)[idx[0]]
        point_pairs.append([point, nearest_point])
        distances.append(np.linalg.norm(point - nearest_point))

    # 将点对和距离转换为NumPy数组
    point_pairs = np.array(point_pairs)
    distances = np.array(distances)

    # 找到最小的五个距离及其索引
    k = 5
    min_dist_indices = np.argsort(distances)[:k]

    # 提取最近的五个点对
    nearest_pairs = point_pairs[min_dist_indices]
    return nearest_pairs


def estimate_arti_info(
    origin_points: torch.tensor,
    target_points: torch.tensor,
    num_pairs: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """for given two pcd, we first find `num_pairs` nearest points pairs, then we calculate:
    1. center of pcd1; 2. center of pcd2; 3. center of the pairs
    for these three terms, we set center 3 as the pivot, the angle at center 3 is the rotation angle
    the axis vertical to the triangle center 123 is the rotation axis

    Args:
        origin_points (torch.tensor)
        target_points (torch.tensor)
        num_pairs (int, optional):Defaults to 5.

    Returns:
        _type_: arti info in torch.tensor types
    """

    origin_points_np = origin_points.detach().cpu().numpy()
    target_points_np = target_points.detach().cpu().numpy()

    origin_pcd = o3d.geometry.PointCloud()
    origin_pcd.points = o3d.utility.Vector3dVector(origin_points_np)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points_np)

    nearest_pairs = find_nearest_point_pairs(origin_pcd, target_pcd, num_pairs)
    x_points, y_points = [], []
    for x, y in nearest_pairs:
        x_points.append(x)
        y_points.append(y)

    middle_point = np.mean(np.vstack((x_points, y_points)), axis=0)
    mean_pcd_loc = np.mean(np.asarray(origin_pcd.points), axis=0)
    mean_target_pcd_loc = np.mean(np.asarray(target_pcd.points), axis=0)

    v1 = mean_pcd_loc - middle_point
    v2 = mean_target_pcd_loc - middle_point
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    theta = np.arccos(dot_product / (norm_v1 * norm_v2))
    pivot = middle_point
    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)

    return pivot, axis, theta


def torch2np(input):
    return input.detach().cpu().numpy()


def np2torch(input):
    return torch.tensor(input, requires_grad=False, device="cuda", dtype=torch.float32)
