import random
import math
import torch
import unittest
from typing import List, Tuple

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.deform_utils import ArticulatedOperator

import numpy as np
import open3d as o3d
import copy


def init_pcd(seed=40):
    np.random.seed(seed)  # 为了结果可复现
    points = np.random.rand(500, 3)  # 100个随机点

    points[:, 0] *= 6
    points[:, 1] *= 5
    points[:, 0] *= 2

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def transform_pcd(pcd):
    new_points = np.asarray(pcd.points)[:360, :]
    noise_scale = 0.1  # 高斯噪声的尺度
    noise = np.random.normal(0, noise_scale, new_points.shape)
    new_points += noise

    x = torch.tensor(
        new_points, requires_grad=False, device="cuda", dtype=torch.float32
    )
    axis = torch.tensor(
        (1, 0, 0), requires_grad=False, device=x.device, dtype=torch.float32
    )
    pivot = torch.tensor(
        (0, -5.5, 0), requires_grad=False, device=x.device, dtype=torch.float32
    )
    theta = torch.tensor(
        60 * math.pi / 180, requires_grad=False, device=x.device, dtype=torch.float32
    )

    op = ArticulatedOperator()
    transformed_loc = torch2np(op.get_new_location_revolute(x, axis, pivot, theta))
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_loc)
    return transformed_pcd


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
    min_dist_indices = np.argsort(distances)[:num_pairs]

    # 提取最近的五个点对
    nearest_pairs = point_pairs[min_dist_indices]
    return nearest_pairs


# core test function
def estimate_axis(pcd, target_pcd, num_pairs=5):
    nearest_pairs = find_nearest_point_pairs(pcd, target_pcd, num_pairs)
    x_points, y_points = [], []
    for x, y in nearest_pairs:
        x_points.append(x)
        y_points.append(y)

    middle_point = np.mean(np.vstack((x_points, y_points)), axis=0)
    mean_pcd_loc = np.mean(np.asarray(pcd.points), axis=0)
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
    arti_info = (theta, pivot, axis)

    return nearest_pairs, middle_point, mean_pcd_loc, mean_target_pcd_loc, arti_info


def torch2np(input):
    return input.detach().cpu().numpy()


def np2torch(input):
    return torch.tensor(input, requires_grad=False, device="cuda", dtype=torch.float32)


def visualize_results(
    pcd, transformed_pcd, estimated_pcd, pairs: List[Tuple[List, List]], spheres
):
    pcd_temp = copy.deepcopy(pcd)
    transformed_pcd_temp = copy.deepcopy(transformed_pcd)
    estimated_pcd_temp = copy.deepcopy(estimated_pcd)
    pcd_temp.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (100, 1)))
    transformed_pcd_temp.colors = o3d.utility.Vector3dVector(
        np.tile([0.8, 0.3, 0.1], (len(transformed_pcd.points), 1))
    )
    estimated_pcd_temp.colors = o3d.utility.Vector3dVector(
        np.tile([0.1, 0.8, 0.8], (len(estimated_pcd.points), 1))
    )

    # draw pairs
    x_points, y_points = [], []
    for x, y in pairs:
        x_points.append(x)
        y_points.append(y)

    print(pairs)

    points = o3d.utility.Vector3dVector(np.vstack((x_points, y_points)))
    L = len(pairs)
    lines = []
    lines = np.array([(i, i + L) for i in range(L)])
    lines = o3d.utility.Vector2iVector(lines)
    line_set = o3d.geometry.LineSet(
        points=points,
        lines=lines,
    )

    # 可视化原始点云和变换后的点云
    o3d.visualization.draw_geometries(
        [pcd_temp, transformed_pcd_temp, estimated_pcd_temp, line_set] + spheres
    )


def make_spheres(*args):
    spheres = []
    for center in args:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(center)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.2, 0.2, 0.2])
        spheres.append(sphere)
    return spheres


if __name__ == "__main__":
    pcd = init_pcd()

    target_pcd = transform_pcd(pcd)

    pairs, mdpt, ppt, tppt, arti_info = estimate_axis(pcd, target_pcd, 30)
    theta, pivot, axis = arti_info
    op = ArticulatedOperator()
    x = torch.tensor(
        np.asarray(pcd.points), requires_grad=False, device="cuda", dtype=torch.float32
    )
    axis = torch.tensor(axis, requires_grad=False, device=x.device, dtype=torch.float32)
    pivot = torch.tensor(
        pivot, requires_grad=False, device=x.device, dtype=torch.float32
    )
    theta = torch.tensor(
        theta, requires_grad=False, device=x.device, dtype=torch.float32
    )
    estimated_loc = torch2np(op.get_new_location_revolute(x, axis, pivot, theta))
    estimated_pcd = o3d.geometry.PointCloud()
    estimated_pcd.points = o3d.utility.Vector3dVector(estimated_loc)

    spheres = make_spheres(mdpt, ppt, tppt)
    visualize_results(pcd, target_pcd, estimated_pcd, pairs, spheres)
