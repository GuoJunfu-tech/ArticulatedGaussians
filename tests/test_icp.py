# import random
# import math
# import torch
# import unittest

# import sys
# from os import path

# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# from utils.deform_utils import ArticulatedOperator
import numpy as np
import open3d as o3d
import copy


def extract_rotation_info(R):
    trace = np.trace(R)
    theta = np.arccos((trace - 1) / 2)
    if theta != 0:
        v = (
            1
            / (2 * np.sin(theta))
            * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        )
    else:
        v = np.array([1, 0, 0])
    v = v / np.linalg.norm(v)
    return v, theta


def init_pcd(seed=40):
    np.random.seed(seed)  # 为了结果可复现
    points = np.random.rand(100, 3)  # 100个随机点

    points[:, 0] *= 6
    points[:, 1] *= 5
    points[:, 0] *= 2

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def transform_pcd(pcd, num=60):
    # 创建Open3D点云对象

    # 生成一个随机齐次变换矩阵
    R = np.random.rand(3, 3)
    R, _ = np.linalg.qr(R)  # 确保R是正交的
    t = np.random.rand(3) * 10  # 生成一个3x1的平移向量
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    print(transformation_matrix)

    transformed_points = np.dot(pcd.points, R.T) + t
    transformed_points = transformed_points[:num, :]

    # noise_scale = 0.05  # 高斯噪声的尺度
    # noise = np.random.normal(0, noise_scale, transformed_points.shape)
    # transformed_points += noise

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    return transformed_pcd


def visualize_two_pcds(pcd, transformed_pcd, v, pivot, theta):
    pcd_temp = copy.deepcopy(pcd)
    transformed_pcd_temp = copy.deepcopy(transformed_pcd)
    pcd_temp.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (100, 1)))
    transformed_pcd_temp.colors = o3d.utility.Vector3dVector(
        np.tile([0.8, 0.3, 0.1], (len(transformed_pcd.points), 1))
    )

    line_points = np.array([pivot + t * v for t in np.linspace(0, 5, 1)])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)

    # 定义直线的两个端点之间的连接
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])

    radius = 0.2  # 球体半径
    resolution = 20  # 细分程度，值越大，球体越平滑
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)

    # 可视化原始点云和变换后的点云
    o3d.visualization.draw_geometries(
        [pcd_temp, transformed_pcd_temp, line_set, sphere_mesh]
    )


def try_icp(pcd, transformed_pcd):
    # 设置最大对应点对距离

    max_correspondence_distance = 5

    init_transform = np.identity(4)
    # print(init_transform)

    # 执行ICP算法
    reg_p2p = o3d.pipelines.registration.registration_icp(  # FIXME
        pcd,
        transformed_pcd,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # 获取齐次变换矩阵
    return reg_p2p.transformation


def get_rotation_info(transformation_matrix):
    R = transformation_matrix[:3, :3]
    t = transformation_matrix[:3, 3]
    trace = np.trace(R)
    # 计算旋转角度
    theta = np.arccos((trace - 1) / 2)
    # 计算旋转轴
    if theta != 0:
        v = (
            1
            / (2 * np.sin(theta))
            * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        )
    else:
        # 如果角度为0，可以选择任意单位向量作为旋转轴
        v = np.array([1, 0, 0])  # 例如，沿x轴
    # 归一化旋转轴
    v = v / np.linalg.norm(v)

    pivot = np.linalg.inv(np.eye(3) - R) @ t
    return v, pivot, theta


if __name__ == "__main__":
    pcd = init_pcd()
    transformed_pcd = transform_pcd(pcd)
    T_matrix = try_icp(pcd, transformed_pcd)
    v, p, theta = get_rotation_info(T_matrix)
    print(v, p, theta)

    visualize_two_pcds(pcd, transformed_pcd, v, p, theta)
