import dill as pickle
import torch
from scene import Scene, GaussianModel, DeformModel, Revolute
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt

# from mayavi import mlab
import numpy as np
import open3d as o3d
from PIL import Image
import sys
import math
import copy
from argparse import ArgumentParser, Namespace
from gaussian_renderer import render, network_gui
from utils.knn_utils import knn
from utils.loss_utils import l1_loss, ssim, chamfer_distance_loss
from utils.general_utils import farthest_point_sampling
from utils.classification_utils import kmeans, gmm
from utils.classification_utils import build_mask


def draw_graph(xyz, factor):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 生成一些随机数据
    # data = factor.detach().cpu().numpy()
    print(max(data))

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(factor, bins=30)
    plt.title("Histogram")

    # # 绘制箱形图
    # plt.subplot(1, 2, 2)
    # sns.boxplot(data=data)
    # plt.title("Box Plot")

    plt.tight_layout()
    plt.show()


def visualize(xyz, factor=None, grad=None):
    # ------------- GMM classification ---------------
    pcd = o3d.geometry.PointCloud()
    # xyz = xyz.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if factor is None:
        factor = np.ones((xyz.shape[0], 1))

    # mask = factor.squeeze()
    factor = factor.reshape(-1, 1)
    mask, centers = build_mask(factor, "gmm")
    # mask = abs(factor) < 5e-2
    # print(centers)
    # mask = abs(factor)

    colors = np.zeros((xyz.shape[0], 3))
    for cluster_id, cluster in enumerate(mask):
        # cluster = int(cluster)
        colors[cluster_id, :] = [1, 0, 0] * cluster + [0, 0, 1] * (1 - cluster)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # colors = (w < 5e-2) * [1, 0, 0] + (w >= 5e-2) * [0, 0, 1]
    vis = [pcd]

    # ----------------- Grads Visualization ----------------

    if grad is not None:
        pcd_grad = o3d.geometry.PointCloud()
        # mask = grad
        print(f"max grad: {max(grad)}")
        # w = (grad - grad.min()) / (grad.max() - grad.min())
        # w = w.squeeze()

        w = grad.squeeze()

        colors = np.zeros((xyz.shape[0], 3))
        for cluster_id, cluster in enumerate(w):
            # colors[cluster_id, :] = np.array([1.0, 0.0, 0.0]) * cluster + np.array(
            #     [0.0, 0.0, 1.0]
            # ) * (1 - cluster)
            colors[cluster_id, :] = (
                np.array([1.0, 0.0, 0.0])
                if cluster > 1e-4
                else np.array([0.0, 0.0, 1.0])
            )

        pcd_grad.colors = o3d.utility.Vector3dVector(colors)
        xyz_hard = xyz.copy()
        xyz_hard[:, 0] += 1.0
        pcd_grad.points = o3d.utility.Vector3dVector(xyz_hard)

        vis.append(pcd_grad)

    o3d.visualization.draw_geometries(vis)


def draw_one_color(xyz, filter=None, dx=0.0):
    pcd = o3d.geometry.PointCloud()
    new_xyz = xyz.copy()

    if dx:
        new_xyz[:, 0] += dx
    pcd.points = o3d.utility.Vector3dVector(new_xyz)

    # colors = np.zeros((xyz.shape[0], 3))
    if not filter:
        filter = np.zeros((xyz.shape[0], 1))
    N = filter.shape[0]
    red = np.tile([1, 0, 0], (N, 1))
    blue = np.tile([0, 0, 1], (N, 1))
    filter = filter.reshape(N, 1)
    colors = filter * red + (1 - filter) * blue
    # for id, is_vis in enumerate(filter):
    #     if is_vis:
    #         colors[id, :] = [1, 0, 0]
    #     else:
    #         colors[id, :] = [0, 0, 1]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def draw_two_grads(xyz, grad_1, grad_2):
    vis = []
    vis.append(get_grad_pcd(xyz, grad_1, dx=0))
    vis.append(get_grad_pcd(xyz, grad_2, dx=1))
    o3d.visualization.draw_geometries(vis)


def get_grad_pcd(xyz, grad, dx=0.0):
    pcd_grad = o3d.geometry.PointCloud()
    # mask = grad
    print(f"max grad: {max(grad)}")
    # w = (grad - grad.min()) / (grad.max() - grad.min())
    # w = w.squeeze()

    w = grad.squeeze()

    colors = np.zeros((xyz.shape[0], 3))
    for cluster_id, cluster in enumerate(w):
        # colors[cluster_id, :] = np.array([1.0, 0.0, 0.0]) * cluster + np.array(
        #     [0.0, 0.0, 1.0]
        # ) * (1 - cluster)
        colors[cluster_id, :] = (
            np.array([1.0, 0.0, 0.0]) if cluster > 1 else np.array([0.0, 0.0, 1.0])
        )

    pcd_grad.colors = o3d.utility.Vector3dVector(colors)
    xyz_hard = xyz.copy()
    xyz_hard[:, 0] += dx
    pcd_grad.points = o3d.utility.Vector3dVector(xyz_hard)
    return pcd_grad


def test_maya():
    num_ellipses = 10000
    centers = torch.rand(num_ellipses, 3)
    axis = torch.rand(num_ellipses, 3) * 2

    x, y, z = centers.T
    sx, sy, sz = axis.T
    scale_factors = np.ones(1000)

    # 创建点云
    pts = mlab.points3d(
        x,
        y,
        z,
        scale_factor=0.01,
        scale_mode="none",
        resolution=18,
        color=(0.5, 0.5, 1.0),
        opacity=0.6,
        mode="sphere",
    )
    pts.actor.actor.scale = [1, 2, 3]

    # 批量设置椭球缩放
    pts.mlab_source.dataset.point_data.vectors = np.column_stack((sx, sy, sz))
    pts.mlab_source.dataset.point_data.vectors.name = "scale_vectors"
    pts.glyph.scale_mode = "scale_by_vector"
    # for i in sample_indices:
    #     center = centers[i]
    #     major_axis = major_axes[i]
    #     minor_axis = minor_axes[i]

    #     # 绘制椭球
    #     x, y, z = center
    #     sx, sy, sz = minor_axis
    #     mlab.points3d(
    #         x,
    #         y,
    #         z,
    #         scale_mode="none",
    #         scale_factor=0.1,
    #         color=(0.5, 0.5, 1.0),
    #         resolution=5,
    #         opacity=0.1,
    #         mode="sphere",
    #     ).actor.actor.scale = [sx, sy, sz]

    # 设置视角
    mlab.view(azimuth=123, elevation=23, distance=7, focalpoint=(0, 0, 0))

    # 显示图像
    mlab.show()


def divide_mask(xyz, mask):
    return xyz[mask == 0], xyz[mask == 1]


def draw_axis(pivot, axis):
    axis = axis / np.linalg.norm(axis)
    end_point = pivot + axis
    points = np.vstack([pivot, end_point])
    lines = [[0, 1]]
    colors = np.array([[1, 0, 0]], dtype=np.float64)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def chamfer_distance_open3d(pcd1, pcd2):
    # 创建 KDTree
    pcd_tree1 = o3d.geometry.KDTreeFlann(pcd1)
    pcd_tree2 = o3d.geometry.KDTreeFlann(pcd2)

    # 计算从pcd1到pcd2的单向距离
    def compute_one_side(pcd1, pcd_tree2):
        distances = []
        for i in range(len(pcd1.points)):
            _, idx, dist = pcd_tree2.search_knn_vector_3d(pcd1.points[i], 1)
            distances.append(dist[0])
        return distances

    dist1 = compute_one_side(pcd1, pcd_tree2)
    dist2 = compute_one_side(pcd2, pcd_tree1)

    # 计算Chamfer距离
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return chamfer_dist


def test_cd_loss():
    a = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.float64)
    b = torch.tensor([[2, 0, 0], [2, 1, 0]], dtype=torch.float64)
    print(chamfer_distance_loss(a, b))


if __name__ == "__main__":
    with open("./load_data/washer.pkl", "rb") as f:
        # with open("./final_params.pkl", "rb") as f:
        data = pickle.load(f)

    gaussians = data["gaussians"]
    xyz = gaussians.get_xyz.detach().cpu().numpy()

    dx = data["dx"]
    dr = data["dr"]

    ndx = torch.norm(dx, dim=-1).detach().cpu().numpy()
    ndr = torch.norm(dr, dim=-1).detach().cpu().numpy()

    ndx = (ndx - min(ndx)) / (max(ndx) - min(ndx))
    ndr = (ndr - min(ndr)) / (max(ndr) - min(ndr))
    print(max(ndx))
    # factors = (ndr > 1e-2).to(torch.float32).to("cuda")
    # mask = (ndr > 1e-2).to(gaussians.get_xyz.device, gaussians.get_xyz.dtype)
    # print(mask)
    draw_graph(xyz, ndr)
    draw_graph(xyz, ndx)
    # mask_x, _ = build_mask(ndx.reshape(-1, 1), "gmm", 20)
    mask_x = ndx > 4e-1
    mask_r, _ = build_mask(ndr.reshape(-1, 1), "gmm", 20)
    mask_union = np.bitwise_and(mask_x, mask_r)

    unmove_pts, move_pts = divide_mask(xyz, mask_union)

    mdx = dx.detach().cpu().numpy()[mask_union == 1]

    pcd_u = draw_one_color(unmove_pts, dx=0)
    pcd_m = draw_one_color(move_pts, dx=1.5)
    pcd_after_move = draw_one_color(move_pts + mdx, dx=3)

    dist = chamfer_distance_open3d(pcd_after_move, pcd_m)
    print(dist)

    # axis = np.array([0.00267, -0.0013, -1.61])
    # pivot = np.array([-0.31, 0.00012, -0.043])
    # line = draw_axis(pivot, axis)

    # o3d.visualization.draw_geometries([pcd_x, pcd_r, pcd_u])
    o3d.visualization.draw_geometries([pcd_u, pcd_m, pcd_after_move])
