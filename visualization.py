import dill as pickle
import torch
from scene import Scene, GaussianModel, DeformModel, Revolute
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
from mayavi import mlab
import numpy as np
import open3d as o3d
from PIL import Image
import sys
import math
import copy
from argparse import ArgumentParser, Namespace
from gaussian_renderer import render, network_gui
from utils.loss_utils import l1_loss, ssim, chamfer_distance_loss
from utils.general_utils import farthest_point_sampling
from utils.classification_utils import kmeans, gmm
from utils.classification_utils import build_mask


def draw_graph(xyz, factor):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 生成一些随机数据
    data = factor.detach().cpu().numpy()
    print(max(data))

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30)
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


def draw_one_color(xyz, filter, dx=0.0):
    pcd = o3d.geometry.PointCloud()
    new_xyz = xyz.copy()

    if dx:
        new_xyz[:, 0] += dx
    pcd.points = o3d.utility.Vector3dVector(new_xyz)

    colors = np.zeros((xyz.shape[0], 3))
    print(filter.shape)
    for id, is_vis in enumerate(filter):
        if is_vis:
            colors[id, :] = [1, 0, 0]
        else:
            colors[id, :] = [0, 0, 1]

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
    # 假设我们有上万个椭球的数据，这里我们用随机数据来模拟
    # 每个椭球由其中心点和三个轴的长度表示
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
        resolution=8,
        color=(0.5, 0.5, 1.0),
        opacity=0.6,
        mode="sphere",
    )

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


if __name__ == "__main__":
    test_maya()
    exit()
    with open("./load_data/sci.pkl", "rb") as f:
        # with open("./final_params.pkl", "rb") as f:
        data = pickle.load(f)

    gaussians = data["gaussians"]
    # # factors = data["factor"].detach().cpu().numpy()
    # factors = gaussians._movable_mask.detach().cpu().numpy()

    # xyz = gaussians.get_xyz.detach().cpu().numpy()
    # pcd_1 = draw_one_color(xyz, factors)

    # with open("./load_data/stage_3.pkl", "rb") as f:
    #     # with open("./final_params.pkl", "rb") as f:
    #     data = pickle.load(f)
    # with open("./load_data/grads.pkl", "rb") as f:
    #     grads = pickle.load(f)

    # gaussians = grads["gaussians"]
    # factors = data["factor"].detach().cpu().numpy()
    factors = gaussians._movable_mask.detach().cpu().numpy()

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    pcd_2 = draw_one_color(
        xyz,
        factors,
    )

    # # mask = gaussians.get_movable_mask.detach().cpu().numpy()

    # xyz_grads = grads["xyz"]
    # print(len(xyz_grads))
    # grad = 0.0
    # for id, xyz_grad in enumerate(xyz_grads):
    #     if id == 90:
    #         break
    #     g = torch.norm(xyz_grad, dim=-1, keepdim=True).numpy()
    #     grad += g.squeeze()
    # # grad = torch.norm(xyz_grads[-3], dim=-1, keepdim=True).numpy()
    # print(max(grad), min(grad))
    # factor = grad > 5e-3

    # pcd_1 = draw_one_color(xyz, factor)

    o3d.visualization.draw_geometries([pcd_2])
    # visualize(xyz, factors)
    # print(xyz_grads[1:10])
    # grad_1 = np.zeros_like(factors.detach().cpu())
    # grad_2 = grad_1.copy()
    # for id, xyz_grad in enumerate(xyz_grads):
    #     g = torch.norm(xyz_grad, dim=-1, keepdim=True).numpy()
    #     if id < 100:
    #         grad_1 += g
    #     else:
    #         grad_2 += g

    # print(max(grad), min(grad))
    # print(grad)
    # xyz_grad = grads["accu"][0].detach().cpu().numpy().squeeze()
    # visualize(xyz, factors, grad)
    # draw_two_grads(xyz, grad_1, grad_2)
    # draw_graph(xyz, data["factors"])
    # g_1 = grads["xyz"][0]
    # g_2 = grads["xyz"][-1]

    # print(torch.equal(g_1, g_2))
    # print(grads["xyz"])
