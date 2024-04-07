import dill as pickle
import torch
from scene import Scene, GaussianModel, DeformModel, Revolute
from arguments import ModelParams, PipelineParams, OptimizationParams
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


def value_to_color(value):
    """将单个数值转换为RGB颜色。"""
    # 确保数值在0和1之间
    value = np.clip(value, 0, 1)
    # 线性插值计算R和B分量
    red = int(255 * value)
    blue = int(255 * (1 - value))
    # 返回RGB颜色
    return [red, 0, blue]


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
    xyz = xyz.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if factor is None:
        factor = np.ones((xyz.shape[0], 1))
    else:
        factor = factor.detach().cpu().numpy()

    mask, centers = build_mask(factor, "gmm")
    # mask = abs(factor) < 5e-2
    # print(centers)
    # mask = abs(factor)

    colors = np.zeros((xyz.shape[0], 3))
    for cluster_id, cluster in enumerate(mask):
        colors[cluster_id, :] = [1, 0, 0] * cluster + [0, 0, 1] * (1 - cluster)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # colors = (w < 5e-2) * [1, 0, 0] + (w >= 5e-2) * [0, 0, 1]
    vis = [pcd]

    # ----------------------------------------------------

    # ----------------- Grads Visualization ----------------

    if grad is not None:
        pcd_grad = o3d.geometry.PointCloud()
        # mask = grad
        w = (grad - grad.min()) / (grad.max() - grad.min())
        w = w.squeeze()

        colors = np.zeros((xyz.shape[0], 3))
        for cluster_id, cluster in enumerate(w):
            colors[cluster_id, :] = np.array([1.0, 0.0, 0.0]) * cluster + np.array(
                [0.0, 0.0, 1.0]
            ) * (1 - cluster)

        pcd_grad.colors = o3d.utility.Vector3dVector(colors)
        xyz_hard = xyz.copy()
        xyz_hard[:, 0] += 1.0
        pcd_grad.points = o3d.utility.Vector3dVector(xyz_hard)

        vis.append(pcd_grad)

    o3d.visualization.draw_geometries(vis)


def draw_one_color(xyz, filter):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    colors = np.zeros((xyz.shape[0], 3))
    for id, is_vis in enumerate(filter):
        if is_vis:
            colors[id, :] = [1, 0, 0]
        else:
            colors[id, :] = [0, 0, 1]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    with open("./load_data/stage_2.pkl", "rb") as f:
        # with open("./final_params.pkl", "rb") as f:
        data = pickle.load(f)

    gaussians = data["gaussians"]
    factors = data["factors"]

    xyz = gaussians.get_xyz

    with open("./load_data/grads.pkl", "rb") as f:
        grads = pickle.load(f)

    # print(grads["xyz"])

    xyz_grads = grads["xyz"]
    grad = np.zeros_like(factors.detach().cpu())
    for id, xyz_grad in enumerate(xyz_grads):
        grad += torch.norm(grads["rotation"][id], dim=-1, keepdim=True).numpy()

    # print(max(grad), min(grad))
    # xyz_grad = grads["accu"][0].detach().cpu().numpy().squeeze()
    visualize(xyz, factors, grad)
    draw_graph(xyz, data["factors"])
    # g_1 = grads["xyz"][0]
    # g_2 = grads["xyz"][-1]

    # print(torch.equal(g_1, g_2))
    # print(grads["xyz"])
