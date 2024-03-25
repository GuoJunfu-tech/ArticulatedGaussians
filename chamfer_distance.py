import os
import sys
import math
from random import randint
from argparse import ArgumentParser, Namespace

import torch
from tqdm import tqdm
import uuid

from utils.loss_utils import l1_loss, ssim, kl_divergence, chamfer_distance_loss
from utils.general_utils import farthest_point_sampling
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, DeformModel, Revolute
from utils.general_utils import safe_state, get_linear_noise_func
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams


def training(opt):
    ITERATION = 3000

    deform = DeformModel()
    deform.train_setting(opt)

    revolute = Revolute()

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    progress_bar = tqdm(range(ITERATION), desc="Training progress")

    import dill as pickle

    print(f"Loading datasets")
    with open("./load_data/first_frame_gaussian.pkl", "rb") as f:
        gaussians = pickle.load(f)
    with open("./load_data/end_frame_gaussian.pkl", "rb") as f:
        end_frame_gaussians = pickle.load(f)

    xyz = farthest_point_sampling(gaussians.get_xyz, 1000)
    rotations = torch.zeros((xyz.shape[0], 4), dtype=torch.float32, device="cuda")
    rotations[:, 0] = 1
    gt_xyz = farthest_point_sampling(
        end_frame_gaussians["gaussians"].get_xyz, 10000
    ).detach()

    for iteration in range(ITERATION):
        new_xyz, new_rotations, movable_factor = deform.step(
            xyz, rotations, revolute._axis, revolute._pivot, revolute._theta
        )
        mid_loss = ((movable_factor - 0.5) ** 2).sum()
        cd_loss = chamfer_distance_loss(new_xyz, gt_xyz)
        loss = cd_loss + mid_loss
        loss.backward()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"CD Loss": f"{cd_loss:.{7}f}", "mid Loss": f"{mid_loss:.{7}f}"}
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            deform.optimizer.step()
            deform.optimizer.zero_grad()
            deform.update_learning_rate(iteration)

            revolute.optimizer.step()
            revolute.optimizer.zero_grad()
            revolute.scheduler.step()

    with torch.no_grad():
        factors = deform.movable_network(xyz)

    print(f"training complete!")
    print(
        f"final rotate params: \naxis: {revolute._axis.tolist()}, \npivot:{revolute._pivot.tolist()}, \ntheta: {revolute._theta.item()}"
    )
    return new_xyz, gt_xyz, factors


def visualization(new_xyz, gt_xyz, factors):
    import numpy as np
    import open3d as o3d

    gt_xyz = gt_xyz.cpu().detach().numpy()
    new_xyz = new_xyz.cpu().detach().numpy()
    factors = factors.cpu().detach().numpy()

    gt_xyz[:, 0] += 2
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_xyz)
    # gt_pcd.colors = o3d.utility.Vector3dVector([1, 0, 0])
    gt_pcd.paint_uniform_color([1, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_xyz)
    print(f"moved points: {( factors > 0.8 ).sum()} / {factors.shape[0]}")
    colors = np.hstack((factors, np.zeros((factors.shape[0], 2))))
    colors = o3d.utility.Vector3dVector(colors)
    pcd.colors = colors

    # o3d.visualization.draw_geometries([pcd, gt_pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(gt_pcd)
    vis.add_geometry(pcd)

    # Here's the crucial part: Adjusting the point size
    opt = vis.get_render_option()
    opt.point_size = 5  # Adjust the point size as needed

    # Run the visualizer
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)),
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[15_000, 20_000, 30_000, 40000],
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    xyz, gt_xyz, factors = training(op.extract(args))
    visualization(xyz, gt_xyz, factors)
