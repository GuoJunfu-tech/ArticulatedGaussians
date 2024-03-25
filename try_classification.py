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

from PIL import Image
import dill as pickle
import numpy as np


def classification(dataset, pipe):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    with open("./load_data/end_frame_gaussian.pkl", "rb") as f:
        gaussians = pickle.load(f)
    # gaussians.training_setup(opt)
    end_gaussian = gaussians["gaussians"]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    # origin_opacity = gaussians.get_opacity
    # binary_tensor = torch.ones_like(origin_opacity)
    # L = binary_tensor.shape[0]
    # print(L)
    # binary_tensor[0 : L // 40] = 0
    # print(binary_tensor)
    # opacities = origin_opacity * binary_tensor
    # opacities = torch.ones_like(gaussians.get_opacity) * 0.2
    # xyz = gaussians.get_xyz
    # binary_tensor = (xyz[:, 2] < 0.0).int()
    # opacities = origin_opacity * binary_tensor
    # opacities = binary_tensor.float()
    xyz = end_gaussian.get_xyz
    grad = torch.zeros(xyz.shape[0], 1, device="cuda", dtype=torch.float32)

    for id, cam in enumerate(viewpoint_stack):
        if cam.fid == 1:
            continue
        render_pkg_re = render(
            cam,
            end_gaussian,
            pipe,
            background,
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg_re["render"],
            render_pkg_re["viewspace_points"],
            render_pkg_re["visibility_filter"],
            render_pkg_re["radii"],
        )

        # Loss
        gt_image = cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(image, gt_image))
        loss.backward()

        print(f"cam #{id}, loss: {loss.item()}")
        # print(end_gaussian._xyz.grad)
        grad += torch.norm(end_gaussian._xyz.grad, p=2, dim=1, keepdim=True)
        grad += torch.norm(end_gaussian._rotation.grad, p=2, dim=1, keepdim=True)
        grad += torch.norm(end_gaussian._scaling.grad, p=2, dim=1, keepdim=True)
        # end_gaussian.optimizer.step()
        end_gaussian._xyz.grad.data.zero_()

        # image_np = image.detach().cpu().numpy().transpose((1, 2, 0))

        # img = Image.fromarray(np.uint8(image_np * 255), "RGB")
        # img.save(f"./rendered_img/{id}_{int(cam.fid.item())}.png", format="PNG")

    # output = np.hstack(xyz.detach().numpy(), grad.detach().numpy())
    grad = grad.detach().cpu().numpy()

    min_g, max_g = grad.min(), grad.max()
    grad = (grad - min_g) / (max_g - min_g)
    grad = grad ** (1 / 2)
    print(grad, (grad > 0.5).sum(), grad.shape[0])
    xyz = xyz.detach().cpu().numpy()
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # colors = intensity_to_color(grad)
    colors = np.array([intensity_to_color(i) for i in grad]).astype(np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


def intensity_to_color(intensity):
    blue = int(intensity * 255)  # 蓝色通道
    red = int((1 - intensity) * 255)  # 红色通道
    green = 0  # 绿色通道固定为0
    return [red / 255.0, green / 255.0, blue / 255.0]


def fun():
    # Create a parameter tensor with requires_grad=True
    param = torch.tensor([1.0], requires_grad=True)

    # Define a loss function
    def loss_function(param):
        return (param - 2) ** 2  # For example, square of the difference from 2

    # Initialize an optimizer (here using SGD for demonstration)
    optimizer = torch.optim.SGD([param], lr=0.1)

    print("no gradient:", param.grad)
    # Forward pass and compute the loss
    loss = loss_function(param)

    # Perform backward pass to compute gradients
    loss.backward()

    # The first gradient is now stored in param.grad
    print("First gradient:", param.grad)

    # Now, let's compute the loss and backward() again without calling optimizer.step()
    loss = loss_function(param)
    loss.backward()

    # Since we didn't call optimizer.step(), the gradients have accumulated
    print("Accumulated gradients:", param.grad)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    classification(
        lp.extract(args),
        pp.extract(args),
    )

    # All done
    print("\nTraining complete.")
