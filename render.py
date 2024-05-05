import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, DeformModel, Revolute, Prismatic, DeformGS
import imageio
import json
import numpy as np
from PIL import Image
import sys


def render_set(
    dataset: ModelParams,
    pipeline: PipelineParams,
    motion,
    ply_path,
    paths,
):
    depth_path = paths["depth"]
    gt_path = paths["gt"]
    render_path = paths["render"]

    deformModel = DeformModel()

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(ply_path)

    with torch.no_grad():
        # deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        # deform.load_weights(dataset.model_path)
        arti_params = Revolute() if motion["type"] == "revolute" else Prismatic()
        arti_params.load_params(motion)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for status in ["start", "end"]:
            scene = Scene(dataset, gaussians, status=status)
            views = scene.getTestCameras()

            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                view.load2device()
                # fid = view.fid
                # time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                if status == "end":
                    new_xyz, new_rotations, _ = deformModel.step(
                        gaussians, arti_params, keep_gs_grad=False
                    )
                else:
                    new_xyz = gaussians.get_xyz
                    new_rotations = gaussians.get_rotation

                render_pkg_re = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    new_xyz,
                    new_rotations,
                )
                # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                rendering = render_pkg_re["render"]
                depth = render_pkg_re["depth"]
                depth = depth / (depth.max() + 1e-5)
                gt = view.original_image[0:3, :, :]

                for path, image in zip(
                    [render_path, gt_path, depth_path], [rendering, gt, depth]
                ):
                    output_path = os.path.join(path, status)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    torchvision.utils.save_image(
                        image,
                        os.path.join(output_path, "{0:05d}".format(idx) + ".png"),
                    )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    # op = OptimizationParams(parser)
    # parser.add_argument(
    #     "--output_path", "-o", required=True, nargs="+", type=str, default=[]
    # )
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    # args = parser.parse_args(sys.argv[1:])
    print("Rendering " + args.model_path)
    output_root = args.model_path
    motion_path = os.path.join(output_root, "motion.json")
    with open(motion_path, "r") as f:
        motion = json.load(f)

    ply_path = os.path.join(output_root, "point_cloud/iteration_40000/point_cloud.ply")
    if not os.path.exists(ply_path):
        print("No ply file found at " + ply_path)
        exit()

    rendered_path = os.path.join(output_root, "rendered_imgs")
    if not os.path.exists(rendered_path):
        os.makedirs(rendered_path)
    depth_path = os.path.join(output_root, "depth_imgs")
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    gt_path = os.path.join(output_root, "gt_imgs")
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)

    img_paths = {
        "render": rendered_path,
        "depth": depth_path,
        "gt": gt_path,
    }

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_set(
        model.extract(args),
        pipeline.extract(args),
        motion,
        ply_path,
        img_paths,
    )
