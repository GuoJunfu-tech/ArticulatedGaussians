#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import math
from random import randint
from argparse import ArgumentParser, Namespace

import torch
from tqdm import tqdm
import uuid
import dill as pickle

from utils.loss_utils import l1_loss, ssim, kl_divergence, chamfer_distance_loss
from utils.general_utils import farthest_point_sampling
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, DeformModel, Revolute
from utils.general_utils import safe_state, get_linear_noise_func
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    # deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform = DeformModel()
    deform.train_setting(opt)

    revolute = Revolute()

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(
        lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000
    )

    # # load gaussians
    # import dill as pickle

    # with open("./load_data/first_frame_gaussian.pkl", "rb") as f:
    #     gaussians = pickle.load(f)
    # with open("./load_data/end_frame_gaussian.pkl", "rb") as f:
    #     end_frame_gaussians = pickle.load(f)

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration == opt.warm_up:
            scene.save(iteration)
            # deform.save_weights(args.model_path, iteration)
            # print(f"predicted articulated params:")
            # print(
            #     f"axis: {revolute.axis.tolist()}\n pivot: {revolute.pivot.tolist()}\n theta: {revolute.theta.item()}"
            # )
            # print(f"movable factors:")
            with torch.no_grad():
                #     factors = deform.movable_network(gaussians.get_xyz)
                #     move_parts = factors[factors > 0.8].shape[0]
                #     unmove_parts = factors[factors < 0.2].shape[0]
                #     print(
                #         f"move_parts: {move_parts}, unmove parts: {unmove_parts}, factors: {factors.shape[0]}"
                #     )
                # get_images(
                #     viewpoint_stack_end_frame,
                #     gaussians,
                #     deform,
                #     revolute,
                #     pipe,
                #     background,
                # )

                data = {"gaussians": gaussians}
                with open("warm_up_gaussians.pkl", "wb") as f:
                    pickle.dump(data, f)

            exit()
        if iteration < opt.warm_up:
            new_xyz, new_rotations = gaussians.get_xyz, gaussians.get_rotation

        elif iteration < opt.deformation:
            print(f"step 1 is over, now training deformation net")

            # N = gaussians.get_xyz.shape[0]

            # deformation
            new_xyz, new_rotations = deform.step(
                gaussians.get_xyz,
                gaussians.get_rotations,
                revolute.axis,
                revolute.pivot,
                revolute.theta,
            )

        d_scaling = 0.0  # TODO delete all d_scaling
        # Render
        render_pkg_re = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            new_xyz,
            new_rotations,
            d_scaling,
            dataset.is_6dof,
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg_re["render"],
            render_pkg_re["viewspace_points"],
            render_pkg_re["visibility_filter"],
            render_pkg_re["radii"],
        )
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device("cpu")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )

            # if iteration in saving_iterations:
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
            #     deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.deformation:  # TODO to be changed
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                # TODO find why these
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            # if iteration < opt.warm_up:
            gaussians.optimizer.step()
            gaussians.update_learning_rate(iteration)
            gaussians.optimizer.zero_grad(set_to_none=True)

            if opt.warm_up < iteration < opt.iterations:
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

                revolute.optimizer.step()
                revolute.optimizer.zero_grad()
                revolute.scheduler.step()

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    deform,
    load2gpu_on_the_fly,
    is_6dof=False,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config["cameras"]):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            *renderArgs,
                            d_xyz,
                            d_rotation,
                            d_scaling,
                            is_6dof,
                        )["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device("cpu")
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if (
                    config["name"] == "test"
                    or len(validation_configs[0]["cameras"]) == 0
                ):
                    test_psnr = psnr_test
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()

    return test_psnr


def get_images(
    viewpoint_cams,
    gaussians,
    deformModel,
    revoluteParams,
    pipe,
    background,
):
    # images = []
    for id, cam in enumerate(viewpoint_cams):
        new_xyz, new_rotations = deformModel.step(
            gaussians,
            revoluteParams.axis,
            revoluteParams.pivot,
            revoluteParams.theta,
            is_render=True,
        )
        render_pkg_re = render(
            cam, gaussians, pipe, background, new_xyz, new_rotations, 0.0, False
        )
        image = render_pkg_re["render"]
        image_np = image.detach().cpu().numpy().transpose((1, 2, 0))

        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.uint8(image_np * 255), "RGB")
        img.save(f"./rendered_img/{id}.png", format="PNG")


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
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
    )

    # All done
    print("\nTraining complete.")
