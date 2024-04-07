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
from PIL import Image
import numpy as np

import os
import sys
import math
from argparse import ArgumentParser, Namespace
import dill as pickle

import torch
from tqdm import tqdm
import uuid
import copy

from utils.loss_utils import (
    ll1_ssim_loss,
)
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, DeformModel, Revolute
from utils.general_utils import safe_state, get_linear_noise_func
from utils.image_utils import psnr
from utils.classification_utils import build_mask
from utils.viewpoint_utils import ViewpointLoader
from utils.visualization_utils import render_results
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

    scene_start = Scene(dataset, gaussians, status="start")
    scene_end = Scene(dataset, gaussians, status="end")

    viewpoint_loader = ViewpointLoader(scene_start, scene_end)

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(
        lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000
    )

    # load gaussians

    grads = {"xyz": [], "rotation": [], "accu": []}

    mask = None
    start = opt.pretrain
    # start = 1
    # end = opt.pretrain
    end = opt.continue_optimize_arti

    if start == opt.pretrain:
        with open("./load_data/stage_2.pkl", "rb") as f:
            data = pickle.load(f)
        gaussians = data["gaussians"]
        factors = data["factors"]
        revolute_params = data["params"]

    for iteration in range(start, end + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if iteration == end:
            # scene.save(iteration)
            # mask, centers = build_mask(factors.detach().cpu().numpy())
            # mask = torch.tensor(
            #     mask, device="cuda", dtype=torch.float32, requires_grad=False
            # )
            # deform.save_weights(args.model_path, iteration)
            print(f"predicted articulated params:")
            print(
                f"axis: {revolute.axis.tolist()}\n pivot: {revolute.pivot.tolist()}\n theta: {revolute.theta}\n"
            )
            print(f"movable factors:")
            #     #     factors = deform.movable_network(gaussians.get_xyz)
            #     # move_parts = factors[factors > 0.8].shape[0]
            #     # print(mask)
            #     # move_parts = mask.sum()
            #     # print(f"move parts: {move_parts}, factors: {mask.shape[0]}")

            if start == opt.pretrain:
                with open("grads.pkl", "wb") as f:
                    pickle.dump(grads, f)
                    print("data saved")

                # with torch.no_grad():
                render_results(
                    viewpoint_loader.get_cameras("start"),
                    gaussians,
                    deform,
                    revolute,
                    mask,
                    pipe,
                    background,
                    type="gif",
                )
            else:
                _, _, factors = render_results(
                    viewpoint_loader.get_cameras("start"),
                    gaussians,
                    deform,
                    revolute,
                    mask,
                    pipe,
                    background,
                    type="img",
                )

                data = {
                    "gaussians": gaussians,
                    "factors": factors,
                    "deformModel": deform,
                    "params": {
                        "axis": revolute.axis.tolist(),
                        "pivot": revolute.pivot.tolist(),
                    },
                }
                with open("./stage_2.pkl", "wb") as f:
                    pickle.dump(data, f)
                    print("data saved")

            exit()

        if iteration < opt.only_train_single_frame:
            viewpoint_cam_start = viewpoint_loader.get_viewpoint_cam(
                status="start", load2device=dataset.load2gpu_on_the_fly
            )

            new_xyz, new_rotations = None, None

        if opt.only_train_single_frame == iteration:
            print("[Training]::step 1 is over, now training deformation net")
            continue

        if opt.pretrain == iteration:
            print(f"[Training]::pretrain finished after {iteration} steps")
            mask, centers = build_mask(factors.detach().cpu().numpy())
            print((mask == 1).sum())
            mask = torch.tensor(
                mask, device="cuda", dtype=torch.float32, requires_grad=False
            )
            # revolute.set_theta(120 / 180 * math.pi)  # TODO delete
            revolute.set_theta(centers[1].item() * math.pi)
            revolute.reset_param_optimizer()
            # revolute.axis = revolute_params["axis"]
            # revolute.pivot = revolute_params["pivot"]
            revolute.axis = [1.0, 0.0, 0.0]
            revolute.pivot = [0.0, 0.008, 0.012]
            print(f"predicted articulated params:")
            print(
                f"axis: {revolute.axis.tolist()}\n pivot: {revolute.pivot.tolist()}\n theta: {revolute.theta}\n"
            )

            # if opt.only_train_single_frame + 1 <= iteration <= opt.continue_optimize_arti:

        if opt.only_train_single_frame < iteration <= opt.continue_optimize_arti:
            viewpoint_cam_start, viewpoint_cam_end = (
                viewpoint_loader.get_viewpoint_cam_dual(dataset.load2gpu_on_the_fly)
            )

            # print(viewpoint_loader._current_fid)
            # deformation
            new_xyz, new_rotations, factors = deform.step(
                gaussians,
                revolute,
                mask,
            )

        d_scaling = 0.0  # TODO delete all d_scaling
        # Render
        # deform frame
        if opt.pretrain < iteration < opt.continue_optimize_arti:
            gaussians._scaling = gaussians._scaling.detach()
            gaussians._opacity = gaussians._opacity.detach()


        loss_start = 0.0
        if iteration < opt.pretrain:
            render_pkg_re = render(
                viewpoint_cam_start,
                gaussians,
                pipe,
                background,
                gaussians.get_xyz,
                gaussians.get_rotation,
            )

            (
                image_start,
                viewspace_point_tensor_start,
                visibility_filter_start,
                radii_start,
            ) = (
                render_pkg_re["render"],
                render_pkg_re["viewspace_points"],
                render_pkg_re["visibility_filter"],
                render_pkg_re["radii"],
            )
            gt_image_start = viewpoint_cam_start.original_image.cuda()
            loss_start = ll1_ssim_loss(image_start, gt_image_start, opt.lambda_dssim)
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam_start.load2device("cpu")

        loss_end = 0.0
        if opt.only_train_single_frame < iteration < opt.continue_optimize_arti:
            render_pkg_re = render(
                viewpoint_cam_end,
                gaussians,
                pipe,
                background,
                new_xyz,
                new_rotations,
                d_scaling,
                dataset.is_6dof,
            )
            image_end, viewspace_point_tensor_end, visibility_filter_end, radii_end = (
                render_pkg_re["render"],
                render_pkg_re["viewspace_points"],
                render_pkg_re["visibility_filter"],
                render_pkg_re["radii"],
            )
            gt_image_end = viewpoint_cam_end.original_image.cuda()
            loss_end = ll1_ssim_loss(image_end, gt_image_end, opt.lambda_dssim)

        # ---------------- output mid results -------------------------------
        if (
            6400 <= iteration < 6450
            or 22000 <= iteration < 22050
            or 26800 <= iteration < 26850
        ):
            image_np = image_end.detach().cpu().numpy().transpose((1, 2, 0))
            img = Image.fromarray(np.uint8(image_np * 255), "RGB")
            save_path = os.path.join(
                os.getcwd(), f"rendered_img/static_{iteration}.png"
            )
            img.save(save_path, "PNG")
        # ----------------------------------------------------------------------

        # Loss

        loss = loss_end + loss_start

        loss.backward()

        if start == opt.pretrain:
            pass
            # grads["xyz"].append(copy.deepcopy(gaussians._xyz.grad.detach().cpu()))
            # grads["rotation"].append(
            #     copy.deepcopy(gaussians._rotation.grad.detach().cpu())
            # )
            # grads["accu"].append(copy.deepcopy(gaussians.xyz_gradient_accum))

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam_end.load2device("cpu")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                sp_loss = 0
                progress_bar.set_postfix(
                    {
                        # "Loss_total": f"{ema_loss_for_log:.{7}f}",
                        "move_loss": f"{loss_end:.{7}f}",
                        "unmove_loss": f"{loss_start:.{7}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if opt.only_train_single_frame < iteration < opt.pretrain:
                gaussians.max_radii2D[visibility_filter_end] = torch.max(
                    gaussians.max_radii2D[visibility_filter_end],
                    radii_end[visibility_filter_end],
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor_end, visibility_filter_end
                )

            if iteration < opt.pretrain:
                gaussians.max_radii2D[visibility_filter_start] = torch.max(
                    gaussians.max_radii2D[visibility_filter_start],
                    radii_start[visibility_filter_start],
                )
                # FIXME sick code! should update together!!

                # if iteration in saving_iterations:
                #     print("\n[ITER {}] Saving Gaussians".format(iteration))
                #     scene.save(iteration)
                #     deform.save_weights(args.model_path, iteration)

                # Optimizer step

                # --------------------- Densification --------------------------

                gaussians.add_densification_stats(
                    viewspace_point_tensor_start, visibility_filter_start
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    # FIXME only densify and prune the start scene
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene_start.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # --------- change mask -----------------
            # TODO

            # ----------------------------------------

            if opt.only_train_single_frame < iteration < opt.pretrain:
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

            if opt.only_train_single_frame < iteration < opt.continue_optimize_arti:
                revolute.axis_pivot_optimizer.step()
                revolute.axis_pivot_optimizer.zero_grad()
                revolute.axis_pivot_scheduler.step()

            if opt.pretrain < iteration < opt.continue_optimize_arti:
                revolute.theta_optimizer.step()
                revolute.theta_optimizer.zero_grad()
                revolute.theta_scheduler.step()

            if iteration < opt.continue_optimize_arti:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

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
