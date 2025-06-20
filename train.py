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

# import math
from argparse import ArgumentParser, Namespace
import dill as pickle

import torch
from tqdm import tqdm
import uuid

from utils.loss_utils import ll1_ssim_loss, l1_loss, arap_loss, chamfer_distance_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, DeformModel, Revolute, Prismatic, DeformGS
from utils.general_utils import safe_state, get_linear_noise_func
from utils.image_utils import psnr
from utils.classification_utils import build_mask, mask_init
from utils.viewpoint_utils import ViewpointLoader
from utils.visualization_utils import render_results
from utils.knn_utils import knn
from utils.arti_estimation_utils import estimate_arti_info
from arguments import ModelParams, PipelineParams, OptimizationParams

from omegaconf import OmegaConf

import copy

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(cfg, dataset, opt, pipe, testing_iterations, saving_iterations):
    if opt.tb_writer:
        tb_writer = prepare_output_and_logger(dataset)
    else:
        tb_writer = False
    gaussians = GaussianModel(dataset.sh_degree)

    deformArti = DeformModel()

    deformGS = DeformGS(opt)
    revolute = Revolute()
    prismatic = Prismatic()

    scene_start = Scene(dataset, gaussians, status="start")
    scene_end = Scene(dataset, gaussians, status="end")
    viewpoint_loader = ViewpointLoader(scene_start, scene_end)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    best_psnr = 0.0
    best_iteration = 0
    smooth_term = get_linear_noise_func(
        lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000
    )

    mask = None
    d_xyz = None
    d_rotations = None  # TODO delete after code finish
    # start = opt.pretrain
    start = 1
    # end = opt.pretrain
    end = opt.update_mask
    progress_bar = tqdm(range(end), desc="Training progress")

    deform = None
    neighbor_dist = None
    deformed_xyz = torch.empty(0)
    # is_end_frame_with_grad = False
    # grad_counter = 0

    arti_params = revolute
    # # arti_params = prismatic

    object_name = dataset.model_path.split("/")[-1]

    for iteration in range(start, end + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree

        if iteration == end:
            # deform.save_weights(args.model_path, iteration)
            if end == opt.update_mask:
                object_name = dataset.model_path.split("/")[-1]
                render_results(
                    viewpoint_loader.get_cameras("start"),
                    gaussians,
                    deform,
                    arti_params,
                    mask,
                    pipe,
                    background,
                    type="gif",
                    note=f"final_{object_name}",
                )

            print("Training finished.")
            scene_start.save(iteration)
            save_motion_path = os.path.join(scene_start.model_path, "motion.json")
            arti_params.save_json(save_motion_path)

            print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))
            exit()

        if opt.only_train_single_frame == iteration:
            print("[Training]::step 1 is over, now training deformation net")
            print("[Training]::building knn trees")
            neighbor_sq_dist, neighbor_indices = knn(
                gaussians.get_xyz.detach().cpu().numpy(), 20
            )
            weight = np.exp(-20 * neighbor_sq_dist)
            dist = np.sqrt(neighbor_sq_dist)
            neighbor_weight = torch.tensor(weight).float().to(gaussians.get_xyz.device)
            neighbor_dist = torch.tensor(dist).float().to(gaussians.get_xyz.device)

            continue

        if opt.pretrain == iteration:
            print("[Training]::step 2 is over, now update the mask")

            with torch.no_grad():
                _, _, (d_xyz, d_rotations) = deformGS.step(gaussians)
                ndx = torch.norm(d_xyz, dim=-1).detach().cpu().numpy()
                ndx = (ndx - min(ndx)) / (max(ndx) - min(ndx))
                mask_xyz = ndx > 1e-1

                # mask_u = mask_init(ndr, ndx, 3e-1)  # TODO set optional threshold
                xyz = gaussians.get_xyz.detach()
                deformed_xyz = xyz + d_xyz.detach()
                deformed_xyz = deformed_xyz[mask_xyz == 1].detach()

            gaussians.initialize_mask(mask_xyz)

            # param init
            # if arti_params.type == "revolute":
            #     axis, pivot, theta = estimate_arti_info(
            #         xyz[mask_xyz == 1], deformed_xyz
            #     )
            #     arti_params.set_params(axis, theta/2, pivot)

            data = {
                "gaussians": gaussians,
                "dx": d_xyz,
                "dr": d_rotations,
                # "deformModel": deform,
                "params": {
                    "axis": revolute.axis.tolist(),
                    "pivot": revolute.pivot.tolist(),
                },
            }

            with open("./stage_2.pkl", "wb") as f:
                pickle.dump(data, f)
                print(" stage 2 data saved")

            continue

        if iteration == opt.update_params:
            print("[Training]::step 3 is over, now update the articulated params")
            with torch.no_grad():
                _, _, (d_xyz, d_rotations) = deformGS.step(gaussians)
                ndx = torch.norm(d_xyz, dim=-1).detach().cpu().numpy()
                ndx = (ndx - min(ndx)) / (max(ndx) - min(ndx))
                mask_xyz = ndx > 1e-1
                gaussians.initialize_mask(mask_xyz)

                if arti_params.type == "revolute":
                    arti_params.theta_normalization()

            render_results(
                viewpoint_loader.get_cameras("start"),
                gaussians,
                deform,
                arti_params,
                mask,
                pipe,
                background,
                type="gif",
                note=f"param_{object_name}",
            )
            continue

        if iteration < opt.only_train_single_frame:
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()  # TODO temporary

            viewpoint_cam_start = viewpoint_loader.get_viewpoint_cam(
                status="start", load2device=dataset.load2gpu_on_the_fly
            )
            new_xyz, new_rotations = None, None
        # elif opt.only_train_single_frame < iteration < opt.pretrain:
        #     viewpoint_cam_start, viewpoint_cam_end = (
        #         viewpoint_loader.get_viewpoint_cam_dual(dataset.load2gpu_on_the_fly)
        #     )
        elif opt.only_train_single_frame < iteration < opt.update_params:
            viewpoint_cam_end = viewpoint_loader.get_viewpoint_cam(
                status="end", load2device=dataset.load2gpu_on_the_fly
            )

        elif opt.update_params < iteration < opt.update_mask:
            viewpoint_cam_start, viewpoint_cam_end = (
                viewpoint_loader.get_viewpoint_cam_dual(dataset.load2gpu_on_the_fly)
            )
            # if iteration % 1000 == 0:
            # gaussians.oneupSHdegree()

        # ------------------- core: deformation ----------------------------

        # before pretrain, we do not train the articulated params
        if iteration > opt.only_train_single_frame:
            gs_no_grad = True if iteration < opt.update_params else False
            # gs_no_grad = True
            deform = deformGS if iteration < opt.pretrain else deformArti
            new_xyz, new_rotations, _ = deform.step(
                gaussians, arti_params, gs_no_grad=gs_no_grad
            )

        # ---------------------- render --------------------------
        loss_start = 0.0
        if (iteration < opt.only_train_single_frame) or (iteration > opt.update_params):
            # or (iter_counter < opt.update_mask_interval / 2):
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
        if opt.only_train_single_frame < iteration:
            # or ( opt.update_mask_interval / 2 <= iter_counter < opt.update_mask_interval
            # ):
            render_pkg_re = render(
                viewpoint_cam_end,
                gaussians,
                pipe,
                background,
                new_xyz,
                new_rotations,
            )
            image_end, viewspace_point_tensor_end, visibility_filter_end, radii_end = (
                render_pkg_re["render"],
                render_pkg_re["viewspace_points"],
                render_pkg_re["visibility_filter"],
                render_pkg_re["radii"],
            )
            gt_image_end = viewpoint_cam_end.original_image.cuda()
            loss_end = ll1_ssim_loss(image_end, gt_image_end, opt.lambda_dssim)

        loss_arap = 0.0
        if opt.only_train_single_frame < iteration < opt.pretrain:
            loss_arap = arap_loss(
                new_xyz, neighbor_indices, neighbor_dist, neighbor_weight
            )

        loss_cd = 0.0
        if opt.pretrain < iteration < opt.update_params:
            source_xyz = new_xyz[gaussians.get_movable_mask == 1]
            loss_cd = chamfer_distance_loss(deformed_xyz, source_xyz)
            # loss_cd = 0

        weighted_loss_arap = loss_arap
        if iteration < opt.update_params:
            # w = 1 / (loss_start.item() + 1e-6) * 1e-3
            loss = loss_end + loss_start + 0.1 * loss_cd + weighted_loss_arap
            # loss = loss_end + loss_start + weighted_loss_arap
        else:
            # loss = loss_end + loss_start
            # when joint optimization, update the state with higher loss
            # enlarge_weight = 2.
            # loss = 2 * loss_end + loss_start
            # total_loss = loss_end.detach().clone + loss_start.detach().clone()
            weight_start = loss_start.item() / (loss_end.item() + 1e-8)
            weight_end = loss_end.item() / (loss_start.item() + 1e-8)
            weight_sum = weight_end + weight_start
            loss = (
                weight_start / weight_sum * loss_start
                + weight_end / weight_sum * loss_end
            )
            # loss = (1.0 / loss_end.item()) * loss_start + (
            #     1.0 / loss_start.item()
            # ) * loss_end

        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam_end.load2device("cpu")

        with torch.no_grad():
            # ---------------------- progress bar -----------------------------
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "m_l": f"{loss_end:.{7}f}",
                        "u_l": f"{loss_start:.{7}f}",
                        # "d_l": f"{dist_loss:.{7}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # ---------------------- training report -----------------------------
            losses = {
                "start": loss_start,
                "end": loss_end,
                "arap": weighted_loss_arap,
                "cd": loss_cd,
            }
            if opt.tb_writer and (iteration % opt.report_interval == 0):
                training_report(
                    tb_writer,
                    iteration,
                    iter_start.elapsed_time(iter_end),
                    losses,
                    gaussians.get_xyz.shape[0],
                    gaussians.get_opacity,
                    arti_params,
                )
            if opt.is_eval:
                if iteration in testing_iterations:
                    is_first_test = (
                        True if iteration == testing_iterations[0] else False
                    )
                    cur_psnr = eval(
                        iteration,
                        scene_start,
                        scene_end,
                        gaussians,
                        render,
                        (pipe, background),
                        deform,
                        arti_params,
                        tb_writer,
                        dataset.load2gpu_on_the_fly,
                        is_first_test,
                    )

                    if cur_psnr > best_psnr:
                        best_psnr = cur_psnr
                        best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene_start.save(iteration)
                save_motion_path = os.path.join(
                    scene_start.model_path, f"motion_{iteration}.json"
                )
                arti_params.save_json(save_motion_path)

            # --------------------- Densification --------------------------
            # Keep track of max radii in image-space for pruning
            if iteration < opt.stop_densify:
                is_densify = False
                if iteration > opt.update_params:
                    gaussians.max_radii2D[visibility_filter_end] = torch.max(
                        gaussians.max_radii2D[visibility_filter_end],
                        radii_end[visibility_filter_end],
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor_end, visibility_filter_end
                    )
                    is_densify = True

                    # FIXME sick code! should update together!!

                if (iteration < opt.only_train_single_frame) or (
                    iteration > opt.update_params
                ):
                    gaussians.max_radii2D[visibility_filter_start] = torch.max(
                        gaussians.max_radii2D[visibility_filter_start],
                        radii_start[visibility_filter_start],
                    )

                    gaussians.add_densification_stats(
                        viewspace_point_tensor_start, visibility_filter_start
                    )
                    is_densify = True

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and is_densify
                ):
                    is_densify = False
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene_end.cameras_extent,
                        size_threshold,
                    )

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        gaussians.reset_opacity()

            # --------------- optimization -------------------------

            if opt.only_train_single_frame < iteration < opt.pretrain:
                # either deformGS or deformArti
                deform.optimizer.step()  # FIXME Deform.optimizer is now only deformGS
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

            if opt.pretrain < iteration:
                arti_params.optimizer.step()
                arti_params.optimizer.zero_grad()
                arti_params.scheduler.step()

            if (iteration < opt.only_train_single_frame) or (
                iteration > opt.update_params
            ):
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.scheduler.step()

            # deform.optimizer.zero_grad()
            gaussians.optimizer.zero_grad(set_to_none=True)
            arti_params.optimizer.zero_grad()


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


def vis(image):
    """TODO delete in the future
    this func is to visualize the mid results
    """
    from PIL import Image

    image_np = image.detach().cpu().numpy().transpose((1, 2, 0))
    img = Image.fromarray(np.uint8(image_np * 255), "RGB")
    img.show()


def eval(
    iteration,
    scene_start,
    scene_end,
    gaussians,
    renderFunc,
    renderArgs,
    deform,
    arti_params,
    tb_writer,
    load2gpu_on_the_fly,
    is_first_test=False,
):
    test_psnr = 0.0
    # Report test and samples of training set
    l1_test, psnr_test = 0.0, 0.0
    for status in ["start", "end"]:
        scene = scene_start if status == "start" else scene_end
        if not scene:
            continue
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

                    if status == "end":
                        xyz, rotation, _ = deform.step(gaussians, arti_params)
                    else:
                        xyz, rotation = gaussians.get_xyz, gaussians.get_rotation

                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            gaussians,
                            *renderArgs,
                            xyz,
                            rotation,
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
                            + f"_view_{viewpoint.image_name}_{status}/render",
                            image[None],
                            global_step=iteration,
                        )
                        if is_first_test:
                            tb_writer.add_images(
                                config["name"]
                                + f"_view_{viewpoint.image_name}_{status}/ground_truth",
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

                type = config["name"]
                print(
                    f"[ITER {iteration}] Evaluating {type} -{status}: L1 {l1_test} PSNR {psnr_test}"
                )
            # if tb_writer:
            #     tb_writer.add_scalar(
            #         config["name"] + "/loss_viewpoint - l1_loss",
            #         l1_test,
            #         iteration,
            #     )
            #     tb_writer.add_scalar(
            #         config["name"] + "/loss_viewpoint - psnr",
            #         psnr_test,
            #         iteration,
            #     )
    torch.cuda.empty_cache()
    return test_psnr


def training_report(
    tb_writer,
    iteration,
    elapsed,
    losses,
    gs_num,
    opacity,
    arti_params,
):
    if tb_writer:
        for name, loss in losses.items():
            loss = loss.item() if isinstance(loss, torch.Tensor) else loss
            tb_writer.add_scalar(f"train_loss_patches/{name}", loss, iteration)

        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_histogram("scene/opacity_histogram", opacity, iteration)
        tb_writer.add_scalar("total_points", gs_num, iteration)

        # for i in range(3):
        #     tb_writer.add_scalar(
        #         "revolute/axis_{}".format(i), revolute.axis[i], iteration
        #     )
        #     tb_writer.add_scalar(
        #         "revolute/pivot_{}".format(i), revolute.pivot[i], iteration
        #     )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument(
        "--cfg_file",
        required=False,
        type=str,
        default="./cfg_files/train.yaml",
    )
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[
            # 5500,
            7000,
            10000,
            14500,
            16500,
            19000,
            22000,
            24000,
            32000,
            35000,
            38000,
            42000,
            46000,
            49000,
        ],
        # default = [20000,]
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[60000],
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    cfg = OmegaConf.load(args.cfg_file)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        cfg,
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
    )

    # All done
    print("\nTraining complete.")
