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

from utils.loss_utils import ll1_ssim_loss, l1_loss, arap_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, DeformModel, Revolute, DeformGS
from utils.general_utils import safe_state, get_linear_noise_func
from utils.image_utils import psnr
from utils.classification_utils import build_mask
from utils.viewpoint_utils import ViewpointLoader
from utils.visualization_utils import render_results
from utils.knn_utils import knn
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    if opt.tb_writer:
        tb_writer = prepare_output_and_logger(dataset)
    else:
        tb_writer = False
    gaussians = GaussianModel(dataset.sh_degree)
    deformGS = DeformGS()
    deformGS.train_setting(opt)

    deformArti = DeformModel(opt)
    revolute = Revolute()

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
    neighbor_weight = None
    # is_end_frame_with_grad = False
    # grad_counter = 0

    for iteration in range(start, end + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree

        if iteration == end:
            # mask, centers = build_mask(factors.detach().cpu().numpy())
            # mask = torch.tensor(
            #     mask, device="cuda", dtype=torch.float32, requires_grad=False
            # )
            # deform.save_weights(args.model_path, iteration)
            print(f"predicted articulated params:")
            print(
                f"axis: {revolute.axis.tolist()}\n pivot: {revolute.pivot.tolist()}\n theta: {revolute.theta}\n"
            )
            if end == opt.update_mask:
                # with open("grads.pkl", "wb") as f:
                #     grads["gaussians"] = gaussians
                #     pickle.dump(grads, f)
                #     print("grad data saved")

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

            # data = {
            #     "gaussians": gaussians,
            #     "mask": gaussians.get_movable_mask,
            #     "factor": factors,
            #     "deformModel": deform,
            #     "params": {
            #         "axis": revolute.axis.tolist(),
            #         "pivot": revolute.pivot.tolist(),
            #     },
            # }
            # with open("./stage_3.pkl", "wb") as f:
            #     pickle.dump(data, f)
            #     print("data saved")

            print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))

        if opt.only_train_single_frame == iteration:
            print("[Training]::step 1 is over, now training deformation net")
            print("[Training]::building knn trees")
            neighbor_sq_dist, neighbor_indices = knn(
                gaussians.get_xyz.detach().cpu().numpy(), 20
            )
            weight = np.exp(-200 * neighbor_sq_dist)
            dist = np.sqrt(neighbor_sq_dist)
            neighbor_weight = torch.tensor(weight).float().to(gaussians.get_xyz.device)
            neighbor_dist = torch.tensor(dist).float().to(gaussians.get_xyz.device)
            continue

        if opt.pretrain == iteration:
            print("[Training]::step 2 is over, now update the mask")
            with torch.no_grad():
                _, _, (d_xyz, d_rotations) = deformGS.step(gaussians)
                ndx = torch.norm(d_xyz, dim=-1).detach().cpu().numpy()
                ndr = torch.norm(d_rotations, dim=-1).detach().cpu().numpy()
                # mask = (ndr > 1e-2).to(
                #     gaussians.get_xyz.device, gaussians.get_xyz.dtype
                # )

                ndx = (ndx - min(ndx)) / (max(ndx) - min(ndx))
                ndr = (ndr - min(ndr)) / (max(ndr) - min(ndr))

                mask_r, _ = build_mask(ndr.reshape(-1, 1), "gmm", 20)
                mask_x = ndx > 1e-1
                mask_u = np.bitwise_and(mask_r, mask_x)

            data = {
                "gaussians": gaussians,
                "dx": d_xyz,
                "dr": d_rotations,
                "deformModel": deform,
                "params": {
                    "axis": revolute.axis.tolist(),
                    "pivot": revolute.pivot.tolist(),
                },
            }

            with open("./stage_2.pkl", "wb") as f:
                pickle.dump(data, f)
                print(" stage 2 data saved")

            gaussians.initialize_mask(mask_u)
            revolute.set_theta(math.pi / 2)  # TODO delete
            continue

        if iteration == opt.update_params:
            print("[Training]::step 3 is over, now update the articulated params")
            print(
                f"axis: {revolute.axis.tolist()}\n pivot: {revolute.pivot.tolist()}\n theta: {revolute.theta}\n"
            )
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
            continue

        if iteration < opt.only_train_single_frame:
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
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

        # ------------------- core: deformation ----------------------------

        # before pretrain, we do not train the articulated params
        if iteration > opt.only_train_single_frame:
            deform = deformGS if iteration < opt.pretrain else deformArti
            new_xyz, new_rotations, _ = deform.step(gaussians, revolute)

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
        if opt.only_train_single_frame < iteration < opt.update_mask:
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

        loss = loss_end + loss_start + loss_arap
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
            if opt.tb_writer and (iteration % opt.report_interval == 0):
                training_report(
                    tb_writer,
                    iteration,
                    iter_start.elapsed_time(iter_end),
                    loss_start,
                    loss_end,
                    loss_arap,
                    gaussians.get_xyz.shape[0],
                    gaussians.get_opacity,
                    revolute,
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
                        revolute,
                        tb_writer,
                        dataset.load2gpu_on_the_fly,
                        is_first_test,
                    )

                    if cur_psnr > best_psnr:
                        best_psnr = cur_psnr
                        best_iteration = iteration

            # --------------------- Densification --------------------------
            # Keep track of max radii in image-space for pruning
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

                # if iteration in saving_iterations:
                #     print("\n[ITER {}] Saving Gaussians".format(iteration))
                #     scene.save(iteration)
                #     deform.save_weights(args.model_path, iteration)

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
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None

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

            if iteration > opt.only_train_single_frame:
                # either deformGS or deformArti
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

                if opt.pretrain < iteration:
                    revolute.axis_pivot_optimizer.step()
                    revolute.axis_pivot_optimizer.zero_grad()
                    revolute.axis_pivot_scheduler.step()

                if opt.pretrain < iteration:
                    revolute.theta_optimizer.step()
                    revolute.theta_optimizer.zero_grad()
                    revolute.theta_scheduler.step()

            if (iteration < opt.only_train_single_frame) or (
                iteration > opt.update_params
            ):
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)


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
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
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
    u_loss,
    m_loss,
    rigid_loss,
    gs_num,
    opacity,
    revolute,
):
    if tb_writer:
        ml = m_loss.item() if isinstance(m_loss, torch.Tensor) else m_loss
        ul = u_loss.item() if isinstance(u_loss, torch.Tensor) else u_loss

        tb_writer.add_scalar("train_loss_patches/m_loss", ml, iteration)
        tb_writer.add_scalar("train_loss_patches/u_loss", ul, iteration)
        tb_writer.add_scalar("train_loss_patches/arap_loss", rigid_loss, iteration)

        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_histogram("scene/opacity_histogram", opacity, iteration)
        tb_writer.add_scalar("total_points", gs_num, iteration)

        for i in range(3):
            tb_writer.add_scalar(
                "revolute/axis_{}".format(i), revolute.axis[i], iteration
            )
            tb_writer.add_scalar(
                "revolute/pivot_{}".format(i), revolute.pivot[i], iteration
            )


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
        default=[6000, 7000, 8000, 9000, 11000, 12000, 14000, 16000, 20000, 24000],
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
