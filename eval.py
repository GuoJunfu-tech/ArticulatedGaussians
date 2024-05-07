"""
Most codes are copied from 3D GS evaluation
"""
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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim

# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

from utils.deform_utils import ArticulatedOperator


def axis_metrics(motion: dict, gt: dict):
    # pred axis
    pred_axis_d = motion["axis"]
    pred_axis_o = motion["pivot"]

    # gt axis
    gt_axis_d = gt["axis_d"]
    gt_axis_o = gt["axis_o"]

    # angular difference between two vectors
    cos_theta = torch.dot(pred_axis_d, gt_axis_d) / (
        torch.norm(pred_axis_d) * torch.norm(gt_axis_d)
    )
    ang_err = torch.rad2deg(torch.acos(torch.abs(cos_theta)))
    # positonal difference between two axis lines
    w = gt_axis_o - pred_axis_o
    cross = torch.cross(pred_axis_d, gt_axis_d)
    if (cross == torch.zeros(3)).sum().item() == 3:
        pos_err = torch.tensor(0)
    else:
        pos_err = torch.abs(torch.sum(w * cross)) / torch.norm(cross)
    return ang_err, pos_err


def geodesic_distance(pred_R, gt_R):
    """
    q is the output from the network (rotation from t=0.5 to t=1)
    gt_R is the GT rotation from t=0 to t=1
    """
    pred_R, gt_R = pred_R.cpu(), gt_R.cpu()
    R_diff = torch.matmul(pred_R, gt_R.T)
    cos_angle = torch.clip((torch.trace(R_diff) - 1.0) * 0.5, min=-1.0, max=1.0)
    angle = torch.rad2deg(torch.arccos(cos_angle))
    return angle


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    renders_dir = Path(renders_dir)
    gt_dir = Path(gt_dir)
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def load_gt_info(path):
    with open(path, mode="r") as f:
        meta = json.load(f)
        trans_info = meta["trans_info"]
        f.close()

    motion_type = trans_info["type"]

    axis_o = torch.tensor(trans_info["axis"]["o"]).float()
    axis_d = torch.tensor(trans_info["axis"]["d"]).float()

    axis_P0 = axis_o.unsqueeze(0)
    axis_P1 = (axis_o + axis_d).unsqueeze(0)
    vis_axis = torch.cat([axis_P0, axis_P1], dim=0)

    if motion_type == "rotate":
        angle = torch.tensor(
            [(trans_info["rotate"]["r"] - trans_info["rotate"]["l"])]
        ).float()
        theta = -1 * torch.deg2rad(angle)  # FIXME opposite from ours
        R = ArticulatedOperator.get_rotation_matrix(axis_d, theta)
        print(theta)
        return {
            "theta": theta,
            "type": "rotate",
            "R": R,
            "axis_o": axis_o,
            "axis_d": axis_d,
            "vis_axis": vis_axis,
        }
    elif motion_type == "translate":
        dist = torch.tensor(
            [(trans_info["translate"]["r"] - trans_info["translate"]["l"])]
        ).float()
        return {
            "type": "translate",
            "dist": dist,
            "axis_o": axis_o,
            "axis_d": axis_d,
            "vis_axis": vis_axis,
        }


def vis_quality_evaluate(scene_path):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for state in ["start", "end"]:
        gt_path = os.path.join(scene_path, "gt_imgs", state)
        # gt_path = Path(scene_path / "gt_imgs" / state)
        render_path = os.path.join(scene_path, "rendered_imgs", state)
        # render_path = Path(scene_path / "rendered_imgs" / state)
        # scene_dir = os.path.join(model_path, state)

        print("Scene:", render_path)
        full_dict[state] = {}
        per_view_dict[state] = {}
        full_dict_polytopeonly[state] = {}
        per_view_dict_polytopeonly[state] = {}

        renders, gts, image_names = readImages(render_path, gt_path)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")

        full_dict[state].update(
            {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            }
        )
        per_view_dict[state].update(
            {
                "SSIM": {
                    name: ssim
                    for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                },
                "PSNR": {
                    name: psnr
                    for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                },
                "LPIPS": {
                    name: lp
                    for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
                },
            }
        )

        # TODO save the results
        # with open(scene_path + "/results.json", "w") as fp:
        #     json.dump(full_dict[scene_dir], fp, indent=True)
        # with open(scene_path + "/per_view.json", "w") as fp:
        #     json.dump(per_view_dict[scene_dir], fp, indent=True)


def geo_quality_evaluate(output_root):
    pass


def motion_evaluate(output_root, gt_path):
    motion_path = os.path.join(output_root, "motion.json")
    if not os.path.exists(motion_path):
        raise ValueError(f"Path {motion_path} does not exist!")
    motion_gt_path = os.path.join(gt_path, "textured_objs", "trans.json")
    if not os.path.exists(motion_gt_path):
        raise ValueError(f"Path {motion_gt_path} does not exist!")

    gt_info = load_gt_info(motion_gt_path)
    with open(motion_path, "r") as f:
        motion = json.load(f)

    tensor_motion = {}
    for key, value in motion.items():
        if not isinstance(value, str):
            tensor_motion[key] = torch.tensor(value)
        else:
            tensor_motion[key] = value

    if gt_info["type"] == "rotate":
        R = ArticulatedOperator.get_rotation_matrix(
            tensor_motion["axis"], tensor_motion["theta"]
        )
        gt_R = ArticulatedOperator.get_rotation_matrix(
            gt_info["axis_d"], gt_info["theta"]
        )
        geo_dist = geodesic_distance(R, gt_R)
        ang_err, pos_err = axis_metrics(tensor_motion, gt_info)
        return {
            "geo_dist": geo_dist.item(),
            "ang_err": ang_err.item(),
            "pos_err": pos_err.item(),
        }

    elif gt_info["type"] == "translate":
        pass
    else:
        raise ValueError("Unknown motion type!")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_path", "-m", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument(
        "--gt_path", "-g", required=True, nargs="+", type=str, default=[]
    )
    args = parser.parse_args()
    output_root = args.model_path[0]
    # scene = os.path.join(output_root, "point_cloud/iteration_40000/point_cloud.ply")
    if not os.path.exists(output_root):
        print(f"Scene address {output_root} not found!")
        exit()

    # vis_quality_evaluate(output_root)
    motion_metrics = motion_evaluate(output_root, args.gt_path[0])
    print(motion_metrics)
