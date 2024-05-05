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

        try:
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

            with open(scene_path + "/results.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_path + "/per_view.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_path)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_path", "-m", required=True, nargs="+", type=str, default=[]
    )
    # parser.add_argument(
    #     "--output_path", "-o", required=True, nargs="+", type=str, default=[]
    # )
    args = parser.parse_args()
    output_root = args.model_path[0]
    # scene = os.path.join(output_root, "point_cloud/iteration_40000/point_cloud.ply")
    if not os.path.exists(output_root):
        print(f"Scene address {output_root} not found!")
        exit()

    vis_quality_evaluate(output_root)
