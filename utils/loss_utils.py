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

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import exp
# from scipy.spatial import KDTree

from utils.knn_utils import knn


def l1_loss(network_output, gt):
    return F.l1_loss(network_output, gt)
    # return torch.abs((network_output - gt)).mean()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5))
        + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5))
    )


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def chamfer_distance_loss(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    assert p1.shape[0] != 0
    assert p2.shape[0] != 0

    p1_square = p1.pow(2).sum(dim=1, keepdim=True)  # [N, 1]
    p2_square = p2.pow(2).sum(dim=1, keepdim=True)  # [M, 1]
    dist = p1_square + p2_square.transpose(0, 1) - 2 * p1 @ p2.transpose(0, 1)  # [N, M]

    # Get min dist for each element in p1 to p2
    min_dist_p1_to_p2, _ = dist.min(dim=1)
    min_dist_p2_to_p1, _ = dist.min(dim=0)

    # Mean distance
    return min_dist_p1_to_p2.mean() + min_dist_p2_to_p1.mean()


def opacity_loss(radii, gaussians, factor=0.1):
    visibility_filter = radii > 0
    vis_opacities = gaussians.get_opacity[visibility_filter]
    opacity_loss = (
        factor
        * (
            -vis_opacities * torch.log(vis_opacities + 1e-10)
            - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
        ).mean()
    )
    return opacity_loss


def ll1_ssim_loss(image, gt_image, factor):
    Ll1 = l1_loss(image, gt_image)
    # ssim_loss = ssim(image, gt_image)
    loss = (1.0 - factor) * Ll1 + factor * (1.0 - ssim(image, gt_image))
    return loss


def arap_loss(new_xyz, neighbor_indices, neighbor_dist, neighbor_weight):
    neighbor_indices = torch.tensor(neighbor_indices, requires_grad=False).long()
    neighbor_pts = new_xyz[neighbor_indices]
    curr_offset = neighbor_pts - new_xyz[:, None]
    curr_offset_mag = torch.sqrt((curr_offset.pow(2)).sum(-1) + 1e-20)
    return torch.sqrt(
        (curr_offset_mag - neighbor_dist).pow(2) * neighbor_weight + 1e-20
    ).mean()
