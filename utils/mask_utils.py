import torch
from utils.loss_utils import l1_loss


def get_mask(cams, gaussians, renderFunc, renderArgs):
    loss = torch.empty(0)
    for viewpoint in cams:
        image = renderFunc(
            viewpoint, gaussians, *renderArgs, gaussians.get_xyz, gaussians.get_rotation
        )["render"]

        gt_image = viewpoint.original_image.to("cuda")
        loss += l1_loss(gt_image, image)
