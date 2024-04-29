import os
from PIL import Image
import numpy as np
import torch

from gaussian_renderer import render


def render_results(
    viewpoint_cams,
    gaussians,
    deformModel,
    arti_params,
    factors,
    pipe,
    background,
    type="gif",
    note=None,
):
    if arti_params.type == "prismatic":
        dist = arti_params.dist.detach().cpu().float()

    elif arti_params.type == "revolute":
        theta = arti_params.theta.detach().cpu().float()

    with torch.no_grad():
        for cid, cam in enumerate(viewpoint_cams):
            if cid == 20:
                break

            if type == "gif":
                k = 20

                images = []
                for i in range(k):
                    if arti_params.type == "prismatic":
                        arti_params.dist = i * dist / k

                    if arti_params.type == "revolute":
                        arti_params.theta = i * theta / k
                    new_xyz, new_rotations, _ = deformModel.step(
                        gaussians, arti_params, keep_gs_grad=False
                    )
                    render_pkg_re = render(
                        cam,
                        gaussians,
                        pipe,
                        background,
                        new_xyz,
                        new_rotations,
                    )
                    image = render_pkg_re["render"]
                    image_np = image.detach().cpu().numpy().transpose((1, 2, 0))

                    img = Image.fromarray(np.uint8(image_np * 255), "RGB")
                    images.append(img)

                save_path = os.path.join(
                    os.getcwd(), f"rendered_img/dynamic_{cid}_{note}.gif"
                )
                # if not os.path.exists(save_path):
                #     raise ValueError(f"Could not find path {save_path}")
                print(f"saving results {save_path}")
                images[0].save(
                    save_path,
                    save_all=True,
                    append_images=images[1:],
                    optimize=False,
                    duration=100,
                    loop=0,
                )
            elif type == "img":
                raise ValueError("Not implemented")  # TODO implement this
                # revoluteParams.set_theta(theta)
                save_path = os.path.join(os.getcwd(), f"rendered_img/static_{cid}.png")
                new_xyz, new_rotations, factors = deformModel.deform(
                    gaussians.get_xyz,
                    gaussians.get_rotation,
                    revoluteParams.axis,
                    revoluteParams.pivot,
                    theta=None,
                    factor=None,
                )
                render_pkg_re = render(
                    cam,
                    gaussians,
                    pipe,
                    background,
                    new_xyz,
                    new_rotations,
                    0.0,
                    False,
                )
                image = render_pkg_re["render"]
                image_np = image.detach().cpu().numpy().transpose((1, 2, 0))

                gt_image = cam.original_image
                gt_image_np = gt_image.detach().cpu().numpy().transpose((1, 2, 0))
                gt_img = Image.fromarray(np.uint8(gt_image_np * 255), "RGB")
                gt_img.save(f"rendered_img/gt_img_{id}.png", "PNG")

                img = Image.fromarray(np.uint8(image_np * 255), "RGB")
                img.save(save_path, "PNG")
            elif type == "raw":
                # revoluteParams.set_theta(theta)
                save_path = os.path.join(os.getcwd(), f"rendered_img/static_{cid}.png")
                new_xyz, new_rotations = None, None
                render_pkg_re = render(
                    cam,
                    gaussians,
                    pipe,
                    background,
                    new_xyz,
                    new_rotations,
                    0.0,
                    False,
                )
                image = render_pkg_re["render"]
                image_np = image.detach().cpu().numpy().transpose((1, 2, 0))

                img = Image.fromarray(np.uint8(image_np * 255), "RGB")
                img.save(save_path, "PNG")
            else:
                raise ValueError("Type not found")

        return new_xyz, new_rotations, factors

        # img.save(f"./rendered_img/{id}.png", format="PNG")
