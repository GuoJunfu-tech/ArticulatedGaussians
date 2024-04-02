import os
from PIL import Image
import numpy as np
import torch

from gaussian_renderer import render


def render_results(
    viewpoint_cams,
    gaussians,
    deformModel,
    revoluteParams,
    factors,
    pipe,
    background,
    type="gif",
):
    with torch.no_grad():
        for id, cam in enumerate(viewpoint_cams):
            # if id == 4:
            #     exit()

            theta = (
                revoluteParams.theta.detach().cpu().float()
                if revoluteParams.theta
                else None
            )

            if type == "gif":
                k = 20
                interval = theta / 20

                images = []
                for i in range(k):
                    revoluteParams.set_theta(i * interval)
                    new_xyz, new_rotations, _ = deformModel.step(
                        gaussians,
                        revoluteParams,
                        factors,
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

                    img = Image.fromarray(np.uint8(image_np * 255), "RGB")
                    images.append(img)

                save_path = os.path.join(os.getcwd(), f"rendered_img/dynamic_{id}.gif")
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
                # revoluteParams.set_theta(theta)
                save_path = os.path.join(os.getcwd(), f"rendered_img/static_{id}.png")
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
            else:
                raise ValueError("Type not found")

            return new_xyz, new_rotations, factors

        # img.save(f"./rendered_img/{id}.png", format="PNG")
