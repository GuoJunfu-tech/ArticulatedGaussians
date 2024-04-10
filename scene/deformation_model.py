import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, MovableNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.rotation_utils import RotationOperator
import math


class DeformModel:
    def __init__(self) -> None:
        self.optimizer = None
        self.movable_network = MovableNetwork().cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.deform_operator = RotationOperator()

    def step(self, gaussians, revolute, keep_gs_grad=False):
        return self.deform(
            gaussians.get_xyz,
            gaussians.get_rotation,
            revolute.axis,
            revolute.pivot,
            revolute.theta,
            gaussians.get_movable_mask,
            keep_gs_grad=keep_gs_grad,
        )

    def deform(
        self,
        xyz,
        rotation,
        axis,
        pivot,
        theta,
        factor=None,
        keep_gs_grad=False,
    ):
        if not keep_gs_grad:
            xyz = xyz.detach()
            quaternions = rotation.detach()
        else:
            quaternions = rotation

        if factor is not None:
            assert factor.shape[0] == xyz.shape[0]
            movable_factor = factor
        else:
            movable_factor = self.movable_network(xyz)

        if theta is None:
            theta = movable_factor * math.pi
        else:
            theta = theta * movable_factor

        # if is_render:
        #     movable_factor = (movable_factor > 1e-3).float()

        axis = axis / torch.linalg.norm(axis)
        new_xyz = self.deform_operator.get_new_location(xyz, axis, pivot, theta)

        moved_quaternion = self.deform_operator.get_new_quaternion(
            quaternions, axis, theta
        )  # return the intermediate quaternion depend on movable_factor

        # new_xyz = xyz + movable_factor * (moved_xyz - xyz)
        new_rotations = moved_quaternion
        return new_xyz, new_rotations, movable_factor

    def train_setting(self, training_args):
        l = [
            {
                "params": list(self.movable_network.parameters()),
                "lr": training_args.movable_lr_init,
                "name": "movable",
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=2e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.movable_lr_init,
            lr_final=training_args.movable_lr_final,
            lr_delay_mult=training_args.movable_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps,
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(
            model_path, "movable/iteration_{}".format(iteration)
        )
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(
            self.deform.state_dict(), os.path.join(out_weights_path, "movable.pth")
        )

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "movable"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(
            model_path, "movable/iteration_{}/movable.pth".format(loaded_iter)
        )
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
