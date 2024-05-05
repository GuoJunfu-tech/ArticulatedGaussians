import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, MovableNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.deform_utils import ArticulatedOperator
import math


class DeformModel:
    def __init__(self) -> None:
        # self.optimizer = None
        self.movable_network = MovableNetwork().cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.deform_operator = ArticulatedOperator()
        # self.train_setting(training_args)

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

    def step(self, gaussians, arti_param, keep_gs_grad=True):
        return self.deform(
            gaussians.get_xyz,
            gaussians.get_rotation,
            arti_param,
            gaussians.get_movable_mask,
            keep_gs_grad=keep_gs_grad,
        )

    def deform(
        self,
        xyz,
        rotation,
        arti_param,
        factor=None,
        keep_gs_grad=False,
    ):
        if factor is not None:
            assert factor.shape[0] == xyz.shape[0]
            movable_factor = factor
        else:
            assert False
            # movable_factor = self.movable_network(xyz)

        if not keep_gs_grad:
            xyz = xyz.detach()
            quaternions = rotation.detach()
        else:
            quaternions = rotation

        if arti_param.type == "revolute":
            axis = arti_param.axis
            pivot = arti_param.pivot
            theta = arti_param.theta
            if theta is None:
                theta = movable_factor * math.pi  # TODO check why?
            else:
                theta = theta * movable_factor

            axis = axis / torch.linalg.norm(axis)
            new_xyz = self.deform_operator.get_new_location_revolute(
                xyz, axis, pivot, theta
            )
            new_rotations = self.deform_operator.get_new_quaternion(
                quaternions, axis, theta
            )  # return the intermediate quaternion depend on movable_factor

        elif arti_param.type == "prismatic":
            axis = arti_param.axis
            dist = arti_param.dist
            dist = dist * movable_factor
            new_xyz = self.deform_operator.get_new_location_prismatic(xyz, axis, dist)
            new_rotations = quaternions
        else:
            raise ValueError("unrecognized articulated params!")

        # if is_render:
        #     movable_factor = (movable_factor > 1e-3).float()

        # new_xyz = xyz + movable_factor * (moved_xyz - xyz)
        return new_xyz, new_rotations, movable_factor

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
