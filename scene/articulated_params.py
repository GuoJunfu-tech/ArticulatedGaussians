import torch
import math
from typing import Union
import json
import numpy as np


def float_to_torch(value: float) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float32, requires_grad=True, device="cuda")


class Prismatic:
    def __init__(self) -> None:
        self._type = "prismatic"
        self._axis = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True, device="cuda"
        )
        self._dist = torch.tensor(
            [0.0], dtype=torch.float32, requires_grad=True, device="cuda"
        )
        self.optimizer = torch.optim.Adam([self._axis, self._dist], lr=0.05, eps=2e-15)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.8
        )

    @property
    def type(self):
        return self._type

    @property
    def axis(self):
        return self._axis / torch.linalg.norm(self._axis)

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, dist: float):
        self._dist = float_to_torch(dist)

    def save_json(self, path):
        data = {
            "type": self._type,
            "axis": self._axis.detach().cpu().numpy().tolist(),
            "dist": self._dist.detach().cpu().numpy().item(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_params(self, data):
        if self._type != data["type"]:
            raise ValueError("[ERROR]::Wrong type of the articulated parameters!")

        self._axis = self.list_to_torch(data["axis"])
        # self._theta = self.list_to_torch(data["theta"])
        self._dist = self.list_to_torch(data["dist"])
        # self._pivot = self.list_to_torch(data["pivot"])

    @staticmethod
    def list_to_torch(values: list) -> torch.Tensor:
        return torch.tensor(
            values, dtype=torch.float32, requires_grad=True, device="cuda"
        )


class Revolute:
    def __init__(
        self,
        axis=None,
        theta=None,
        pivot=None,
    ) -> None:
        # Initial
        self._type = "revolute"
        self.set_params(axis, theta, pivot)

        self.optimizer = torch.optim.Adam(
            [self._axis, self._pivot, self._theta], lr=0.05, eps=2e-15
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.8
        )

    def set_params(self, axis=None, theta=None, pivot=None):
        if axis is None:
            axis = [0, 1.0, 0]
        self._axis = torch.tensor(
            axis,
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        if theta is None:
            theta = 0.0
        self._theta = float_to_torch(theta)
        if pivot is None:
            pivot = [0.0, 0.0, 0.0]
        self._pivot = torch.tensor(
            pivot,
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        # self._axis.to(device="cuda")
        # self._pivot.to(device="cuda")
        # self._theta.to(device="cuda")

    @property
    def type(self):
        return self._type

    def reset_param_optimizer(self):
        self.axis_pivot_optimizer = torch.optim.Adam(
            [self._axis, self._pivot], lr=0.05, eps=2e-15
        )
        self.axis_pivot_scheduler = torch.optim.lr_scheduler.StepLR(
            self.axis_pivot_optimizer, step_size=100, gamma=0.8
        )

    @property
    def theta(self):
        return self._theta  # TODO check if here need to add tanh

    @theta.setter
    def theta(self, theta: float):
        self._theta = float_to_torch(theta)

    @staticmethod
    def list_to_torch(values: list) -> torch.Tensor:
        return torch.tensor(
            values, dtype=torch.float32, requires_grad=True, device="cuda"
        )

    @property
    def axis(self):
        return self._axis / torch.linalg.norm(self._axis)

    @axis.setter
    def axis(self, axis: Union[torch.Tensor, list]):
        if isinstance(axis, torch.Tensor):
            if axis.device != self._axis.device:
                axis = axis.to(self._axis.device)
        elif isinstance(axis, list):
            axis = self.list_to_torch(axis)
        else:
            raise ValueError("[ERROR]::Wrong type of the axis!")

        self._axis = axis

    @property
    def pivot(self):
        return self._pivot

    @pivot.setter
    def pivot(self, pivot: Union[torch.Tensor, list]):
        if isinstance(pivot, torch.Tensor):
            if pivot.device != self._axis.device:
                pivot = pivot.to(self._axis.device)
        elif isinstance(pivot, list):
            pivot = self.list_to_torch(pivot)
        else:
            raise ValueError("[ERROR]::Wrong type of the axis!")

        self._pivot = pivot

    def save_json(self, path):
        data = {
            "type": self._type,
            "axis": self._axis.detach().cpu().tolist(),
            "pivot": self._pivot.detach().cpu().tolist(),
            "theta": self._theta.detach().cpu().item(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def theta_normalization(self):
        normalized_angle = self._theta % (2 * math.pi)  # 将角度对 2*pi 取余数
        if normalized_angle > math.pi:
            normalized_angle -= 2 * math.pi  # 如果大于 pi，减去 2*pi
        elif normalized_angle < -math.pi:
            normalized_angle += 2 * math.pi  # 如果小于 -pi，加上 2*pi
        self._theta = normalized_angle

    def load_params(self, data):
        if self._type != data["type"]:
            raise ValueError("[ERROR]::Wrong type of the articulated parameters!")

        self._axis = self.list_to_torch(data["axis"])
        self._theta = self.list_to_torch(data["theta"])
        self._pivot = self.list_to_torch(data["pivot"])
