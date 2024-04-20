import torch
import math
from typing import Union


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


class Revolute:
    def __init__(self) -> None:
        # Initial
        self._type = "revolute"
        self._axis = torch.tensor(
            [0, 1, 0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self._theta = float_to_torch(math.pi / 2)
        # gt: [0.73, 0.175, -0.152],
        self._pivot = torch.tensor(
            [0.0, 0.0, 0.0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self.optimizer = torch.optim.Adam(
            [self._axis, self._pivot, self._theta], lr=0.05, eps=2e-15
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.8
        )

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
