import torch
import math
from typing import Union


class Revolute:
    def __init__(self) -> None:
        # Initial
        self._axis = torch.tensor(
            [0, 1, 0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        # gt: [0.73, 0.175, -0.152],
        self._pivot = torch.tensor(
            [0.0, 0.0, 0.0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self.axis_pivot_optimizer = torch.optim.Adam(
            [self._axis, self._pivot], lr=0.05, eps=2e-15
        )
        self.axis_pivot_scheduler = torch.optim.lr_scheduler.StepLR(
            self.axis_pivot_optimizer, step_size=100, gamma=0.8
        )

        self._theta = None

    def set_theta(self, theta):
        self._theta = self.float_to_torch(theta)
        self.theta_optimizer = torch.optim.Adam([self._theta], lr=0.005, eps=2e-15)
        self.theta_scheduler = torch.optim.lr_scheduler.StepLR(
            self.theta_optimizer, step_size=100, gamma=0.9
        )

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
        self._theta = self.float_to_torch(theta)

    @staticmethod
    def float_to_torch(value: float) -> torch.Tensor:
        return torch.tensor(
            [value], dtype=torch.float32, requires_grad=True, device="cuda"
        )

    @staticmethod
    def list_to_torch(values: list) -> torch.Tensor:
        return torch.tensor(
            values, dtype=torch.float32, requires_grad=True, device="cuda"
        )

    @property
    def axis(self):
        return self._axis

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
