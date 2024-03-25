import torch
import math


class Revolute:
    def __init__(self) -> None:
        # Initial
        self._axis = torch.tensor(
            [0, 1, 0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self._pivot = torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self._theta = torch.tensor(
            [math.pi / 2],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self.optimizer = torch.optim.Adam(
            [self._axis, self._pivot], lr=0.01, eps=2e-15
        )  # Remark! no theta!!

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.99
        )

    @property
    def get_axis(self):
        return self._axis

    @get_axis.setter
    def set_axis(self, axis: torch.Tensor):
        self._axis = axis

    @property
    def get_theta(self):
        return torch.tanh(self._theta) * math.pi

    @get_theta.setter
    def set_theta(self, theta: torch.Tensor):
        if theta.size() != torch.Size([1, 1]):
            raise ValueError

        self._theta = theta

    @property
    def get_pivot(self):
        return self._pivot

    @get_pivot.setter
    def set_pivot(self, pivot: torch.Tensor):
        self._pivot = pivot
