import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3

import numpy as np
from math import ceil


class PointNet(nn.Module):
    def __init__(self, point_num):
        super(PointNet, self).__init__()
        self.W = 64
        self.point_num = point_num
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.W)
        self.gaussian_warp = nn.Linear(self.W, 3)
        self.gaussian_rotation = nn.Linear(self.W, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, gaussians):
        x = gaussians.get_xyz.detach()  # Nx3
        # h = x.view(1, -1, 3)
        h = x.unsqueeze(0)
        B, N, C = h.shape
        h = torch.permute(h, dims=(0, 2, 1))

        # h = h.view(B, C, N)
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        # h = h.transpose(1, 2).contiguous()

        h = F.max_pool1d(h, kernel_size=1).squeeze(2)
        h = torch.permute(h, dims=(0, 2, 1))
        h = h.squeeze(0)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        d_xyz = self.gaussian_warp(h)
        d_rotation = self.gaussian_rotation(h)

        return (
            x + d_xyz,
            gaussians.get_rotation + d_rotation,
            (d_xyz, d_rotation),
        )


class PointEmbed(nn.Module):
    def __init__(self, data_dim=3, hidden_dim=48, dim=128):
        super().__init__()

        cos_sin_data_dim = 2 * data_dim

        self.embedding_dim = ceil(hidden_dim // cos_sin_data_dim) * cos_sin_data_dim

        e = (
            torch.pow(2, torch.arange(self.embedding_dim // cos_sin_data_dim)).float()
            * np.pi
        )
        basis = torch.zeros(
            [data_dim, data_dim * self.embedding_dim // cos_sin_data_dim]
        )
        for i in range(data_dim):
            basis[i, e.shape[0] * i : e.shape[0] * (i + 1)] = e

        # data_dim x (embedding_dim // cos_sin_data_dim)
        self.register_buffer("basis", basis)

        self.mlp = nn.Linear(self.embedding_dim + data_dim, dim)

        self.dim = dim
        return

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x cos_sin_data_dim
        embed = self.embed(input.unsqueeze(0), self.basis)

        embed = torch.cat([embed, input.unsqueeze(0)], dim=2)

        # B x N x C
        embed = self.mlp(embed)
        return embed.squeeze()


class DeformNetwork(nn.Module):  # FIXME: input x to forward(), not gaussian!
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=59,
        multires=10,
        is_blender=True,
    ):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        # self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        xyz_input_ch = 64
        self.embbeder = PointEmbed(3, 48, 64)
        # self.embed_fn, xyz_input_ch = embbeder.forward, 63
        self.input_ch = xyz_input_ch

        if is_blender:
            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch, W)]
                + [
                    (
                        nn.Linear(W, W)
                        if i not in self.skips
                        else nn.Linear(W + xyz_input_ch, W)
                    )
                    for i in range(D - 1)
                ]
            )

        else:
            print("[ERROR]::Should not be here")

            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)]
                + [
                    (
                        nn.Linear(W, W)
                        if i not in self.skips
                        else nn.Linear(W + self.input_ch, W)
                    )
                    for i in range(D - 1)
                ]
            )

        self.is_blender = is_blender

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        # self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, gaussians):
        x = gaussians.get_xyz.detach()
        # x_emb = self.embed_fn(x)
        x_emb = self.embbeder.forward(x)
        h = x_emb
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        d_rotation = self.gaussian_rotation(h)

        return (
            x + d_xyz,
            gaussians.get_rotation + d_rotation,
            (d_xyz, d_rotation),
        )


# ---------- not used anymore ------------------------------------------------------
class MovableNetwork(nn.Module):
    def __init__(
        self,
        D=4,
        W=32,
        input_ch=3,
        output_ch=59,
        multires=10,
        # is_6dof=False,
    ):
        super(MovableNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6
        # self.skips = [D // 2]

        # self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        xyz_input_ch = None
        self.input_ch = xyz_input_ch
        self.movable_warp = nn.Linear(W, 1)

        # self.linear = nn.ModuleList(
        #     [nn.Linear(xyz_input_ch, W)] + [(nn.Linear(W, W)) for i in range(D - 1)]
        # )

    def forward(self, x):
        x_emb = self.embed_fn(x)
        # h = torch.cat([x_emb, t_emb], dim=-1)
        h = x_emb
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = nn.ReLU(h)

        h = self.movable_warp(h)
        # is_movable = torch.tanh(
        #     h * 0.1
        # )  # TODO maybe try more activate functions with higher gradient
        # is_movable = (1 + is_movable) / 2.0
        is_movable = torch.tanh(h)

        return is_movable
