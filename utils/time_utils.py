import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3

import numpy as np
from math import ceil


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


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": i,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(
        self,
        D=8,
        W=128,
        input_ch=3,
        output_ch=59,
        multires=10,
        is_blender=True,
        is_6dof=False,
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
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
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

            if self.is_6dof:
                w = self.branch_w(h)
                v = self.branch_v(h)
                theta = torch.norm(w, dim=-1, keepdim=True)
                w = w / theta + 1e-5
                v = v / theta + 1e-5
                screw_axis = torch.cat([w, v], dim=-1)
                d_xyz = exp_se3(screw_axis, theta)

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

        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch
        self.movable_warp = nn.Linear(W, 1)

        self.linear = nn.ModuleList(
            [nn.Linear(xyz_input_ch, W)] + [(nn.Linear(W, W)) for i in range(D - 1)]
        )

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
