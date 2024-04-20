import numpy as np
import torch
import math


class ArticulatedOperator:
    # ------------ Prismatic --------------------------------
    def get_new_location_prismatic(self, x, axis, dist):
        assert abs(torch.linalg.norm(axis) - 1) < 1e-2
        N = x.shape[0]
        axis = axis.unsqueeze(0)
        dist = dist.view(N, 1)
        return x + axis * dist

    # ------------ Revolute ---------------------------------
    @staticmethod
    def get_rotation_matrix(axis: torch.Tensor, theta: torch.Tensor):
        axis = axis / torch.linalg.norm(axis)  # normalize
        kx, ky, kz = axis

        if theta.numel() == 1:
            cos, sin = torch.cos(theta), torch.sin(theta)
            one = torch.tensor(1.0, dtype=axis.dtype, device=axis.device).expand_as(cos)

            R = torch.zeros(3, 3, device=axis.device)

            R[0, 0] = cos.squeeze() + (kx**2) * (one - cos).squeeze()
            R[0, 1] = kx * ky * (one - cos).squeeze() - kz * sin.squeeze()
            R[0, 2] = kx * kz * (one - cos).squeeze() + ky * sin.squeeze()
            R[1, 0] = kx * ky * (one - cos).squeeze() + kz * sin.squeeze()
            R[1, 1] = cos.squeeze() + (ky**2) * (one - cos).squeeze()
            R[1, 2] = ky * kz * (one - cos).squeeze() - kx * sin.squeeze()
            R[2, 0] = kx * kz * (one - cos).squeeze() - ky * sin.squeeze()
            R[2, 1] = ky * kz * (one - cos).squeeze() + kx * sin.squeeze()
            R[2, 2] = cos.squeeze() + (kz**2) * (one - cos).squeeze()
        else:
            cos = torch.cos(theta).squeeze()  # Nx1 -> N
            sin = torch.sin(theta).squeeze()  # Nx1 -> N

            one = torch.ones_like(cos)  # N
            # one = torch.tensor(1.0, dtype=axis.dtype, device=axis.device).expand_as(cos)

            # Initialize the rotation matrices batch
            R = torch.zeros(theta.size(0), 3, 3, device=axis.device, dtype=axis.dtype)

            # Fill in each matrix
            R[:, 0, 0] = cos + (kx**2) * (one - cos)
            R[:, 0, 1] = kx * ky * (one - cos) - kz * sin
            R[:, 0, 2] = kx * kz * (one - cos) + ky * sin
            R[:, 1, 0] = kx * ky * (one - cos) + kz * sin
            R[:, 1, 1] = cos + (ky**2) * (one - cos)
            R[:, 1, 2] = ky * kz * (one - cos) - kx * sin
            R[:, 2, 0] = kx * kz * (one - cos) - ky * sin
            R[:, 2, 1] = ky * kz * (one - cos) + kx * sin
            R[:, 2, 2] = cos + (kz**2) * (one - cos)

        return R

    def get_new_location_revolute(self, x, axis, pivot_point, theta):
        """
        Computes the new location of a point after rotation in a revolute joint.

        Args:
        x: Point coordinates as a 3D vector (numpy array).
        u: Revolute axis vector as a unit vector (numpy array).
        q: Pivot point coordinates as a 3D vector (numpy array).
        theta: Rotation angle about the revolute axis.

        Returns:
        A numpy array representing the new location of the point.
        """
        # x = np.asarray(x)
        # p_local = torch.transpose(x - pivot_point, 0, 1)
        assert abs(torch.linalg.norm(axis) - 1) < 1e-2
        x_local = x - pivot_point
        R = self.get_rotation_matrix(axis, theta)
        if theta.numel() == 1:
            R = R.unsqueeze(0)
            x_expand = x_local.unsqueeze(-1)
            # p_local_rotated = R.dot(p_local)
            x_local_rotated = torch.matmul(R, x_expand).squeeze(-1)
            # x_new = torch.transpose(p_local_rotated, 0, 1) + pivot_point
            x_new = (x_local_rotated + pivot_point).squeeze()
        else:
            x_local_rotated = torch.matmul(R, x_local.unsqueeze(-1)).squeeze(-1)
            x_new = x_local_rotated + pivot_point

        return x_new

    def get_new_quaternion(
        self, q: torch.Tensor, axis: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        assert abs(torch.linalg.norm(axis) - 1) < 1e-2
        if theta.numel() == 1:
            q_rot = torch.tensor(
                [
                    torch.cos(theta / 2),
                    axis[0] * torch.sin(theta / 2),
                    axis[1] * torch.sin(theta / 2),
                    axis[2] * torch.sin(theta / 2),
                ]
            )
            q_rot = q_rot.unsqueeze(0).expand(q.size(0), -1)
        else:
            # Since theta is Nx1, these operations are broadcasted to each theta
            q_rot = torch.stack(
                [
                    torch.cos(theta / 2).squeeze(-1),
                    axis[0]
                    * torch.sin(theta / 2).squeeze(),  # Sine components need squeezing
                    axis[1] * torch.sin(theta / 2).squeeze(),
                    axis[2] * torch.sin(theta / 2).squeeze(),
                ],
                dim=-1,
            )  # Stack along dim=-1 to form the quaternion

        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = q_rot.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        # Form the new quaternion and normalize it
        q_new = torch.stack([w, x, y, z], dim=-1)
        q_new = q_new / torch.linalg.norm(q_new, dim=1, keepdim=True)

        return q_new


if __name__ == "__main__":
    axis = np.array([0, 0, 1])
    pivot = np.array([0, 0, 0])
    theta = math.pi / 2

    # test quaternion_update function
    N = 5  # Batch size
    q = torch.rand(
        N, 4
    )  # Batch of quaternions, assuming the first component is the scalar part
    axis = torch.rand(N, 3)  # Batch of rotation axes
    axis = axis / torch.norm(axis, dim=1, keepdim=True)
    theta = 0.5  # Rotation angle in radians

    rotate = ArticulatedOperator()
    q_updated = rotate.get_new_quaternion(q, axis, theta)

    print(q_updated)
