import random
import math
import torch

from utils.deform_utils import ArticulatedOperator


def test_rotation_operator():
    # theta = random.rand() * math.pi * 2

    rotate = ArticulatedOperator()
    for test_id in range(10):
        print(f"[testRotationOperator]::Testing with {test_id} intervals")
        # setup
        bn = random.randint(1, 100000)  # batch size
        axis = torch.rand(3)
        axis = axis / torch.norm(axis)
        pivot = torch.rand(3)
        xyz = torch.rand(bn, 3)
        q = torch.rand(bn, 4)
        q = q / torch.norm(q, dim=1, keepdim=True)

        intervals = random.randint(2, 10)

        rotate_interval = 2 * math.pi / intervals
        theta = torch.tensor(rotate_interval, dtype=torch.float32)

        theta_matrix = theta.expand(bn, 1)
        q_origin = q.clone()
        q_matrix = q.clone()
        xyz_origin = xyz.clone()
        xyz_matrix = xyz.clone()
        for i in range(intervals):
            # print(f"xyz: {xyz}")
            # print(f"xyz_matrix: {xyz_matrix}")
            xyz = rotate.get_new_location_revolute(xyz, axis, pivot, theta)
            q = rotate.get_new_quaternion(q, axis, theta)

            xyz_matrix = rotate.get_new_location_revolute(
                xyz_matrix, axis, pivot, theta_matrix
            )
            q_matrix = rotate.get_new_quaternion(q_matrix, axis, theta_matrix)

        # test
        tolerance = 1e-5
        diff_q_p = (q_origin - q).max().item()
        diff_q_n = (q_origin + q).max().item()
        diff_xyz = (xyz_origin - xyz).max().item()

        diff_q_p_theta_is_matrix = (q_origin - q_matrix).max().item()
        diff_q_n_theta_is_matrix = (q_origin + q_matrix).max().item()
        diff_xyz_theta_is_matrix = (xyz_origin - xyz_matrix).max().item()

        assert diff_q_p <= tolerance or diff_q_n <= tolerance
        assert diff_xyz <= tolerance

        assert (
            diff_q_p_theta_is_matrix <= tolerance
            or diff_q_n_theta_is_matrix <= tolerance
        )
        assert diff_xyz_theta_is_matrix <= tolerance


if __name__ == "__main__":
    # Assuming q_rot is a 4-element tensor representing a quaternion
    q_rot = torch.randn(4)  # Example quaternion

    # Assuming q is a Nx4 tensor representing a batch of quaternions
    N = 5
    q = torch.randn(N, 4)  # Batch of quaternions

    # Correctly expand q_rot to match the batch size of q
    print(q.size())
    q_rot_expanded = q_rot.unsqueeze(0).expand(q.size(0), -1)

    print(q_rot_expanded.shape)  # Should print torch.Size([N, 4])
