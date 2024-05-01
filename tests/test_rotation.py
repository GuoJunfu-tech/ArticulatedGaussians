import random
import math
import torch
import unittest

from utils.deform_utils import ArticulatedOperator


class TestRotationOperator(unittest.TestCase):
    def test_rotate_a_round(self):
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

            intervals = random.randint(2, 40)

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

            self.assertTrue(diff_q_p < tolerance or diff_q_n < tolerance)
            self.assertLessEqual(diff_xyz, tolerance)

            self.assertTrue(
                diff_q_p_theta_is_matrix < tolerance
                or diff_q_n_theta_is_matrix < tolerance
            )
            self.assertLessEqual(diff_xyz_theta_is_matrix, tolerance)

    def test_specific_angle(self):
        dtype = torch.float32
        axis = torch.tensor([0, 0, 1], dtype=dtype)
        pivot = torch.tensor([0, 0, 0], dtype=dtype)

        xyz = torch.full((2, 3), 0.0)
        xyz[:, 0] = 1.0

        q = torch.full((2, 4), 0.0)
        q[:, 0] = 1.0

        theta = torch.tensor(math.pi / 2, dtype=dtype)
        theta_t = torch.tensor(theta, dtype=torch.float32)

        rotate = ArticulatedOperator()
        xyz = rotate.get_new_location_revolute(xyz, axis, pivot, theta_t)
        q = rotate.get_new_quaternion(q, axis, theta_t)

        # gt_xyz = torch.tensor([0, 1, 0], dtype=torch.float32)
        gt_xyz = torch.full((2, 3), 0.0)
        gt_xyz[:, 1] = 1.0

        # gt_q = torch.tensor([1 / (2**0.5), 0, 0, 1 / (2**0.5)], dtype=torch.float32)
        gt_q = torch.full((2, 4), 0.0)
        gt_q[:, 0] = 1 / (2**0.5)
        gt_q[:, 3] = 1 / (2**0.5)

        self.assertTrue(torch.allclose(xyz, gt_xyz, atol=1e-6))
        self.assertTrue(torch.allclose(q, gt_q, atol=1e-6))


if __name__ == "__main__":
    # Assuming q_rot is a 4-element tensor representing a quaternion
    unittest.main()
