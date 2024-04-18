import torch
import unittest

from utils.loss_utils import chamfer_distance_loss


class TestChamferDistance(unittest.TestCase):
    def test_empty_point_clouds(self):
        """test empty pcd"""
        p1 = torch.empty(0, 3)
        p2 = torch.empty(0, 3)
        with self.assertRaises(AssertionError):
            self.assertTrue(torch.isnan(chamfer_distance_loss(p1, p2)))

    def test_identical_point_clouds(self):
        """test two same pcd"""
        p1 = torch.rand(10, 3)
        p2 = p1.clone()
        self.assertAlmostEqual(chamfer_distance_loss(p1, p2).item(), 0, places=5)

    def test_symmetry(self):
        """test symmetry pcd"""
        p1 = torch.rand(10, 3)
        p2 = torch.rand(15, 3)
        dist1 = chamfer_distance_loss(p1, p2)
        dist2 = chamfer_distance_loss(p2, p1)
        self.assertAlmostEqual(dist1.item(), dist2.item(), places=5)

    def test_known_result(self):
        """test a predefined pcd"""
        p1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        p2 = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        expected_distance = (3 + 3) / 2
        actual_distance = chamfer_distance_loss(p1, p2)
        self.assertAlmostEqual(actual_distance.item(), expected_distance, places=5)


if __name__ == "__main__":
    unittest.main()
