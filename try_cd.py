import torch
import sys

sys.path.append("./submodules/chamfer-distance")
from chamfer3D.dist_chamfer_3D import chamfer_3DDist
from fscore import fscore

chamLoss = chamfer_3DDist()

points1 = torch.rand(32, 1000, 3).cuda()
points2 = torch.rand(32, 2000, 3).cuda()

dist1, dist2, idx1, idx2 = chamLoss(points1, points2)

f_score, precision, recall = fscore(dist1, dist2)
print(f_score, precision, recall)
print(torch.mean(dist1) + torch.mean(dist2))
