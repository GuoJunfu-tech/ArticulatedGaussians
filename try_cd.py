import torch
# import chamfer_distance.chamfer3D.dist_chamfer_3D
# import chamfer_distance.fscore

from chamfer_distance.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from chamfer_distance.fscore import fscore

chamLoss = chamfer_3DDist()
points1 = torch.rand(32, 1000, 3).cuda()
points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
f_score, precision, recall = fscore(dist1, dist2)
