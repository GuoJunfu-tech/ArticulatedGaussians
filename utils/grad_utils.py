# def get_grad(tensor, filter, type="xyz"):
#     if type == "xyz":
#         xyz_gradient_accum[update_filter] += torch.norm(
#             viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
#         )
#         self.denom[update_filter] += 1
