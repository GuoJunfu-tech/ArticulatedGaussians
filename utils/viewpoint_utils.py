from random import choice
import copy


class ViewpointLoader:
    def __init__(self, scene_start, scene_end) -> None:
        self._viewpoint = {
            "start": scene_start.getTrainCameras().copy(),
            "end": scene_end.getTrainCameras().copy(),
        }

        self._stack = copy.deepcopy(self._viewpoint)

    def _reload_stack(self, status):
        if status == "all":
            self._stack = copy.deepcopy(self._viewpoint)
        elif status in ["start", "end"]:
            self._stack[status] = copy.deepcopy(self._viewpoint[status])
        else:
            raise ValueError(f"Invalid status {status}")

    def _get_random_cam_from_stack(self, status: str, load2device: bool = False):
        if not self._stack[status]:
            self._reload_stack(status)

        cam = choice(self._stack[status])
        self._stack[status].remove(cam)
        return cam.load2device() if load2device else cam

    def get_viewpoint_cam(self, status, load2device=False):
        return self._get_random_cam_from_stack(status, load2device)

    def get_viewpoint_cam_dual(self, load2device=False):
        cam_0 = self._get_random_cam_from_stack("start", load2device)
        cam_1 = self._get_random_cam_from_stack("end", load2device)
        return cam_0, cam_1

    def get_cameras(self, status):
        return self._viewpoint[status].copy()
