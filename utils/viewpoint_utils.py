from random import randint


class ViewpointLoader:
    def __init__(self, scene) -> None:
        self._scene = scene
        self._full_viewpoint = self._scene.getTrainCameras().copy()
        self._viewpoint_frames = {}
        self._current_stack = None
        self._current_fid = None
        self._current_stack_dual = {}

    def _load_frame_viewpoint(self):
        for viewpoint_cam in self._full_viewpoint:
            fid = viewpoint_cam.fid.item()
            if fid not in self._viewpoint_frames:
                self._viewpoint_frames[fid] = []
            self._viewpoint_frames[fid].append(viewpoint_cam)

    def get_viewpoint_frame(self, fid: int):
        if not self._viewpoint_frames:
            self._load_frame_viewpoint()

        return self._viewpoint_frames[fid].copy()

    @property
    def current_stack(self):
        return self._current_stack

    def refresh_current_stack(self, fid=None):
        if fid is None:
            self._current_stack = self._full_viewpoint
            return

        self._current_stack = self.get_viewpoint_frame(fid)
        self._current_fid = fid

    def refresh_current_stack_dual(self):
        for stack_id in self._current_stack_dual:
            self._current_stack_dual[stack_id] = self.get_viewpoint_frame(stack_id)

    @property
    def viewpoint_cam(self, load2device: bool = False):
        if not self._current_stack:
            self.refresh_current_stack(self._current_fid)

        cam = self._current_stack.pop(randint(0, len(self._current_stack) - 1))
        return cam if not load2device else cam.load2device()

    @property
    def viewpoint_cam_dual(self, load2device: bool = False):
        if not self._current_stack_dual[1]:
            self._current_stack_dual[1] = self.get_viewpoint_frame(1)
        cam_1 = self._current_stack_dual[1].pop(
            randint(0, len(self._current_stack_dual[1]) - 1)
        )

        if not self._current_stack_dual[2]:
            self._current_stack_dual[2] = self.get_viewpoint_frame(2)
        cam_2 = self._current_stack_dual[2].pop(
            randint(0, len(self._current_stack_dual[2]) - 1)
        )

        return (
            (cam_1, cam_2)
            if not load2device
            else (cam_1.load2device(), cam_2.load2device())
        )
