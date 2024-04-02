from random import randint, choice


class ViewpointLoader:
    def __init__(self, scene) -> None:
        self._scene = scene
        self._full_viewpoint = self._scene.getTrainCameras().copy()
        self._viewpoint_frames = {}
        self._load_frame_viewpoint()
        # self._current_stack = None
        # self._current_fid = None

        self._current_stack_dual = {}
        for stack_id in self._viewpoint_frames:
            self._current_stack_dual[stack_id] = self.get_viewpoint_frame(stack_id)

    def _load_frame_viewpoint(self):
        for viewpoint_cam in self._full_viewpoint:
            fid = viewpoint_cam.fid.item()
            if fid not in self._viewpoint_frames:
                self._viewpoint_frames[fid] = []
            self._viewpoint_frames[fid].append(viewpoint_cam)

    def _ensure_frame_loaded(self, fid):
        if not self._current_stack_dual[fid]:
            self._current_stack_dual[fid] = self.get_viewpoint_frame(fid)

    def _get_random_cam_from_stack(self, fid: int, load2device: bool = False):
        self._ensure_frame_loaded(fid)
        cam = choice(self._current_stack_dual[fid])
        self._current_stack_dual[fid].remove(cam)
        return cam.load2device() if load2device else cam

    def get_viewpoint_frame(self, fid: int):
        if not self._viewpoint_frames:
            self._load_frame_viewpoint()

        return self._viewpoint_frames[fid].copy()

    def get_viewpoint_cam(self, fid, load2device=False):
        self._ensure_frame_loaded(fid)
        return self._get_random_cam_from_stack(fid, load2device)

    def get_viewpoint_cam_dual(self, load2device=False):
        cam_0 = self._get_random_cam_from_stack(0, load2device)
        cam_1 = self._get_random_cam_from_stack(1, load2device)
        return cam_0, cam_1
