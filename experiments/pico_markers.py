"""Convert FM link poses into the calibrated Pico device frame."""
import numpy as np
from scipy.spatial.transform import Rotation
from pico5.prepare import MOUNTS


class PicoMarkers:
    """Match motion 42's standing calibration and palm sites using FM link poses."""
    order = [0, 3, 4, 1, 2]

    def __init__(self, m, d, ids):
        ids = ids[self.order]
        rotations = d.xmat[ids].reshape(5, 3, 3)
        points = d.xpos[ids].copy()
        torso = m.body("torso_link").id
        points[0] = d.xpos[torso] + d.xmat[torso].reshape(3, 3) @ [.07, 0, .48]
        points[1:3] = d.site_xpos[[m.site(f"{s}_palm").id for s in ("left", "right")]]
        points[3:] += (rotations[3:] @ MOUNTS[3:, :, None])[..., 0]
        self.offsets = (rotations.transpose(0, 2, 1) @ (points - d.xpos[ids])[..., None])[..., 0]
        self.origin = points[0].copy()
        self.origin[2] = points[3:, 2].mean() - .09
        self.scale = 1.65 / (points[0, 2] - self.origin[2])
        self.rest_inv = rotations.transpose(0, 2, 1).copy()

    def encode(self, pos, quat):
        rotations = Rotation.from_quat(quat[self.order], scalar_first=True).as_matrix()
        xyz = pos[self.order] + (rotations @ self.offsets[..., None])[..., 0]
        relative = Rotation.from_matrix(rotations @ self.rest_inv).as_quat()
        return np.column_stack(((xyz - self.origin) * self.scale, relative, np.ones(5)))
