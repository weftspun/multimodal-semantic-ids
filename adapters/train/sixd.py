# SPDX-License-Identifier: MIT
# Pure 6D-rotation helpers (Zhou et al. continuous representation), importable for
# property tests.  Used by device_apply.py.
import numpy as np


def sixd_to_R(v6):
    """6D vector -> rotation matrix, columns [a, b, a×b] (Gram-Schmidt)."""
    a, b = v6[:3], v6[3:]
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b - a * (a @ b)
    b = b / (np.linalg.norm(b) + 1e-9)
    c = np.cross(a, b)
    return np.stack([a, b, c], 1)


def sixd_angle(v6):
    """Geodesic angle (deg) of the 6D rotation from identity: arccos((tr R − 1)/2)."""
    R = sixd_to_R(v6)
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
