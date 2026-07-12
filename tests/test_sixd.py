#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Property tests for mount_drift's 6D-rotation helpers, using Hypothesis
# (QuickCheck for Python).  Run with pytest, or directly: `python test_sixd.py`.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters", "train"))

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sixd import sixd_angle, sixd_to_R

_f = st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
_v6 = st.lists(_f, min_size=6, max_size=6)


@given(_v6)
@settings(max_examples=200)
def test_angle_in_range(v6):
    # The geodesic angle is always a valid arccos result in [0, 180] degrees.
    ang = sixd_angle(np.array(v6, dtype=np.float64))
    assert -1e-6 <= ang <= 180.0 + 1e-6


@given(_v6)
@settings(max_examples=200)
def test_decode_is_a_rotation(v6):
    a = np.array(v6[:3])
    b = np.array(v6[3:])
    if np.linalg.norm(a) < 1e-2 or np.linalg.norm(np.cross(a, b)) < 1e-2:
        return  # skip ill-conditioned Gram-Schmidt inputs
    R = sixd_to_R(np.array(v6, dtype=np.float64))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-4)  # orthonormal
    assert abs(np.linalg.det(R) - 1.0) < 1e-4          # proper (det = +1)


def test_identity():
    # a = x̂, b = ŷ  ->  R = I  ->  angle ~0 (the 1e-9 normalization eps leaves <0.01°).
    assert sixd_angle(np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)) < 1e-2


if __name__ == "__main__":
    test_angle_in_range()
    test_decode_is_a_rotation()
    test_identity()
    print("mount_drift/sixd: all Hypothesis properties passed")
