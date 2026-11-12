"""The 4 s windows of `hl_ratio` and of the step rates are centred on their second, like the 2 s rows of x, y, z.

Before, window k started at second k (k to k + 4 s, centre k + 2 s), so these columns described a time 2 s
later than the other columns of the same row.
"""

import numpy as np
import pandas as pd

from actimotus import Features
from actimotus.cadence import step_rate

FS = 30
CHANGE_S = 100  # the signal changes here, at second 100


def test_hl_ratio_window_is_centred():
    t = np.arange(200 * FS) / FS
    z = -0.9 + np.where(t < CHANGE_S, 0.3 * np.sin(2 * np.pi * 3 * t), 0.0)  # high frequency only before the change
    df = pd.DataFrame({'acc_x': 0 * t, 'acc_y': 0 * t, 'acc_z': z})

    hl = Features(calibrate=False).get_hl_ratio(df).to_numpy()

    full = hl[50]
    assert hl[CHANGE_S - 1] > 0.5 * full  # window 97-101 s: 3 of 4 s before the change
    assert hl[CHANGE_S + 1] < 0.5 * full  # window 99-103 s: 1 of 4 s before the change


def test_step_rate_window_is_centred():
    t = np.arange(200 * FS) / FS
    stride_hz = np.where(t < CHANGE_S, 0.8, 1.2)
    phase = 2 * np.pi * np.cumsum(stride_hz) / FS
    angle = 20 * np.sin(phase)

    rate = step_rate(angle, FS)

    assert abs(rate[CHANGE_S - 1] - 1.6) < 0.15  # window 97-101 s, mostly before; steps = 2 x strides
    assert abs(rate[CHANGE_S + 2] - 2.4) < 0.15  # window 100-104 s: all after the change
