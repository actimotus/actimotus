"""Feature rows stay on real time when the real sampling rate drifts, and no whole second is lost.

SENS sensors run at 12.38-12.49 Hz, not at the nominal 12.5 Hz, and the rate changes during the day. Before
the fix, the rows were labelled by counting samples, so the labels ran late by up to tens of seconds.
"""

import numpy as np
import pandas as pd
import pytest

from actimotus import Features

TZ = 'Europe/Copenhagen'
PERIOD_S = 600  # a slow wave in x: a label that is 10 s late gives an error of about 0.04 g


def drifting_frame(start, rates, hours_each=1.0):
    """Samples of a known signal at a real rate that changes each `hours_each` hours."""
    dt = np.concatenate([np.full(int(3600 * hours_each * r), 1 / r) for r in rates])
    t = np.concatenate([[0.0], np.cumsum(dt)[:-1]])  # real seconds after `start`
    x = -0.5 + 0.4 * np.sin(2 * np.pi * t / PERIOD_S)
    df = pd.DataFrame(
        {'acc_x': x, 'acc_y': 0.1 + 0 * t, 'acc_z': -0.85 + 0 * t},
        index=(start + pd.to_timedelta(t, unit='s')).as_unit('ns'),
        dtype=np.float32,
    )
    df.index.name = 'datetime'
    return df


def test_rows_stay_on_real_time_when_the_rate_drifts():
    start = pd.Timestamp('2024-09-02 08:00:00.3', tz=TZ)
    df = drifting_frame(start, rates=[12.44, 12.36, 12.44, 12.36])

    features = Features(calibrate=False).compute(df, sampling_frequency=12.5)

    assert features.index.tz is not None
    assert features.index[0] == start.ceil('s')
    assert (features.index == features.index.floor('s')).all()

    inner = features.iloc[60:-60]  # away from the edges of the filters and of the resampling
    seconds = (inner.index - start).total_seconds().to_numpy()
    expected = -0.5 + 0.4 * np.sin(2 * np.pi * seconds / PERIOD_S)
    assert np.abs(inner['x'].to_numpy() - expected).max() < 0.003


@pytest.mark.parametrize('before_boundary_s', [0.080, 0.004])
def test_one_block_per_day_keeps_every_second_once(before_boundary_s):
    # The chunk boundary is 8 ms after a whole second. If a chunk keeps rows from its first sample on, the row of
    # that whole second is in neither chunk (0.080: the sample before the boundary is in the second before).
    # If a chunk keeps rows up to the second of its last sample, the row is in both (0.004: the sample before the
    # boundary is in the same second as the boundary).
    start = pd.Timestamp('2024-09-02 22:00:00.008', tz=TZ)
    df = drifting_frame(start, rates=[12.5], hours_each=26)
    ns = df.index.as_unit('ns').asi8.copy()
    boundary = int(86_400 * 12.5)  # the first sample of the second day is exactly on the chunk boundary
    ns[boundary] = (start + pd.Timedelta(days=1)).as_unit('ns').value
    ns[boundary - 1] = ns[boundary] - int(before_boundary_s * 1e9)
    df.index = pd.to_datetime(ns, utc=True).tz_convert(TZ).rename('datetime')

    features = Features(calibrate=False, chunking=True, size='1D').compute(df, sampling_frequency=12.5)

    seconds = (features.index[-1] - features.index[0]).total_seconds()
    assert features.index.is_unique
    assert features.index.is_monotonic_increasing
    assert len(features) == seconds + 1
