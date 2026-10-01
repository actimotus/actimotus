"""The stairs threshold is set from the seconds that `get_walk` and `get_stairs` judge.

`_get_stairs_threshold` takes the median `direction` of a pool of walking seconds and adds
`stairs_threshold` to it. A second above that line is `stairs`, a second below it is `walk`.
So the pool must hold walking: moving (`sd_x` above `movement_threshold`), upright
(`inclination` below `inclination_angle`) and below running.

The pool used `sd_x` above 0.25 and no upright test. Slow walking was left out and lying was let
in, so on a recording with little brisk walking the median fell and walking read as stairs.
"""

import pandas as pd
import pytest

from actimotus.classifications.thigh import Thigh

TZ = 'Europe/Copenhagen'
STAIRS_KW = dict(run_threshold=0.65, anterior_posterior_angle=20, stairs_threshold=5)


def _thigh() -> Thigh:
    # _get_stairs_threshold uses no instance state.
    return Thigh(system_frequency=12, vendor='Sens', config={}, orientation=False)


def _seconds(*blocks: tuple[int, float, float, float]) -> pd.DataFrame:
    """Blocks of (seconds, sd_x, inclination, direction), one row per second."""
    rows = [(sd_x, inclination, direction) for n, sd_x, inclination, direction in blocks for _ in range(n)]
    index = pd.date_range('2024-09-02 07:00:00', periods=len(rows), freq='1s', tz=TZ, name='datetime')
    return pd.DataFrame(rows, columns=['sd_x', 'inclination', 'direction'], index=index)


def _threshold(df: pd.DataFrame, movement_threshold: float = 0.075, inclination_angle: float = 47.5) -> float:
    return _thigh()._get_stairs_threshold(
        df, movement_threshold=movement_threshold, inclination_angle=inclination_angle, **STAIRS_KW
    )


def test_lying_does_not_enter_the_pool():
    # Lying with leg movement: sd_x in range and a low direction, but the thigh is horizontal.
    df = _seconds((60, 0.30, 85.0, 2.0), (20, 0.30, 20.0, 12.0))
    assert _threshold(df) == pytest.approx(12.0 + 5)


def test_slow_walking_enters_the_pool():
    # Slow walking moves the thigh less than 0.25 g, but it is walking and get_walk judges it.
    df = _seconds((60, 0.12, 20.0, 10.0), (20, 0.30, 20.0, 14.0))
    assert _threshold(df) == pytest.approx(10.0 + 5)


def test_the_preset_movement_threshold_is_the_floor():
    # 0.09 is walking under DEFAULT (0.075), but not under LEGACY (0.1).
    df = _seconds((60, 0.09, 20.0, 10.0), (20, 0.30, 20.0, 14.0))
    assert _threshold(df, movement_threshold=0.075) == pytest.approx(10.0 + 5)
    assert _threshold(df, movement_threshold=0.1) == pytest.approx(14.0 + 5)


def test_an_empty_pool_assumes_a_typical_walking_direction():
    # `stairs_threshold` alone (5 deg) sits below walking: every walking second would read as stairs.
    df = _seconds((60, 0.30, 85.0, 2.0))
    assert _threshold(df) == pytest.approx(5 + 10)


def test_a_pool_under_ten_seconds_assumes_a_typical_walking_direction():
    # Nine seconds of walking are too few to measure; ten are enough.
    nine = _seconds((9, 0.30, 20.0, 2.0), (60, 0.30, 85.0, 2.0))
    ten = _seconds((10, 0.30, 20.0, 2.0), (60, 0.30, 85.0, 2.0))
    assert _threshold(nine) == pytest.approx(5 + 10)
    assert _threshold(ten) == pytest.approx(5 + 2)
