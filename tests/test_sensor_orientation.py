"""Flip correction must recover the features a correctly-mounted sensor would have produced.

`sum_dot_xz` is the sum of x*z over the window -- the cross term `Thigh._rotate_sd` uses to
rebuild sd_x/sd_z after rotating by the reference angle. Being a PRODUCT of two axes, it
behaves differently from every other stored statistic under a flip: the single flips negate
one of x/z so it changes sign, while the combined flip negates both so it does not.

The combined branch used to negate it anyway, which silently corrupted sd_x/sd_z for every
downstream activity gate.
"""

import numpy as np
import pandas as pd
import pytest

from actimotus.classifications.sensor import Sensor
from actimotus.features import Features

SYSTEM_FREQUENCY = 30

# (upside_down, inside_out) -> the sign each raw axis picks up. Each is a 180 degree
# rotation about the axis whose sign is left alone.
FLIPS = {
    (True, False): (-1, -1, 1),  # about z
    (False, True): (1, -1, -1),  # about x
    (True, True): (-1, 1, -1),  # about y
}

STATISTICS = [
    'x', 'y', 'z',
    'sd_x', 'sd_y', 'sd_z',
    'sum_x', 'sum_y', 'sum_z',
    'sq_sum_x', 'sq_sum_y', 'sq_sum_z',
    'sum_dot_xz',
    'hl_ratio',
]


class _Sensor(Sensor):
    """A Sensor whose flip detection is dictated rather than measured.

    The detectors are a separate concern with their own failure modes; wiring real ones in
    would make this test depend on them and stop it being about the correction.
    """

    def __init__(self, upside_down: bool, inside_out: bool):
        super().__init__(orientation=True)
        self._upside_down = upside_down
        self._inside_out = inside_out

    def check_upside_down_flip(self, df):
        return self._upside_down

    def check_inside_out_flip(self, df):
        return self._inside_out

    def rotate_by_reference_angle(self, df, angle):  # pragma: no cover - unused here
        return df

    def calculate_reference_angle(self, df):  # pragma: no cover - unused here
        return 0.0, None


def _raw(seconds: int = 90) -> pd.DataFrame:
    """Thigh-like acceleration with genuinely correlated x and z.

    The correlation matters: if `sum_dot_xz` averaged to zero the sign under test would be
    unobservable and the whole test would pass vacuously.
    """
    n = seconds * SYSTEM_FREQUENCY
    t = np.arange(n) / SYSTEM_FREQUENCY
    rng = np.random.default_rng(0)

    stride = 2 * np.pi * 1.9 * t  # ~1.9 Hz, a walking cadence
    x = 0.85 + 0.30 * np.sin(stride) + 0.02 * rng.standard_normal(n)
    z = -0.35 + 0.25 * np.sin(stride + 0.6) + 0.02 * rng.standard_normal(n)
    y = 0.05 + 0.10 * np.cos(stride) + 0.02 * rng.standard_normal(n)

    return pd.DataFrame(
        {'acc_x': x, 'acc_y': y, 'acc_z': z},
        index=pd.date_range('2026-01-01', periods=n, freq=pd.Timedelta(seconds=1 / SYSTEM_FREQUENCY)),
    )


def _features(raw: pd.DataFrame) -> pd.DataFrame:
    df = Features(calibrate=False, system_frequency=SYSTEM_FREQUENCY).compute(raw)
    sensor = _Sensor(False, False)
    df[['inclination', 'side_tilt', 'direction']] = sensor.get_angles(df)
    return df


@pytest.fixture(scope='module')
def upright() -> pd.DataFrame:
    """Features from a correctly-mounted sensor -- what a correction must reproduce."""
    return _features(_raw())


def test_the_signal_actually_exercises_the_cross_term(upright):
    """Guards every assertion below: a `sum_dot_xz` near zero would hide a sign error."""
    assert upright['sum_dot_xz'].abs().median() > 1.0


@pytest.mark.parametrize(('upside_down', 'inside_out'), list(FLIPS))
def test_correction_recovers_the_upright_features(upright, upside_down, inside_out):
    """Flip the sensor physically, then undo it: every stored statistic must come back."""
    signs = FLIPS[(upside_down, inside_out)]
    flipped = _features(_raw() * signs)

    corrected = _Sensor(upside_down, inside_out).fix_sensor_orientation(flipped)

    for column in STATISTICS:
        np.testing.assert_allclose(
            corrected[column], upright[column], atol=1e-5, rtol=1e-3,
            err_msg=f'{column} not recovered for upside_down={upside_down}, inside_out={inside_out}',
        )


def test_combined_flip_leaves_the_cross_term_alone(upright):
    """The regression. Negating x and z leaves sum(x*z) unchanged -- (-x)(-z) == x*z."""
    flipped = _features(_raw() * FLIPS[(True, True)])

    np.testing.assert_allclose(flipped['sum_dot_xz'], upright['sum_dot_xz'], atol=1e-5, rtol=1e-3)

    corrected = _Sensor(True, True).fix_sensor_orientation(flipped)

    np.testing.assert_allclose(corrected['sum_dot_xz'], flipped['sum_dot_xz'], atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize(('upside_down', 'inside_out'), [(True, False), (False, True)])
def test_single_flips_do_negate_the_cross_term(upright, upside_down, inside_out):
    """The other half: removing the negation everywhere would be just as wrong."""
    flipped = _features(_raw() * FLIPS[(upside_down, inside_out)])

    np.testing.assert_allclose(flipped['sum_dot_xz'], -upright['sum_dot_xz'], atol=1e-5, rtol=1e-3)

    corrected = _Sensor(upside_down, inside_out).fix_sensor_orientation(flipped)

    np.testing.assert_allclose(corrected['sum_dot_xz'], -flipped['sum_dot_xz'], atol=1e-5, rtol=1e-3)
