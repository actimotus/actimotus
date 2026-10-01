"""The step rate behind `walk_feature` and `run_feature`, on signals whose step rate is known.

The shipped `walk_feature` band-passed the thigh's long axis to 1.5-2.5 Hz (90-150 steps/min)
before it looked for a peak. Outside that band a different harmonic won: below it the rate read
1.5 times the truth, above it half. Its bin was also scaled with 0.0588 Hz where the transform's
bin is 0.0625 Hz, so every rate ran about 6% low. The repair reads the thigh's angle from vertical,
which swings once per stride, finds the stride rate on its harmonics and doubles it.

Validated against counted steps (84 treadmill stages, 21 adults, 73-112 steps/min): 98.7% of
seconds within 10%, 0.05% doubled. That check needs recorded data; these tests need none.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from actimotus.cadence import RUN_STRIDE_BAND, long_axis_angle, step_rate
from actimotus.features import Features

FS = 30
TZ = 'Europe/Copenhagen'

#: A decaying harmonic series, the shape of a clean thigh swing.
THIGH_HARMONICS = (1.00, 0.50, 0.25, 0.10)


def thigh_like(spm: float, seconds: int = 120, harmonics: tuple[float, ...] = THIGH_HARMONICS) -> np.ndarray:
    """One signal at a known step rate: the fundamental is the stride, half the step rate."""
    t = np.arange(0, seconds, 1 / FS)
    stride_hz = spm / 120.0

    return sum(a * np.sin(2 * np.pi * (h + 1) * stride_hz * t) for h, a in enumerate(harmonics)).astype(np.float32)


def sway(hz: float, amplitude: float, seconds: int = 120) -> np.ndarray:
    """A slow sine under the stride band, a body swaying as it walks."""
    t = np.arange(0, seconds, 1 / FS)

    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def drift(amplitude: float, seconds: int = 120, seed: int = 1) -> np.ndarray:
    """Broadband noise below 0.35 Hz: slow posture changes, without a peak."""
    noise = np.random.default_rng(seed).standard_normal(seconds * FS)
    b, a = signal.butter(2, 0.35 / (FS / 2), 'low')
    slow = signal.filtfilt(b, a, noise)

    return (amplitude * slow / np.std(slow)).astype(np.float32)


def swinging_axes(spm: float, seconds: int = 120) -> pd.DataFrame:
    """Three axes of a thigh rocking 12 degrees either side of 30 degrees, once per stride."""
    swing = thigh_like(spm, seconds=seconds)
    angle = np.radians(30.0 + 12.0 * swing / np.abs(swing).max())
    index = pd.date_range('2024-09-02 07:00:00', periods=len(angle), freq=pd.Timedelta(seconds=1 / FS), tz=TZ)

    return pd.DataFrame(
        {'acc_x': np.cos(angle), 'acc_y': np.zeros_like(angle), 'acc_z': np.sin(angle)},
        index=index,
        dtype=np.float32,
    )


def spm_of(x: np.ndarray) -> float:
    """One number for a steady signal: the median step rate, in steps per minute."""
    return float(np.median(step_rate(x, FS)) * 60)


class TestInclination:
    def test_a_thigh_straight_down_is_zero_degrees(self):
        axes = pd.DataFrame({'acc_x': [1.0], 'acc_y': [0.0], 'acc_z': [0.0]})

        assert long_axis_angle(axes)[0] == pytest.approx(0.0, abs=1e-6)

    def test_a_thigh_horizontal_is_ninety_degrees(self):
        axes = pd.DataFrame({'acc_x': [0.0], 'acc_y': [0.0], 'acc_z': [1.0]})

        assert long_axis_angle(axes)[0] == pytest.approx(90.0, abs=1e-6)

    def test_it_does_not_depend_on_how_hard_the_segment_accelerates(self):
        one = pd.DataFrame({'acc_x': [0.7071], 'acc_y': [0.0], 'acc_z': [0.7071]})
        two = pd.DataFrame({'acc_x': [1.4142], 'acc_y': [0.0], 'acc_z': [1.4142]})

        assert long_axis_angle(one)[0] == pytest.approx(long_axis_angle(two)[0])

    def test_a_zero_vector_and_a_reversed_axis_give_no_nan(self):
        axes = pd.DataFrame({'acc_x': [0.0, -1.0], 'acc_y': [0.0, 0.0], 'acc_z': [0.0, 0.0]})

        assert long_axis_angle(axes) == pytest.approx([0.0, 180.0], abs=1e-6)


class TestStepRate:
    @pytest.mark.parametrize('spm', [55, 70, 90, 110, 130, 160, 190])
    def test_a_clean_thigh_reads_its_step_rate(self, spm: int):
        """The whole walking range, not only the 90-150 the shipped band-pass allowed."""
        assert spm_of(thigh_like(spm)) == pytest.approx(spm, abs=2.0)

    def test_it_returns_one_value_per_second_in_hz(self):
        rate = step_rate(thigh_like(100, seconds=40), FS)

        assert rate.shape == (40,)
        assert np.median(rate) == pytest.approx(100 / 60, abs=0.04)

    def test_it_tells_apart_two_steps_a_minute(self):
        """The bin alone is 7.5 steps/min; interpolation between bins resolves less."""
        slower, faster = spm_of(thigh_like(104)), spm_of(thigh_like(106))

        assert faster - slower > 0.5

    @pytest.mark.parametrize('spm', [71, 83, 97, 104, 111, 123, 137])
    def test_a_rate_off_the_bin_grid_is_not_pulled_onto_it(self, spm: int):
        assert spm_of(thigh_like(spm)) == pytest.approx(spm, abs=2.0)

    def test_a_dominant_second_harmonic_does_not_double_the_answer(self):
        """Measured on real thighs at 3.3 km/h: the step rate is the tallest peak."""
        assert spm_of(thigh_like(90, harmonics=(0.59, 1.00, 0.51, 0.26))) == pytest.approx(90, abs=3.0)

    def test_a_dominant_third_harmonic_does_not_treble_the_answer(self):
        """Measured on real thighs at 2.5 km/h."""
        assert spm_of(thigh_like(76, harmonics=(1.00, 0.94, 0.91, 0.74))) == pytest.approx(76, abs=3.0)

    @pytest.mark.parametrize('spm', [90, 100])
    def test_even_harmonics_that_dominate_do_not_double_it(self, spm: int):
        """The shape of the seconds the flat harmonic sum doubled; it read 179.8 and 200.5."""
        harmonics = (0.60, 1.00, 0.20, 0.60, 0.10, 0.50, 0.10, 0.50)

        assert spm_of(thigh_like(spm, harmonics=harmonics)) == pytest.approx(spm, abs=2.0)

    @pytest.mark.parametrize('spm', [100, 105])
    def test_drift_below_the_band_does_not_pin_it_to_the_floor(self, spm: int):
        """52.5 steps/min is the lowest candidate doubled, not a reading."""
        assert spm_of(thigh_like(spm) + drift(1.0)) == pytest.approx(spm, abs=2.0)

    def test_a_pick_at_the_band_edge_is_not_a_peak(self):
        assert spm_of(thigh_like(100) + sway(0.31, 0.8)) == pytest.approx(100, abs=2.0)

    @pytest.mark.parametrize(('hz', 'amplitude'), [(0.28, 1.5), (0.30, 1.2)])
    def test_slow_walking_over_sway_is_not_halved_out_of_the_band(self, hz: float, amplitude: float):
        """Without the band guard the sway reads as the stride and 67 halves to 34."""
        assert spm_of(thigh_like(67) + sway(hz, amplitude)) == pytest.approx(67, abs=2.0)

    def test_blocks_do_not_change_the_answer(self):
        """The transform runs in blocks so memory stays flat on multi-day recordings."""
        x = thigh_like(100, seconds=300) + drift(0.5, seconds=300)

        np.testing.assert_array_equal(step_rate(x, FS, block=7), step_rate(x, FS, block=10_000))


class TestStepsFeatures:
    def test_walk_feature_is_the_step_rate_of_the_thigh_angle_in_hz(self):
        features = Features().get_steps_features(swinging_axes(90.0))

        assert np.median(features['walk_feature']) * 60 == pytest.approx(90.0, abs=2.0)

    @pytest.mark.parametrize('spm', [70, 140])
    def test_walk_feature_reads_outside_the_old_band(self, spm: int):
        """The shipped feature read 70 as about 105 and 160 as about 80."""
        features = Features().get_steps_features(swinging_axes(spm))

        assert np.median(features['walk_feature']) * 60 == pytest.approx(spm, abs=2.0)

    @pytest.mark.parametrize('spm', [140, 170, 200, 240])
    def test_run_feature_is_the_step_rate_of_the_thigh_angle_in_the_running_band(self, spm: int):
        """The walking estimator with a running stride band. 240 is past both the shipped band-pass
        (about 206) and the walking band (216), where the fastest children run."""
        features = Features().get_steps_features(swinging_axes(spm))

        assert np.median(features['run_feature']) * 60 == pytest.approx(spm, abs=2.0)

    def test_run_feature_does_not_halve_running(self):
        """The walking band lets the octave test fall to 48 steps/min, and on child running it did,
        on 4% of seconds. The running band starts at 120, so half of 170 cannot be read."""
        features = Features().get_steps_features(swinging_axes(170.0))

        assert features['run_feature'].min() * 60 >= RUN_STRIDE_BAND[0] * 120

    def test_both_features_give_one_value_per_second(self):
        features = Features().get_steps_features(swinging_axes(100.0, seconds=45))

        assert len(features) == 45
        assert features['walk_feature'].dtype == np.float32
        assert features['run_feature'].dtype == np.float32
