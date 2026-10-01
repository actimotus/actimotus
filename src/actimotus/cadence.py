"""The walking step rate, read from the thigh's angle from vertical.

A thigh swings once per stride, so its angle from vertical carries the stride rate and its harmonics.
The estimator scores every candidate stride rate by the sum of its first harmonics, keeps the best
interior peak, tests it for an octave error on its odd harmonics, and doubles it: the step rate is
exactly twice the stride rate.

It replaces a 1.5-2.5 Hz band-pass on the long axis, which was right only between 90 and 150 steps a
minute. Against counted steps (84 treadmill stages, 21 adults) it puts 98.7% of seconds within 10%,
with 0.05% doubled and 0.31% halved.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

#: The stride band searched, in Hz: 0.4-1.8 Hz is 48-216 steps a minute.
STRIDE_BAND = (0.40, 1.80)

#: The band-pass applied before the harmonic sum, in Hz: wide enough to keep four harmonics of the
#: fastest stride, narrow enough to drop drift below and impact ringing above.
SEARCH_BAND = (0.30, 7.50)

#: The running bands, in Hz: 1.0-2.2 Hz is 120-264 steps a minute. `run_feature` reads only seconds
#: already classed as running, so its band can leave out the slow rates that walking needs. Against
#: the back sensor on child running it halves 0.2% of seconds where the walking band halves 4.7%.
RUN_STRIDE_BAND = (1.00, 2.20)
RUN_SEARCH_BAND = (0.80, 9.00)

#: Harmonics summed per candidate. Unweighted: `1 / h` weights doubled real recordings.
HARMONICS = 4

#: The octave test's threshold. Anything from 0.4 to 0.6 gives 98.0-98.7% of seconds within 10%.
OCTAVE_RATIO = 0.5


def long_axis_angle(axes: pd.DataFrame) -> np.ndarray:
    """How far the thigh points away from straight down, in degrees, for every sample.

    0 degrees is a vertical thigh (standing), 90 degrees a horizontal one (sitting). When you walk,
    the thigh swings forward and back once per stride, so this angle rises and falls once per stride.
    `step_rate` reads that rhythm.

    It is the same angle as `inclination` in `Sensor.get_angles`, but for every raw sample rather than
    once per second: one value a second is too coarse to see a swing that repeats about once a second.
    """
    x = axes['acc_x'].to_numpy(dtype=np.float64)
    y = axes['acc_y'].to_numpy(dtype=np.float64)
    z = axes['acc_z'].to_numpy(dtype=np.float64)
    length = np.sqrt(x * x + y * y + z * z)

    with np.errstate(invalid='ignore', divide='ignore'):
        cosine = np.where(length > 0, x / length, 1.0)

    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _interior_peak(scores: np.ndarray) -> np.ndarray:
    """The best interior local maximum per row, or the argmax where a row has none.

    An edge candidate has no neighbour, so a score still rising past the band edge cannot be told
    from a peak. Slow drift below the band did exactly that.
    """
    best = np.argmax(scores, axis=1)
    peak = np.zeros(scores.shape, dtype=bool)
    middle = scores[:, 1:-1]
    peak[:, 1:-1] = (middle >= scores[:, :-2]) & (middle >= scores[:, 2:])
    interior = np.argmax(np.where(peak, scores, -np.inf), axis=1)

    return np.where(peak.any(axis=1), interior, best)


def _parabolic_offset(scores: np.ndarray, peak: np.ndarray) -> np.ndarray:
    """Where the peak sits between bins, from a parabola through it and its neighbours, in [-0.5, 0.5]."""
    rows = np.arange(len(peak))
    inside = (peak > 0) & (peak < scores.shape[1] - 1)
    offset = np.zeros(len(peak), dtype=np.float64)

    if not inside.any():
        return offset

    left = scores[rows[inside], peak[inside] - 1]
    middle = scores[rows[inside], peak[inside]]
    right = scores[rows[inside], peak[inside] + 1]
    curvature = left - 2.0 * middle + right

    with np.errstate(invalid='ignore', divide='ignore'):
        fitted = 0.5 * (left - right) / curvature

    fitted = np.where(curvature < 0, fitted, 0.0)
    offset[inside] = np.clip(np.nan_to_num(fitted), -0.5, 0.5)

    return offset


def _magnitude_at(magnitudes: np.ndarray, stride_bins: np.ndarray, multiple: float) -> np.ndarray:
    """The spectrum at `multiple` times each row's stride rate: the larger of the two bins around it."""
    last = magnitudes.shape[1] - 1
    rows = np.arange(len(stride_bins))
    below = np.clip(np.floor(stride_bins * multiple).astype(int), 0, last)
    above = np.clip(below + 1, 0, last)

    return np.maximum(magnitudes[rows, below], magnitudes[rows, above])


def _octave_moves(
    magnitudes: np.ndarray, stride_bins: np.ndarray, bin_hz: float, stride_band: tuple[float, float]
) -> np.ndarray:
    """The factor each row's stride rate is multiplied by: 2, 0.5 or 1.

    A true stride rate `f` carries its odd harmonics. Double when `(a(f) + a(3f)) / (a(2f) + a(4f))`
    is under `OCTAVE_RATIO`, as `f` is then a subharmonic; otherwise halve when
    `(a(f/2) + a(3f/2)) / (a(f) + a(2f))` is over it, as `f` is then the step rate. Neither move may
    leave `stride_band`, or sway near 0.28 Hz halves slow walking.
    """
    at = {m: _magnitude_at(magnitudes, stride_bins, m) for m in (0.5, 1, 1.5, 2, 3, 4)}

    with np.errstate(invalid='ignore', divide='ignore'):
        odd = (at[1] + at[3]) / (at[2] + at[4])
        below = (at[0.5] + at[1.5]) / (at[1] + at[2])

    stride_hz = stride_bins * bin_hz
    up = (odd < OCTAVE_RATIO) & (stride_hz * 2 <= stride_band[1])
    down = ~up & (below > OCTAVE_RATIO) & (stride_hz / 2 >= stride_band[0])

    return np.where(up, 2.0, np.where(down, 0.5, 1.0))


def _stride_bins(windows: np.ndarray, nfft: int, bin_hz: float, stride_band: tuple[float, float]) -> np.ndarray:
    """The stride rate of each window, in fractional bins."""
    magnitudes = np.abs(np.fft.rfft(windows, nfft, axis=1))

    low = int(np.ceil(stride_band[0] / bin_hz))
    high = int(np.floor(stride_band[1] / bin_hz))
    candidates = np.arange(low, high + 1)

    # A harmonic past the end of the spectrum is clipped to the last bin, where a band-passed signal
    # holds nothing.
    last = magnitudes.shape[1] - 1
    bins = np.clip(candidates[None, :] * np.arange(1, HARMONICS + 1)[:, None], 0, last)
    scores = magnitudes[:, bins].sum(axis=1)

    peak = _interior_peak(scores)
    stride_bins = candidates[peak] + _parabolic_offset(scores, peak)

    return stride_bins * _octave_moves(magnitudes, stride_bins, bin_hz, stride_band)


def step_rate(
    x: np.ndarray,
    fs: int,
    block: int = 3600,
    stride_band: tuple[float, float] = STRIDE_BAND,
    search_band: tuple[float, float] = SEARCH_BAND,
) -> np.ndarray:
    """Steps per second, from a signal that repeats once per stride; one value per second.

    For each second it looks at the next 4 s and asks which rhythm in `stride_band` (by default 24 to
    108 strides a minute) fits the signal best, counting the rhythm's overtones too. It checks that the winner is not half
    or double the true rhythm, and doubles it, because one stride is two steps. Not median-filtered.

    Args:
        x: One signal at `fs`. `Features` passes the thigh's `long_axis_angle`.
        fs: The sampling frequency, in Hz.
        block: Windows transformed at once. Every row is independent, so this changes memory only.
        stride_band: The stride rates searched, in Hz.
        search_band: The band-pass applied first, in Hz. Its top must keep `HARMONICS` harmonics of
            the fastest stride in `stride_band`.

    Returns:
        The step rate in Hz, `len(x) // fs` values long.
    """
    window = fs * 4
    nfft = window * 4
    bin_hz = fs / nfft

    nyquist = fs / 2
    b, a = signal.butter(4, [search_band[0] / nyquist, search_band[1] / nyquist], 'band')
    filtered = signal.filtfilt(b, a, np.asarray(x, dtype=np.float64))

    padded = np.pad(filtered.astype(np.float32), (0, window - 1), mode='edge')
    windows = sliding_window_view(padded, window)[::fs]

    stride_bins = np.empty(len(windows), dtype=np.float64)

    for start in range(0, len(windows), block):
        chunk = windows[start : start + block]
        chunk = chunk - chunk.mean(axis=1, keepdims=True, dtype=np.float32)
        stride_bins[start : start + block] = _stride_bins(chunk, nfft, bin_hz, stride_band)

    return (stride_bins * bin_hz * 2.0).astype(np.float32)
