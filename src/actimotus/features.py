import logging
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from numpy.fft import fft as np_fft
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from .calibration import AutoCalibrate
from .iterators import DataFrameIterator
from .settings import FEATURES, SENS__FLOAT_FACTOR, SENS__NORMALIZATION_FACTOR

logger = logging.getLogger(__name__)


@dataclass
class Features:
    """Processes raw accelerometer data to extract features.

    This class provides a pipeline for transforming raw accelerometer time-series
    data into a set of features. The process includes input validation,
    sampling frequency detection, resampling, optional auto-calibration, and
    the computation of metrics (e.g., High-Low ratio, step-related metrics).

    The class can process data in a single batch or in overlapping chunks to
    mimic Sens's infrastructure.

    Attributes:
        system_frequency: The target frequency (in Hz) to which data is resampled.
            **Warning:** Defaults to 30 Hz. Changing this is not recommended as
            downstream pipelines depend on this frequency.
        validation: If `True`, performs schema and format validation on the input.
        calibrate: If `True`, applies gravitational auto-calibration to the raw data.
        chunking: If `True`, processes the data in overlapping chunks.
        size: The duration of each data chunk. Only used if `chunking` is `True`.
        overlap: The duration of overlap between consecutive chunks. Only used
            if `chunking` is `True`.

    Examples:
        Basic usage with default settings:

        >>> from datetime import timedelta
        >>> extractor = Features()
        >>> # features = extractor.compute(df)

        Configuration for chunked processing:

        >>> extractor = Features(
        ...     chunking=True,
        ...     size=timedelta(days=1),
        ...     overlap=timedelta(minutes=15)
        ... )
    """

    system_frequency: int = 30
    validation: bool = True
    calibrate: bool = True
    chunking: bool = False
    size: timedelta = timedelta(hours=24)
    overlap: timedelta = timedelta(minutes=15)

    def __post_init__(self):
        """Initializes the Features class after its construction."""

        if isinstance(self.size, str):
            self.size = pd.Timedelta(self.size).to_pytimedelta()

        if isinstance(self.overlap, str):
            self.overlap = pd.Timedelta(self.overlap).to_pytimedelta()

    def get_nyquist_freq(self, sampling_frequency: float) -> float:
        """Calculates the Nyquist frequency."""

        return sampling_frequency / 2

    @staticmethod
    def get_sampling_frequency(
        df: pd.DataFrame, *, samples: int | None = 30_000, round_to_nearest: float | None = 0.5
    ) -> float:
        """Calculates the sampling frequency of a time-series DataFrame."""

        time_subset = df.index[:samples] if samples else df.index

        if len(time_subset) < 2:
            raise ValueError('DataFrame must have at least 2 samples to calculate sampling frequency.')

        # Convert to nanoseconds then to seconds for time differences
        time_diffs_seconds = pd.Series(np.diff(time_subset.astype('int64')) / 1e9)

        sf = time_diffs_seconds.mode().values[0]

        if sf <= 0:
            raise ValueError('Invalid time intervals detected in data.')

        sf = 1.0 / sf
        if round_to_nearest and round_to_nearest > 0:
            sf = round(sf / round_to_nearest) * round_to_nearest

        logging.info(f'Detected sampling frequency: {sf:.2f} Hz.', extra={'sampling_frequency': sf})

        return sf

    def _resample_fft(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resamples a DataFrame using the FFT method."""

        start = df.index[0]
        end = df.index[-1]

        n_out = np.floor((end - start).total_seconds() * self.system_frequency).astype(int)
        resampled = signal.resample(df, n_out)

        df = pd.DataFrame(
            resampled,
            columns=df.columns,
            index=pd.date_range(start=start, end=end, periods=n_out),
            dtype=np.float32,
        )

        return df

    # def _resample_poly(self, df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
    #     """Resamples a DataFrame using the polynomial method."""

    #     input_fs = int(sampling_frequency)
    #     target_fs = self.system_frequency

    #     frac_gcd = gcd(input_fs, target_fs)
    #     up = int(target_fs // frac_gcd)
    #     down = int(input_fs // frac_gcd)

    #     start = df.index[0]
    #     end = df.index[-1]

    #     resampled = signal.resample_poly(df, up, down)

    #     df = pd.DataFrame(
    #         resampled,
    #         columns=df.columns,
    #         index=pd.date_range(start=start, end=end, periods=len(resampled)),
    #         dtype=np.float32,
    #     )

    #     return df

    # def _resample_resampy(self, df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
    #     """Resamples a DataFrame using the resampy library."""
    #     time = df.index
    #     acc_data = [
    #         resampy.resample(df[col].values, sampling_frequency, self.system_frequency, parallel=True)
    #         for col in df.columns
    #     ]
    #     acc_data = np.column_stack(acc_data)

    #     n_epochs = len(acc_data)

    #     start, end = time[0], time[-1]
    #     time = pd.date_range(start=start, end=end, periods=n_epochs)
    #     time.name = 'datetime'

    #     df = pd.DataFrame(
    #         acc_data,
    #         index=time,
    #         columns=df.columns,
    #         dtype=np.float32,
    #     )

    #     return df

    def resampling(self, df: pd.DataFrame, sampling_frequency: float, tolerance=1) -> pd.DataFrame:
        """Resamples a DataFrame to the system frequency if necessary."""

        if math.isclose(sampling_frequency, self.system_frequency, abs_tol=tolerance):
            logger.info(
                f'Sampling frequency is {self.system_frequency} Hz, no resampling needed.',
                extra={'sampling_frequency': sampling_frequency},
            )
            return df

        df = self._resample_fft(df)
        # df = self._resample_resampy(df, sampling_frequency)
        # df = self._resample_poly(df, sampling_frequency)

        return df

    def get_hl_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculates the High-Low Ratio (HL-Ratio) from the Z-axis accelerometer data."""

        order = 3
        cut_off = 1
        window = self.system_frequency * 4
        cut_off = cut_off / self.get_nyquist_freq(self.system_frequency)

        axis_z = df['acc_z'].values

        b, a = signal.butter(order, cut_off, 'low')
        low = signal.filtfilt(b, a, axis_z, axis=0)
        low = np.abs(low.astype(np.float32))

        b, a = signal.butter(order, cut_off, 'high')
        high = signal.filtfilt(b, a, axis_z, axis=0)
        high = np.abs(high.astype(np.float32))

        pad_width = window - 1
        high = np.pad(high, (0, pad_width), mode='edge')
        low = np.pad(low, (0, pad_width), mode='edge')

        high_windows = sliding_window_view(high, window)[:: self.system_frequency]
        mean_high = np.mean(high_windows, axis=1, dtype=np.float32)

        low_windows = sliding_window_view(low, window)[:: self.system_frequency]
        mean_low = np.mean(low_windows, axis=1, dtype=np.float32)

        hl_ratio = np.divide(
            mean_high, mean_low, out=np.zeros_like(mean_high), where=mean_low != 0
        )  # NOTE: Check what happens if mean_low is zero

        return pd.Series(hl_ratio, name='hl_ratio')

    def _get_steps_feature(self, arr: np.ndarray) -> np.ndarray:
        """Computes the steps feature from an array."""

        window = self.system_frequency * 4  # 120 (system frequency = 30) samples equal to 2 seconds
        steps_window = 4 * window  # 480 (system frequency = 30) samples equal to 8 seconds
        half_size = window * 2  # 240 (system frequency = 30) samples equal to 4 seconds
        arr = arr.astype(np.float32)

        pad_width = window - 1
        arr = np.pad(arr, (0, pad_width), mode='edge')

        windows = sliding_window_view(arr, window)[:: self.system_frequency]
        windows = windows - np.mean(windows, axis=1, keepdims=True, dtype=np.float32)

        fft_result = np_fft(windows, steps_window)[:, :half_size]
        magnitudes = 2 * np.abs(fft_result)

        return np.argmax(magnitudes, axis=1)

    def get_steps_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates walking and running features from accelerometer data."""

        axis_x = df['acc_x'].values
        nyquist_frequency = self.get_nyquist_freq(self.system_frequency)

        b, a = signal.butter(6, 2.5 / nyquist_frequency, 'low')
        filtered = signal.lfilter(b, a, axis_x, axis=0)

        b, a = signal.butter(6, 1.5 / nyquist_frequency, 'high')
        walk = signal.lfilter(b, a, filtered, axis=0)

        b, a = signal.butter(6, 3 / nyquist_frequency, 'high')
        run = signal.lfilter(b, a, walk)

        df = pd.DataFrame(
            {
                'walk_feature': self._get_steps_feature(walk),
                'run_feature': self._get_steps_feature(run),
            },
        )

        return df

    def get_tensor(self, arr: np.ndarray) -> np.ndarray:
        """Creates a tensor from an array."""
        pb = np.vstack((arr[: self.system_frequency], arr))
        pa = np.vstack((arr, arr[-self.system_frequency :]))
        n = pb.shape[0] // self.system_frequency
        tensor = np.concatenate(
            [
                pb[: n * self.system_frequency].reshape(self.system_frequency, n, 3, order='F'),
                pa[: n * self.system_frequency].reshape(self.system_frequency, n, 3, order='F'),
            ],
            axis=0,
        )
        return tensor[:, :-1, :]

    def downsampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsamples the input signal data and computes various statistical features."""

        axes = df.values

        b, a = signal.butter(4, 5 / self.get_nyquist_freq(self.system_frequency), 'low')
        filtered = signal.lfilter(b, a, axes, axis=0).astype(np.float32)

        tensor = self.get_tensor(filtered)

        mean = np.mean(tensor, axis=0)
        sd = tensor.std(axis=0, ddof=1)
        sum = np.sum(tensor, axis=0)
        sq_sum = np.sum(np.square(tensor), axis=0)
        sum_dot_xz = np.sum((tensor[:, :, 0] * tensor[:, :, 2]), axis=0)

        df = np.concatenate([mean, sd, sum, sq_sum], axis=1)

        df = pd.DataFrame(
            df,
            columns=[
                'x',
                'y',
                'z',
                'sd_x',
                'sd_y',
                'sd_z',
                'sum_x',
                'sum_y',
                'sum_z',
                'sq_sum_x',
                'sq_sum_y',
                'sq_sum_z',
            ],
        )
        df['sum_dot_xz'] = sum_dot_xz

        return df

    def check_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates and standardizes the input accelerometer DataFrame."""

        if not self.validation:
            return df

        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input must be a pandas DataFrame.')

        if df.empty:
            raise ValueError('DataFrame cannot be empty.')

        if df.shape[1] != 3:
            raise ValueError(
                f'DataFrame must have exactly 3 columns for accelerometer data, but has {df.shape[1]} columns.'
            )

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError(f'DataFrame index must be of datetime type, but got {df.index.dtype}.')

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must contain numeric data, but got {df[col].dtype}.")

        df = df.iloc[:, :3]
        df.columns = ['acc_x', 'acc_y', 'acc_z']

        return df

    def _compute_chunk(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Computes features for a single chunk of data."""

        not_overlaps = df[~df['overlap']]
        start, end = not_overlaps.index[0], not_overlaps.index[-1]

        df = self._compute(
            df.iloc[:, :3],
            **kwargs,
        )
        df = df.loc[(df.index >= start) & (df.index < end)]

        return df

    def _compute_chunks(self, df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
        """Computes features for a DataFrame in chunks."""

        chunks = DataFrameIterator(df, size=self.size, overlap=self.overlap)
        computed = []

        for chunk in chunks:
            computed.append(self._compute_chunk(chunk, sampling_frequency=sampling_frequency))

        computed = pd.concat(computed)
        computed.sort_index(inplace=True)

        return computed

    def segments(self, df: pd.DataFrame, duration: str | timedelta) -> Any:
        """Splits a DataFrame into segments based on time gaps."""
        td = df.index.to_series().diff()
        duration = pd.Timedelta(duration)

        diff = td > duration
        gaps = diff.cumsum()
        gaps.name = 'gaps'

        return df.groupby(gaps)

    def _compute(self, df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
        """Computes features for a DataFrame."""

        dfs = []

        for name, segment in self.segments(df, timedelta(seconds=1)):
            if segment.index[-1] - segment.index[0] < timedelta(seconds=5):
                continue

            segment = self.resampling(segment, sampling_frequency)
            hl_ratio = self.get_hl_ratio(segment)
            steps_features = self.get_steps_features(segment)
            downsampled = self.downsampling(segment)

            n = min(len(hl_ratio), len(steps_features), len(downsampled))
            start = segment.index[0].ceil('s')
            segment = pd.concat([downsampled, hl_ratio, steps_features], axis=1)
            segment = segment.iloc[:n]
            segment.index = pd.date_range(
                start=start,
                periods=n,
                freq=timedelta(seconds=1),
                name='datetime',
            )
            dfs.append(segment)

        dfs = pd.concat(dfs)
        dfs['sf'] = sampling_frequency

        logger.info('Features computed.')

        return dfs

    def compute(self, df: pd.DataFrame, sampling_frequency: float | None = None) -> pd.DataFrame:
        """Computes extracted features from raw accelerometer data.

        This method orchestrates the pipeline: it handles format validation,
        frequency inference, resampling, and optional calibration. It then
        dispatches the computation to either a chunked or batch processing
        backend based on the instance configuration.

        Args:
            df: The input DataFrame containing accelerometer data. Must possess
                a `DatetimeIndex` and contain only accelerometer columns (axes X, Y, Z).
            sampling_frequency: The sampling frequency of the data in Hertz.
                If `None`, it is inferred automatically from the index of `df`.

        Returns:
            A DataFrame containing the computed features.

        Examples:
            Basic usage where frequency is inferred:

            >>> extractor = Features()
            >>> features = extractor.compute(df)

            Explicitly providing frequency:

            >>> features = extractor.compute(df, sampling_frequency=12.5)
        """

        df = self.check_format(df)

        sampling_frequency = sampling_frequency or self.get_sampling_frequency(df)

        if self.calibrate:
            df = AutoCalibrate().compute(df)

        if self.chunking:
            return self._compute_chunks(df, sampling_frequency)
        else:
            return self._compute(df, sampling_frequency)

    def to_sens(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converts a DataFrame to the SENS format.

        Args:
            df: The input DataFrame.

        Returns:
            A tuple containing timestamps, data, features, and verbose arrays.
        """
        df = df.copy()
        df.index = df.index.astype(np.int64) // 10**6  # Time in milliseconds
        df.drop(columns=['sum_y', 'sq_sum_y'], inplace=True)

        df.fillna(0, inplace=True)
        df[FEATURES] = (df[FEATURES] * SENS__FLOAT_FACTOR).astype(np.int32)

        df['data'] = 1
        df['data'] = df['data'].astype(np.int16)

        df['verbose'] = 0
        df['verbose'] = df['verbose'].astype(np.int32)

        return (
            df.index.values,
            df['data'].values,
            df[FEATURES].values,
            df['verbose'].values,
        )

    def _raw_from_sens(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
    ) -> pd.DataFrame:
        """Converts raw SENS data to a DataFrame."""
        df = pd.DataFrame(
            data,
            index=timestamps,
            columns=['acc_x', 'acc_y', 'acc_z'],
        )

        df = df * SENS__NORMALIZATION_FACTOR
        df.index = pd.to_datetime(df.index, unit='ms')
        df.index.name = 'datetime'

        return df

    def _df_to_sens(
        self,
        df: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """Converts a DataFrame to the SENS format."""

        df = df / SENS__NORMALIZATION_FACTOR
        timestamps = (df.index.astype(np.int64) // 10**6).values
        timestamps = np.array([timestamps])

        data = df.values
        data = np.array([data])

        return timestamps, data

    def compute_sens(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes features from SENS data.

        Args:
            timestamps: An array of timestamps.
            data: An array of accelerometer data.

        Returns:
            A tuple containing timestamps, data, features, and verbose arrays in the SENS format.
        """
        df = self._raw_from_sens(timestamps[0], data[0])
        features = self.compute(df)

        return self.to_sens(features)
