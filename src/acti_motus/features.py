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
    sampling frequency detection, resampling to a consistent frequency, optional
    auto-calibration, and the computation of various features like High-Low (HL)
    ratio, step-related metrics, and downsampled statistical summaries.

    The class can process data in a single batch or in overlapping chunks to
    mimic Sen's infrastructure processing.

    Attributes:
        system_frequency (int): The target frequency (in Hz) to which the data
            is resampled. Defaults to 30 Hz and should not be changed as all pipelines are
            designed to work with this frequency.
        validation (bool): Performs validation on the input DataFrame's
            format.
        calibrate (bool): Applies auto-calibration to the raw data.
        chunking (bool): Processes the data in chunks.
        size (timedelta): The size of each data chunk when chunking is enabled.
        overlap (timedelta): The overlap between consecutive chunks.
    """

    system_frequency: int = 30
    validation: bool = True
    calibrate: bool = True
    chunking: bool = False
    size: timedelta = timedelta(hours=24)
    overlap: timedelta = timedelta(minutes=15)

    def __post_init__(self):
        """Initializes the Features class after its construction.

        Converts size and overlap to timedelta objects if they are provided as strings.
        """
        if isinstance(self.size, str):
            self.size = pd.Timedelta(self.size).to_pytimedelta()

        if isinstance(self.overlap, str):
            self.overlap = pd.Timedelta(self.overlap).to_pytimedelta()

    def get_nyquist_freq(self, sampling_frequency: float) -> float:
        """Calculates the Nyquist frequency.

        Args:
            sampling_frequency: The sampling frequency.

        Returns:
            The Nyquist frequency.
        """
        return sampling_frequency / 2

    @staticmethod
    def get_sampling_frequency(
        df: pd.DataFrame, *, samples: int | None = 30_000, round_to_nearest: float | None = 0.5
    ) -> float:
        """Calculates the sampling frequency of a time-series DataFrame.
        This function determines the sampling frequency by finding the mode of the
        time differences between consecutive samples in the DataFrame's index.
        The calculation can be performed on a subset of the data for
        efficiency.

        Args:
            df (pd.DataFrame): The input DataFrame with a time-based index.
            samples (int | None, optional): The number of initial samples to use for
                the calculation. If None, the entire DataFrame is used.
            round_to_nearest (float | None, optional): The value to which the final
                frequency is rounded. For example, 0.5 will round to the nearest
                0.5 Hz. If None or non-positive, no rounding is performed.

        Returns:
            float: The estimated sampling frequency in Hertz (Hz).

        Raises:
            ValueError: If the DataFrame contains fewer than two samples, making
                frequency calculation impossible.
            ValueError: If the calculated time intervals are non-positive,
                indicating a non-monotonic or invalid time index.
        """
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
        """Resamples a DataFrame using the FFT method.

        Args:
            df: The input DataFrame.

        Returns:
            The resampled DataFrame.
        """
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

    def resampling(self, df: pd.DataFrame, sampling_frequency: float, tolerance=1) -> pd.DataFrame:
        """Resamples a DataFrame to the system frequency if necessary.

        Args:
            df: The input DataFrame.
            sampling_frequency: The sampling frequency of the input DataFrame.
            tolerance: The tolerance for comparing sampling frequencies.

        Returns:
            The resampled DataFrame.
        """
        if math.isclose(sampling_frequency, self.system_frequency, abs_tol=tolerance):
            logger.info(
                f'Sampling frequency is {self.system_frequency} Hz, no resampling needed.',
                extra={'sampling_frequency': sampling_frequency},
            )
            return df

        df = self._resample_fft(df)

        return df

    def get_hl_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculates the High-Low Ratio (HL-Ratio) from the Z-axis accelerometer data.
        This method processes the 'acc_z' signal by first separating it into
        high-frequency and low-frequency components using third-order low-pass and
        high-pass Butterworth filters with a cutoff frequency of 1 Hz.

        The absolute values of these filtered signals are then processed using a
        sliding window. The mean of the high-frequency components and the mean
        of the low-frequency components are calculated for each window. The
        HL-Ratio is the ratio of the mean of the high-frequency component to the
        mean of the low-frequency component. Division by zero is handled by
        returning zero for that window.

        Args:
            df (pd.DataFrame): Input DataFrame containing the time-series data.
                               Must include an 'acc_z' column for the Z-axis
                               accelerometer readings.

        Returns:
            pd.Series: A pandas Series named 'hl_ratio' containing the
                       calculated high-low ratio for each window.
        """

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
        """Computes the steps feature from an array.

        Args:
            arr: The input array.

        Returns:
            An array containing the steps feature.
        """
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
        """Calculates walking and running features from accelerometer data.
        This method processes the x-axis accelerometer signal ('acc_x') to
        extract features indicative of walking and running. It applies a
        series of Butterworth filters to isolate the frequency bands
        typically associated with these activities.

        The process involves:
        1. Applying a 6th-order low-pass filter with a 2.5 Hz cutoff.
        2. Applying a 6th-order high-pass filter with a 1.5 Hz cutoff to the
           result of step 1 to isolate the 'walk' signal.
        3. Applying a 6th-order high-pass filter with a 3 Hz cutoff to the
           'walk' signal to isolate the 'run' signal.
        4. Computing a feature value for both the 'walk' and 'run' signals
           using frequency component with the highest magnitude.

        Args:
            df (pd.DataFrame): The input DataFrame containing accelerometer data.
                               It must have an 'acc_x' column.

        Returns:
            pd.DataFrame: A new DataFrame with two columns: 'walk_feature' and
                          'run_feature'.
        """

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
        """Creates a tensor from an array.

        Args:
            arr: The input array.

        Returns:
            A tensor.
        """
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
        """Downsamples the input signal data and computes various statistical features.
        For each window, it calculates the mean, standard deviation, sum, and sum of squares for each axis
        (x, y, z), as well as the sum of the dot product between the x and z axes.

        Args:
            df (pd.DataFrame): The input DataFrame containing time-series signal
                data, expected to have columns representing different axes.

        Returns:
            pd.DataFrame: A new DataFrame where each row corresponds to a
                window of the original signal. The columns contain the
                computed features: 'x', 'y', 'z' (mean values), 'sd_x',
                'sd_y', 'sd_z', 'sum_x', 'sum_y', 'sum_z', 'sq_sum_x',
                'sq_sum_y', 'sq_sum_z', and 'sum_dot_xz'.
        """

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
        """Validates and standardizes the input accelerometer DataFrame.
        This method performs a series of checks to ensure the input DataFrame
        conforms to the expected format for accelerometer data.

        The validation ensures that the input is a non-empty pandas DataFrame with
        exactly three numeric columns and a datetime index. After validation, it
        renames the columns to ['acc_x', 'acc_y', 'acc_z'].

        Args:
            df (pd.DataFrame): The input DataFrame to validate. It is expected
                to have a datetime index and three numeric columns representing
                the x, y, and z axes of an accelerometer.

        Returns:
            pd.DataFrame: The validated and standardized DataFrame with columns
                renamed to ['acc_x', 'acc_y', 'acc_z'].

        Raises:
            TypeError: If the input is not a pandas DataFrame.
            ValueError: If the DataFrame is empty, does not have exactly 3
                columns, has a non-datetime index, or contains non-numeric
                column data.
        """

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
        """Computes features for a single chunk of data.

        Args:
            df: The input DataFrame chunk.
            **kwargs: Additional keyword arguments.

        Returns:
            A DataFrame with computed features for the chunk.
        """
        not_overlaps = df[~df['overlap']]
        start, end = not_overlaps.index[0], not_overlaps.index[-1]

        df = self._compute(
            df.iloc[:, :3],
            **kwargs,
        )
        df = df.loc[(df.index >= start) & (df.index < end)]

        return df

    def _compute_chunks(self, df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
        """Computes features for a DataFrame in chunks.

        Args:
            df: The input DataFrame.
            sampling_frequency: The sampling frequency of the DataFrame.

        Returns:
            A DataFrame with computed features.
        """
        chunks = DataFrameIterator(df, size=self.size, overlap=self.overlap)
        computed = []

        for chunk in chunks:
            computed.append(self._compute_chunk(chunk, sampling_frequency=sampling_frequency))

        computed = pd.concat(computed)
        computed.sort_index(inplace=True)

        return computed

    def _compute(self, df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
        """Computes features for a DataFrame.

        Args:
            df: The input DataFrame.
            sampling_frequency: The sampling frequency of the DataFrame.

        Returns:
            A DataFrame with computed features.
        """
        df = self.resampling(df, sampling_frequency)
        hl_ratio = self.get_hl_ratio(df)
        steps_features = self.get_steps_features(df)
        downsampled = self.downsampling(df)

        n = min(len(hl_ratio), len(steps_features), len(downsampled))
        start = df.index[0].ceil('s')
        df = pd.concat([downsampled, hl_ratio, steps_features], axis=1)
        df = df.iloc[:n]
        df.index = pd.date_range(
            start=start,
            periods=n,
            freq=timedelta(seconds=1),
            name='datetime',
        )
        df['sf'] = sampling_frequency
        logger.info('Features computed.')

        return df

    def compute(self, df: pd.DataFrame, sampling_frequency: float | None = None) -> pd.DataFrame:
        """Computes features from the provided accelerometer data DataFrame.
        This method serves as the main entry point for feature computation. It handles
        pre-processing steps like data format validation, sampling frequency
        determination, and optional calibration before dispatching the computation
        to either a chunked or a non-chunked processing function based on the
        instance's configuration.

        Args:
            df (pd.DataFrame): The input DataFrame containing accelerometer data.
                It is expected to have a time-based index from which the sampling
                frequency can be inferred if not provided.

            sampling_frequency (float | None, optional): The sampling frequency of
                the data in Hertz. If None, it will be automatically calculated
                from the DataFrame's index.

        Returns:
            pd.DataFrame: A new DataFrame containing the computed features.
        """

        df = self.check_format(df)

        sampling_frequency = sampling_frequency or self.get_sampling_frequency(df)

        if self.calibrate:
            df = AutoCalibrate().compute(df, hertz=sampling_frequency)

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
        """Converts raw SENS data to a DataFrame.

        Args:
            timestamps: An array of timestamps.
            data: An array of accelerometer data.

        Returns:
            A DataFrame created from the SENS data.
        """
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
        """Converts a DataFrame to the SENS format.

        Args:
            df: The input DataFrame.

        Returns:
            A tuple containing timestamps and data arrays.
        """
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
