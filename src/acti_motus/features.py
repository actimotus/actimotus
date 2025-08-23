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
    system_frequency: float = 30
    validation: bool = True
    chunking: bool = False
    size: timedelta = timedelta(hours=24)
    overlap: timedelta = timedelta(minutes=15)
    calibrate: bool = False

    def __post_init__(self):
        if isinstance(self.size, str):
            self.size = pd.Timedelta(self.size).to_pytimedelta()

        if isinstance(self.overlap, str):
            self.overlap = pd.Timedelta(self.overlap).to_pytimedelta()

    def get_nyquist_freq(self, sampling_frequency: float) -> float:
        return sampling_frequency / 2

    @staticmethod
    def get_sampling_frequency(
        df: pd.DataFrame,
        *,
        samples: int | None = 30_000,
    ) -> float:
        time = df.index

        if samples:
            time = time[:samples]

        sf = (1 / np.mean(np.diff(time.astype(int) / 1e9))).item()

        logging.info(f'Detected sampling frequency: {sf:.2f} Hz.', extra={'sampling_frequency': sf})

        return sf

    def _resample_fft(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if math.isclose(sampling_frequency, self.system_frequency, abs_tol=tolerance):
            logger.info(
                f'Sampling frequency is {self.system_frequency} Hz, no resampling needed.',
                extra={'sampling_frequency': sampling_frequency},
            )
            return df

        df = self._resample_fft(df)

        return df

    def get_hl_ratio(self, df: pd.DataFrame) -> pd.Series:
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
        not_overlaps = df[~df['overlap']]
        start, end = not_overlaps.index[0], not_overlaps.index[-1]

        df = self._compute(
            df,
            **kwargs,
        )
        df = df.loc[(df.index >= start) & (df.index < end)]

        return df

    def _compute_chunks(self, df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
        chunks = DataFrameIterator(df, size=self.size, overlap=self.overlap)
        computed = []

        for chunk in chunks:
            computed.append(self._compute_chunk(chunk, sampling_frequency=sampling_frequency))

        computed = pd.concat(computed)
        computed.sort_index(inplace=True)

        return computed

    def _compute(self, df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
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
        df = self._raw_from_sens(timestamps[0], data[0])
        features = self.compute(df)

        return self.to_sens(features)
