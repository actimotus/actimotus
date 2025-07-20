import importlib.util
import logging
import math
import multiprocessing as mp
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from multiprocessing.pool import ThreadPool as Pool
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.fft import fft as np_fft
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from .iterators import DataFrameIterator
from .settings import FEATURES, RAW, SENS__FLOAT_FACTOR, SENS__NORMALIZATION_FACTOR, SYSTEM_SF

logger = logging.getLogger(__name__)

NYQUIST_FREQ = SYSTEM_SF / 2


@dataclass
class Features:
    sampling_frequency: float | None = None
    validation: bool = True
    chunks: bool = False
    size: timedelta = '1d'
    overlap: timedelta = '15min'
    multithread: int = 'max'
    resample: Literal['fft', 'legacy'] = 'fft'
    calibrate: bool | timedelta = False

    def __post_init__(self):
        self.multithread = mp.cpu_count() if self.multithread == 'max' else int(self.multithread)

        if self.multithread < 1:
            raise ValueError('Number of cores must be at least 1.')

        if isinstance(self.size, str):
            self.size = pd.Timedelta(self.size).to_pytimedelta()

        if isinstance(self.overlap, str):
            self.overlap = pd.Timedelta(self.overlap).to_pytimedelta()

        if self.resample not in {'fft', 'legacy'}:
            raise ValueError("Method must be one of 'fft' or 'legacy'.")

        if self.calibrate:
            if importlib.util.find_spec('labda_accelerometers') is None:
                raise ImportError(
                    "AutoCalibrate is not available. Please install 'labda-accelerometers' package to use calibration."
                )

            self.calibrate = (
                self.size if isinstance(self.calibrate, bool) else pd.Timedelta(self.calibrate).to_pytimedelta()
            )

    @staticmethod
    def get_sampling_frequency(
        df: pd.DataFrame,
        *,
        samples: int | None = 5_000,
    ) -> float:
        time = df.index

        if samples:
            time = time[:samples]

        sf = (1 / np.mean(np.diff(time.astype(int) / 1e9))).item()

        logging.info(f'Detected sampling frequency: {sf:.2f} Hz.', extra={'sampling_frequency': sf})

        return sf

    def _legacy_resample(self, df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
        n_in = len(df)
        n_out = int(SYSTEM_SF * np.fix(n_in / sampling_frequency))

        datetimes = pd.date_range(start=df.index[0], end=df.index[-1], periods=n_out)
        df = pd.DataFrame(
            {col: np.interp(datetimes, df.index, df[col]) for col in df.columns},
            index=datetimes,
        )

        return df

    def _resample_fft(self, df: pd.DataFrame) -> pd.DataFrame:
        start = df.index[0]
        end = df.index[-1]

        n_out = np.floor((end - start).total_seconds() * SYSTEM_SF).astype(int)
        resampled = signal.resample(df, n_out)

        df = pd.DataFrame(
            resampled,
            columns=df.columns,
            index=pd.date_range(start=start, end=end, periods=n_out),
            dtype=np.float32,
        )

        return df

    def resampling(self, df: pd.DataFrame, sampling_frequency: float, tolerance=1) -> pd.DataFrame:
        if math.isclose(sampling_frequency, SYSTEM_SF, abs_tol=tolerance):
            logger.info(
                f'Sampling frequency is {SYSTEM_SF} Hz, no resampling needed.',
                extra={'sampling_frequency': sampling_frequency},
            )
            return df

        match self.resample:
            case 'fft':
                df = self._resample_fft(df)
            case 'legacy':
                df = self._legacy_resample(df, sampling_frequency)
            case _:
                raise ValueError("Method must be one of 'fft' or 'legacy'.")

        return df

    def get_hl_ratio(self, df: pd.DataFrame) -> pd.Series:
        order = 3
        cut_off = 1
        window = SYSTEM_SF * 4
        cut_off = cut_off / NYQUIST_FREQ

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

        high_windows = sliding_window_view(high, window)[::SYSTEM_SF]
        mean_high = np.mean(high_windows, axis=1, dtype=np.float32)

        low_windows = sliding_window_view(low, window)[::SYSTEM_SF]
        mean_low = np.mean(low_windows, axis=1, dtype=np.float32)

        hl_ratio = np.divide(
            mean_high, mean_low, out=np.zeros_like(mean_high), where=mean_low != 0
        )  # NOTE: Check what happens if mean_low is zero

        return pd.Series(hl_ratio, name='hl_ratio')

    def _get_steps_feature(self, arr: np.ndarray) -> np.ndarray:
        window = SYSTEM_SF * 4  # 120 samples equal to 2 seconds
        steps_window = 4 * window  # 480 samples equal to 8 seconds
        half_size = window * 2  # 240 samples equal to 4 seconds
        arr = arr.astype(np.float32)

        pad_width = window - 1
        arr = np.pad(arr, (0, pad_width), mode='edge')

        windows = sliding_window_view(arr, window)[::SYSTEM_SF]
        windows = windows - np.mean(windows, axis=1, keepdims=True, dtype=np.float32)

        fft_result = np_fft(windows, steps_window)[:, :half_size]
        magnitudes = 2 * np.abs(fft_result)

        return np.argmax(magnitudes, axis=1)

    def get_steps_features(self, df: pd.DataFrame) -> pd.DataFrame:
        axis_x = df['acc_x'].values

        b, a = signal.butter(6, 2.5 / NYQUIST_FREQ, 'low')
        filtered = signal.lfilter(b, a, axis_x, axis=0)

        b, a = signal.butter(6, 1.5 / NYQUIST_FREQ, 'high')
        walk = signal.lfilter(b, a, filtered, axis=0)

        b, a = signal.butter(6, 3 / NYQUIST_FREQ, 'high')
        run = signal.lfilter(b, a, walk)

        df = pd.DataFrame(
            {
                'walk_feature': self._get_steps_feature(walk),
                'run_feature': self._get_steps_feature(run),
            },
        )

        return df

    def get_tensor(self, arr: np.ndarray) -> np.ndarray:
        pb = np.vstack((arr[:SYSTEM_SF], arr))
        pa = np.vstack((arr, arr[-SYSTEM_SF:]))
        n = pb.shape[0] // SYSTEM_SF
        tensor = np.concatenate(
            [
                pb[: n * SYSTEM_SF].reshape(SYSTEM_SF, n, 3, order='F'),
                pa[: n * SYSTEM_SF].reshape(SYSTEM_SF, n, 3, order='F'),
            ],
            axis=0,
        )
        return tensor[:, :-1, :]

    def downsampling(self, df: pd.DataFrame) -> pd.DataFrame:
        axes = df.values

        b, a = signal.butter(4, 5 / NYQUIST_FREQ, 'low')
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
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input must be a pandas DataFrame.')

        if df.empty:
            raise ValueError('DataFrame cannot be empty.')

        required_columns = {'acc_x', 'acc_y', 'acc_z'}
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise ValueError(
                f'DataFrame must contain columns: {list(required_columns)}. Missing: {list(missing_cols)}.'
            )

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError(f'DataFrame index must be of datetime type, but got {df.index.dtype}.')

        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must contain numeric data, but got {df[col].dtype}.")

        return df[['acc_x', 'acc_y', 'acc_z']]

    def _extract_chunk(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        not_overlaps = df[~df['overlap']]
        start, end = not_overlaps.index[0], not_overlaps.index[-1]

        df = self._extract(
            df[['acc_x', 'acc_y', 'acc_z']],
            **kwargs,
        )
        df = df.loc[(df.index >= start) & (df.index < end)]

        return df

    def _extract_chunks(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        chunks = DataFrameIterator(df, size=self.size, overlap=self.overlap)

        with Pool(self.multithread) as pool:
            output = pool.map(
                partial(self._extract_chunk, **kwargs),
                chunks,
            )

        output = pd.concat(output)
        output.sort_index(inplace=True)

        return output

    def _extract(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if self.validation:
            df = self.check_format(df)

        df = df.iloc[:, :3]
        df.columns = ['acc_x', 'acc_y', 'acc_z']

        sf = self.sampling_frequency or self.get_sampling_frequency(df)

        if self.calibrate:
            from labda_accelerometers import AutoCalibrate

            df = AutoCalibrate(min_hours=int(self.calibrate.total_seconds() // 3600), sampling_frequency=sf).calibrate(
                df
            )

        df = self.resampling(df, sf)
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
        df['sf'] = sf
        logger.info('Features extracted.')

        return df

    def extract(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if self.chunks:
            return self._extract_chunks(df, **kwargs)
        else:
            return self._extract(df, **kwargs)

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
            columns=RAW,
        )

        df = df * SENS__NORMALIZATION_FACTOR
        df.index = pd.to_datetime(df.index, unit='ms')  # type: ignore
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

    def extract_sens(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = self._raw_from_sens(timestamps[0], data[0])
        features = self.extract(df)

        return self.to_sens(features)
