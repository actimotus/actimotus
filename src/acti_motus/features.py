from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from numpy.fft import fft as np_fft
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

SYSTEM_SF = 30
NYQUIST_FREQ = SYSTEM_SF / 2


@dataclass
class Features:
    calibrate: bool = False
    sampling_frequency: float | None = None
    backend: bool = False

    @staticmethod
    def get_sampling_frequency(
        df: pd.DataFrame,
        *,
        samples: int | None = 5_000,
    ) -> float:
        time = df.index

        if samples:
            time = time[:samples]

        return 1 / np.mean(np.diff(time.astype(int) / 1e9))

    def upsample(self, df: pd.DataFrame) -> pd.DataFrame:
        sf = self.sampling_frequency or self.get_sampling_frequency(df)

        n = len(df)
        periods = int(SYSTEM_SF * np.fix(n / sf))

        # arr = df.to_numpy()
        # upsampled = signal.resample(arr, periods, axis=0)

        # df = pd.DataFrame(
        #     upsampled,
        #     columns=df.columns,
        #     index=pd.date_range(start=df.index[0], end=df.index[-1], periods=periods),
        # )

        datetimes = pd.date_range(start=df.index[0], end=df.index[-1], periods=periods)
        df = pd.DataFrame(
            {col: np.interp(datetimes, df.index, df[col]) for col in df.columns},
            index=datetimes,
        )

        return df

    def get_hl_ratio(self, df: pd.DataFrame) -> pd.Series:
        order = 3
        cut_off = 1
        window = SYSTEM_SF * 4

        cut_off = cut_off / NYQUIST_FREQ

        axis_z = df["acc_z"].to_numpy()

        b, a = signal.butter(order, cut_off, "low")
        low = signal.filtfilt(b, a, axis_z, axis=0)
        low = np.abs(low.astype(np.float32))

        b, a = signal.butter(order, cut_off, "high")
        high = signal.filtfilt(b, a, axis_z, axis=0)
        high = np.abs(high.astype(np.float32))

        pad_width = window - 1
        high = np.pad(high, (0, pad_width), mode="edge")
        low = np.pad(low, (0, pad_width), mode="edge")

        high_windows = sliding_window_view(high, window_shape=window)[::SYSTEM_SF]
        mean_high = np.mean(high_windows, axis=1, dtype=np.float32)

        low_windows = sliding_window_view(low, window_shape=window)[::SYSTEM_SF]
        mean_low = np.mean(low_windows, axis=1, dtype=np.float32)

        hl_ratio = np.divide(
            mean_high, mean_low, out=np.zeros_like(mean_high), where=mean_low != 0
        )  # NOTE: Check what happens if mean_low is zero

        return pd.Series(hl_ratio, name="hl_ratio")

    @staticmethod
    def _get_steps_feature(arr: np.ndarray) -> np.ndarray:
        window = SYSTEM_SF * 4
        fft_size = 480
        half_size = fft_size // 2
        arr = arr.astype(np.float32)

        pad_width = window - 1
        arr = np.pad(arr, (0, pad_width), mode="edge")

        windows = sliding_window_view(arr, window)[::SYSTEM_SF]
        windows = windows - np.mean(windows, axis=1, keepdims=True)

        fft_result = np_fft(windows, fft_size)[:, :half_size]
        magnitudes = 2 * np.abs(fft_result)

        return np.argmax(magnitudes, axis=1)

    def get_steps_features(self, df: pd.DataFrame) -> pd.DataFrame:
        axis_x = df["acc_x"].to_numpy()

        b, a = signal.butter(6, 2.5 / NYQUIST_FREQ, "low")
        filtered = signal.lfilter(b, a, axis_x, axis=0)

        b, a = signal.butter(6, 1.5 / NYQUIST_FREQ, "high")
        walk = signal.lfilter(b, a, filtered, axis=0)

        b, a = signal.butter(6, 3 / NYQUIST_FREQ, "high")
        run = signal.lfilter(b, a, walk)

        df = pd.DataFrame(
            {
                "walk_feature": self._get_steps_feature(walk),
                "run_feature": self._get_steps_feature(run),
            },
        )

        return df

    @staticmethod
    def get_tensor(arr: np.ndarray) -> np.ndarray:
        pb = np.vstack((arr[:SYSTEM_SF], arr))
        pa = np.vstack((arr, arr[-SYSTEM_SF:]))
        n = pb.shape[0] // SYSTEM_SF
        tensor = np.concatenate(
            [
                pb[: n * SYSTEM_SF].reshape(SYSTEM_SF, n, 3, order="F"),
                pa[: n * SYSTEM_SF].reshape(SYSTEM_SF, n, 3, order="F"),
            ],
            axis=0,
        )
        return tensor[:, :-1, :]

    def downsample(self, df: pd.DataFrame) -> pd.DataFrame:
        axes = df.to_numpy()

        b, a = signal.butter(4, 5 / NYQUIST_FREQ, "low")
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
                "x",
                "y",
                "z",
                "sd_x",
                "sd_y",
                "sd_z",
                "sum_x",
                "sum_y",
                "sum_z",
                "sq_sum_x",
                "sq_sum_y",
                "sq_sum_z",
            ],
        )
        df["sum_dot_xz"] = sum_dot_xz

        return df

    @staticmethod
    def check_format(df: pd.DataFrame) -> bool:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if df.empty:
            raise ValueError("DataFrame cannot be empty.")

        required_columns = {"acc_x", "acc_y", "acc_z"}
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise ValueError(
                f"DataFrame must contain columns: {list(required_columns)}. "
                f"Missing: {list(missing_cols)}."
            )

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError(
                f"DataFrame index must be of datetime type, but got {df.index.dtype}."
            )

        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Column '{col}' must contain numeric data, but got {df[col].dtype}."
                )
        return True

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        self.check_format(df)

        df = self.upsample(df)
        hl_ratio = self.get_hl_ratio(df)
        steps_features = self.get_steps_features(df)
        downsampled = self.downsample(df)

        start = df.index[0].ceil("s")
        df = pd.concat([downsampled, hl_ratio, steps_features], axis=1)
        df.index = pd.date_range(
            start=start,
            periods=len(df),
            freq=timedelta(seconds=1),
        )

        return df
