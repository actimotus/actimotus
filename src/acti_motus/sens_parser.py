from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class FileParser:
    """Abstract base class for file parsers.

    Provides methods for checking file extensions and existence.
    """

    def check_extension(self, path: Path, extension: str) -> None:
        """Check if the file has the expected extension."""
        if isinstance(path, str):
            path = Path(path)

        if not path.suffix == extension:
            raise ValueError(f'Invalid file type: {path.suffix}. Expected {extension}.')

    def check_existence(self, path: Path) -> None:
        """Check if the file exists and is a file."""
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f'File not found: {path}.')

        if not path.is_file():
            raise ValueError(f'Path is not a file: {path}.')

    def check_file(self, path: Path, extension: str) -> None:
        """Check if the file exists and has the expected extension."""
        if isinstance(path, str):
            path = Path(path)

        self.check_existence(path)
        self.check_extension(path, extension)

    def check_empty(self, df: pd.DataFrame) -> None:
        """Check if the DataFrame is empty."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame.')

        if df.empty:
            raise ValueError('No data found in the file.')


SENS_NORMALIZATION_FACTOR = -4 / 512
DTYPE = np.dtype([('timestamp', '6uint8'), ('x', '>i2'), ('y', '>i2'), ('z', '>i2')])


@dataclass
class Sens(FileParser):
    normalize: bool = True

    def _read(
        self,
        obj: Path | bytes,
        func: Callable,
    ) -> pd.DataFrame:
        data = func(obj, dtype=DTYPE, count=-1, offset=0)
        timestamps = np.dot(data['timestamp'], [1 << 40, 1 << 32, 1 << 24, 1 << 16, 1 << 8, 1])

        df = pd.DataFrame(
            {
                'datetime': pd.to_datetime(timestamps, unit='ms', utc=True),
                'acc_x': data['x'].astype(np.int16),
                'acc_y': data['y'].astype(np.int16),
                'acc_z': data['z'].astype(np.int16),
            }
        )

        self.check_empty(df)

        df.set_index('datetime', inplace=True)

        if self.normalize:
            df = df * SENS_NORMALIZATION_FACTOR

        return df.astype(np.float32)

    def from_bin(self, path: str | Path) -> pd.DataFrame:
        if isinstance(path, str):
            path = Path(path)

        self.check_file(path, '.bin')

        return self._read(path, np.fromfile)

    def from_buffer(self, buffer: bytes) -> pd.DataFrame:
        if not isinstance(buffer, bytes):
            raise TypeError('Expected a bytes object.')

        return self._read(buffer, np.frombuffer)
