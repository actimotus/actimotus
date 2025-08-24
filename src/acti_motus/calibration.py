import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from skdh.preprocessing import CalibrateAccelerometer

logger = logging.getLogger(__name__)


@dataclass
class AutoCalibrate:
    min_hours: int = 24

    def compute(
        self,
        df: pd.DataFrame,
        hertz: float,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Performs accelerometer calibration on the input data.
        This method uses the `CalibrateAccelerometer` class from `skdh.preprocessing` library to correct for biases
        and scaling errors in raw accelerometer signals. It prepares the data,
        runs the calibration, and returns a new DataFrame with the calibrated values.
        If the calibration process fails, it logs a warning and returns the
        original, uncalibrated data.

        Args:
            df (pd.DataFrame): The input DataFrame containing raw accelerometer data.
                The index should be a datetime-like object representing time, and
                the columns should represent the accelerometer axes (e.g., 'x', 'y', 'z').
            hertz (float): The sampling frequency of the accelerometer data in Hertz.
            **kwargs (Any): Additional keyword arguments to be passed to the
                `CalibrateAccelerometer` initializer.

        Returns:
            pd.DataFrame: A DataFrame with the same shape and index as the input,
                containing the calibrated accelerometer data as float32 type. If
                calibration fails, the original data is returned.
        """

        columns = df.columns

        # Prepare data for calibration
        time = df.index
        accel = df.values
        del df

        # Initialize calibrator
        calibrator = CalibrateAccelerometer(min_hours=self.min_hours, **kwargs)

        # Perform calibration
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            calibrated = calibrator.predict(
                time=(time.astype(np.int64) // 10**9).values,
                accel=accel,
                fs=hertz,
            ).get('accel')

        if calibrated is None:
            logger.warning('Calibration did not produce valid results. Returning original accelerometer data.')
        else:
            logger.info('Calibration completed successfully.')
            accel = calibrated
            del calibrated

        # Create result DataFrame with calibrated accelerometer data
        return pd.DataFrame(accel, columns=columns, index=time, dtype=np.float32)
