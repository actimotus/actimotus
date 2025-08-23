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
    min_hours: int = 72

    def compute(
        self,
        df: pd.DataFrame,
        hertz: float,
        **kwargs: Any,
    ) -> pd.DataFrame:
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
