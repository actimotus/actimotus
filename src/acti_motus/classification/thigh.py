import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..settings import SYSTEM_SF
from .sensor import Sensor

logger = logging.getLogger(__name__)


@dataclass
class Thigh(Sensor):
    def check_inside_out_flip(self, df: pd.DataFrame) -> bool:
        rows_per_hour = SYSTEM_SF * 60 * 2
        window = rows_per_hour * 3
        step = rows_per_hour
        min_periods = rows_per_hour

        inclination = df['inclination']
        z = df['z']

        valid_windows = inclination.rolling(window=window, step=step, min_periods=min_periods).quantile(0.02) <= 45

        valid_points_mask = (
            pd.Series(index=df.index, dtype=bool).reindex(valid_windows.index, method='ffill').fillna(False)
        )

        valid_points = z.loc[valid_points_mask & (inclination > 45)]

        if valid_points.empty:
            logger.warning('Not enough data to check inside out flip. Skipping.')
            return False

        mdn = np.median(valid_points)
        flip = True if mdn > 0 else False

        logger.info(f"Inside out flip {'detected' if flip else 'not detected'} (median z: {mdn:.2f}).")

        return flip

    def check_upside_down_flip(self, df: pd.DataFrame) -> bool:
        valid_points = df[(df['inclination'] < 45) | (df['inclination'] > 135)]

        if valid_points.empty:
            logger.warning('Not enough data to check upside down flip. Skipping.')
            return False

        mdn = np.median(valid_points['x'])
        flip = True if mdn < 0 else False

        logger.info(f"Upside down flip {'detected' if flip else 'not detected'} (median x: {mdn:.2f}).")

        return flip
