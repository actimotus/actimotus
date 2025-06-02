import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..settings import SYSTEM_SF
from .sensor import Calculation, Sensor

logger = logging.getLogger(__name__)


@dataclass
class Thigh(Sensor):
    # rotate: bool = False # TODO: Implement rotation logic first.

    def check_inside_out_flip(self, df: pd.DataFrame) -> bool:
        rows_per_hour = SYSTEM_SF * 60 * 2
        window = rows_per_hour * 3
        step = rows_per_hour
        min_periods = rows_per_hour

        inclination = df['inclination']
        z = df['z']

        valid_windows = inclination.rolling(window=window, step=step, min_periods=min_periods).quantile(0.02) <= 45

        valid_points_mask = pd.Series(df.index.map(valid_windows), index=df.index, dtype='boolean').ffill()
        valid_points = z.loc[valid_points_mask & (inclination > 45)]

        if valid_points.empty:
            logger.warning('Not enough data to check inside out flip. Skipping.')
            return False

        mdn = np.median(valid_points)
        flip = True if mdn > 0 else False

        if flip:
            logger.warning(f'Inside out flip detected (median z: {mdn:.2f}).')

        return flip

    def check_upside_down_flip(self, df: pd.DataFrame) -> bool:
        valid_points = df[(df['inclination'] < 45) | (df['inclination'] > 135)]

        if valid_points.empty:
            logger.warning('Not enough data to check upside down flip. Skipping.')
            return False

        mdn = np.median(valid_points['x'])
        flip = True if mdn < 0 else False

        if flip:
            logger.warning(f'Upside down flip detected (median x: {mdn:.2f}).')

        return flip

    def calculate_reference_angle(self, df: pd.DataFrame) -> dict[float, Calculation]:
        x_threshold_lower = 0.1
        x_threshold_upper = 0.72  # NOTE: Originally 0.7. To match the walk.py, 0.72 should be used.
        inclination_threshold = 45  # NOTE: Same as stationary_threshold for walking.
        direction_threshold = 10

        angle_mdn_coefficient = 6
        angle_threshold_lower = -30  # FIXME: In new code this is -30, original: -28
        angle_threshold_upper = 15
        default_angle = -16
        angle_status = Calculation.DEFAULT

        walk_mask = (
            (df['sd_x'].between(x_threshold_lower, x_threshold_upper, inclusive='neither'))
            & (df['inclination'] < inclination_threshold)
            & (df['direction'] < direction_threshold)
        )
        walk = df[walk_mask]

        if not walk.empty:
            reference_angle = (
                np.median(walk['direction']) - angle_mdn_coefficient
            ).item()  # Walk direction reference angle (median, degrees)

            reference_angle = reference_angle * 0.725 - 5.569  # Correction factor based on RAW data.

            if (reference_angle < angle_threshold_lower) or (reference_angle > angle_threshold_upper):
                logger.warning(
                    f'Reference angle {reference_angle:.2f} degrees is outside the threshold range. Using manual reference angle: {default_angle:.2f} degrees.'
                )
            else:
                angle_status = Calculation.AUTOMATIC
                logger.info(f'Reference angle calculated: {reference_angle:.2f} degrees.')
        else:
            reference_angle = default_angle
            logger.warning(f'No valid walk data found. Using manual reference angle: {default_angle:.2f} degrees.')

        return np.float32(np.radians(reference_angle)).item(), angle_status

    def _rotate_sd(self, df: pd.DataFrame, angle: float) -> pd.DataFrame:
        sin = np.sin(angle)
        cos = np.cos(angle)

        sq_sin = np.square(sin)
        sq_cos = np.square(cos)

        sq_x = np.square(df['x'])
        sq_z = np.square(df['z'])

        sd = pd.DataFrame(index=df.index)

        sd['terms_x'] = (
            (sq_sin * df['sq_sum_z'])
            + (sq_cos * df['sq_sum_x'])
            + (2 * SYSTEM_SF * sq_x)
            + (2 * sin * df['x'] * df['sum_z'])
            + (-2 * sin * cos * df['sum_dot_xz'])
            + (-2 * cos * df['x'] * df['sum_x'])
        )
        sd.loc[sd['terms_x'] <= 0, 'terms_x'] = 0
        sd['sd_x'] = np.sqrt(1 / (2 * SYSTEM_SF - 1) * sd['terms_x'])

        sd['terms_z'] = (
            (sq_sin * df['sq_sum_x'])
            + (sq_cos * df['sq_sum_z'])
            + (2 * SYSTEM_SF * sq_z)
            + (2 * sin * cos * df['sum_dot_xz'])
            + (-2 * sin * df['z'] * df['sum_x'])
            + (-2 * cos * df['z'] * df['sum_z'])
        )
        sd.loc[sd['terms_z'] <= 0, 'terms_z'] = 0
        sd['sd_z'] = np.sqrt(1 / (2 * SYSTEM_SF - 1) * sd['terms_z'])

        sd['sd_y'] = df['sd_y']

        return sd[['sd_x', 'sd_y', 'sd_z']].astype(np.float32)

    def rotate_by_reference_angle(self, df: pd.DataFrame, angle: float) -> pd.DataFrame:
        df = df.copy()
        angle = np.float32(angle)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotation_matrix = np.array(
            [
                [cos_angle, 0, sin_angle],
                [0, 1, 0],
                [-sin_angle, 0, cos_angle],
            ]
        )

        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].dot(rotation_matrix).astype(np.float32)
        df[['sd_x', 'sd_y', 'sd_z']] = self._rotate_sd(df, angle)  # TODO: Maybe not needed

        df[['inclination', 'side_tilt', 'direction']] = self.get_angles(df)

        return df
