import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .references import Angle, AngleStatus, Calibration

# NON-WEAR BOUTS SETTINGS
SHORT_OFF_BOUTS = 600
LONG_OFF_BOUTS = 5400
ON_BOUTS = 60
DEGREES_TOLERANCE = 5

logger = logging.getLogger(__name__)


class Calculation(Enum):
    AUTOMATIC = 'automatic'
    DEFAULT = 'default'


@dataclass
class Sensor(ABC):
    orientation: bool

    def get_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        axes = df[['x', 'y', 'z']].to_numpy()

        euclidean_distance = np.linalg.norm(axes, axis=1)
        euclidean_distance = np.where(euclidean_distance == 0, np.nan, euclidean_distance)
        inclination = np.arccos(axes[:, 0] / euclidean_distance)
        side_tilt = -np.arcsin(axes[:, 1] / euclidean_distance)
        direction = -np.arcsin(axes[:, 2] / euclidean_distance)

        angles = np.column_stack((inclination, side_tilt, direction))
        angles = np.degrees(angles).astype(np.float32)

        return pd.DataFrame(
            angles,
            columns=['inclination', 'side_tilt', 'direction'],
            index=df.index,
        )

    def _get_small_bout_max(self, bout: pd.DataFrame, sd_sum: pd.Series) -> bool:
        start, end, _ = bout
        return sd_sum[start:end].max() > 0.5

    def _fix_off_bouts(self, df: pd.DataFrame) -> pd.Series:
        sd_mean = df[['sd_x', 'sd_y', 'sd_z']].mean(axis=1)
        sd_sum = df[['sd_x', 'sd_y', 'sd_z']].sum(axis=1)
        non_wear = sd_mean < 0.01
        non_wear.name = 'non-wear'

        bouts = (non_wear != non_wear.shift()).cumsum()
        bout_sizes = bouts[non_wear].value_counts()

        large_bouts = bout_sizes[bout_sizes > LONG_OFF_BOUTS].index.values
        small_bouts = bout_sizes[(bout_sizes <= LONG_OFF_BOUTS) & (bout_sizes > SHORT_OFF_BOUTS)].index.values

        if small_bouts.size > 0:
            small_bouts = bouts[bouts.isin(small_bouts)].drop_duplicates(keep='first').reset_index(drop=False)
            small_bouts.rename(columns={'datetime': 'end', 'non-wear': 'bout'}, inplace=True)
            small_bouts['end'] = small_bouts['end'] - pd.Timedelta(seconds=10)  # X seconds before start of short bout
            small_bouts.insert(
                0, 'start', small_bouts['end'] - pd.Timedelta(seconds=5)
            )  # Y seconds before end of short bout

            small_bouts['non-wear'] = small_bouts.apply(lambda x: self._get_small_bout_max(x, sd_sum), axis=1)
            small_bouts = small_bouts.loc[small_bouts['non-wear'], 'bout'].values

        non_wear = bouts.isin(large_bouts) | bouts.isin(small_bouts)

        return non_wear

    def _fix_on_bouts(
        self,
        non_wear: pd.Series,
    ) -> pd.Series:
        bouts = (non_wear != non_wear.shift()).cumsum()
        bout_sizes = bouts[~non_wear].value_counts()
        short_on_bouts = bout_sizes[bout_sizes < ON_BOUTS].index.values

        non_wear.loc[bouts.isin(short_on_bouts)] = True

        return non_wear

    def _get_angle_mean(self, df: pd.DataFrame) -> bool:
        mean = df.mean(axis=0)

        rule_1 = (abs(mean - [90, 0, 90]) < DEGREES_TOLERANCE).all().astype(bool)
        rule_2 = (abs(mean - [90, 0, -90]) < DEGREES_TOLERANCE).all().astype(bool)

        return rule_1 or rule_2

    def _fix_off_bouts_angles(
        self,
        non_wear: pd.Series,
        df: pd.DataFrame,
    ) -> pd.Series:
        bouts = (non_wear != non_wear.shift()).cumsum()
        bout_sizes = bouts[non_wear].value_counts()

        large_bouts = bout_sizes[bout_sizes > LONG_OFF_BOUTS].index.values
        other_bouts = bout_sizes[bout_sizes <= LONG_OFF_BOUTS].index.values

        other = bouts[bouts.isin(other_bouts)]

        other_bouts = df.groupby(other)[['inclination', 'side_tilt', 'direction']].apply(self._get_angle_mean)
        other_bouts = other_bouts[
            other_bouts
        ].index.values  # FIXME: Should this use FALSE or TRUE bouts from _get_angle_mean?

        non_wear = bouts.isin(large_bouts) | bouts.isin(other_bouts)

        return non_wear

    def get_non_wear(self, df: pd.DataFrame) -> pd.Series:
        non_wear = self._fix_off_bouts(df)
        non_wear = self._fix_on_bouts(non_wear)
        non_wear = self._fix_off_bouts_angles(non_wear, df)

        ratio = non_wear.value_counts(normalize=True).to_dict()
        total_time = (df.index[-1] - df.index[0]).floor('s')
        non_wear_ratio = ratio.get(True)

        if non_wear_ratio:
            non_wear_time = pd.Timedelta(seconds=non_wear_ratio * total_time.total_seconds()).floor('s')
            non_wear_percentage = non_wear_ratio * 100
        else:
            non_wear_time = pd.Timedelta(seconds=0)
            non_wear_percentage = 0

        logger.info(
            f'Non-wear detection: {non_wear_time} ({non_wear_percentage:.2f}%) out of {total_time} classified as non-wear time.'
        )

        return non_wear

    def fix_sensor_orientation(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df.copy()
        upside_down = self.check_upside_down_flip(df)
        inside_out = self.check_inside_out_flip(df)
        columns, text = None, None

        if upside_down and inside_out:
            columns = ['x', 'z', 'sum_x', 'sum_z', 'sum_dot_xz', 'direction']
            text = 'upside down and inside out'

        elif inside_out:
            columns = ['y', 'z', 'sum_y', 'sum_z', 'sum_dot_xz', 'side_tilt', 'direction']
            text = 'inside out'

        elif upside_down:
            columns = ['x', 'y', 'sum_x', 'sum_y', 'sum_dot_xz', 'side_tilt']
            text = 'upside down'

        if columns:
            df[columns] = -df[columns]
            df[['inclination', 'side_tilt', 'direction']] = self.get_angles(df)

            logger.warning(f'Sensor is {text}. Data flipped {columns} and angles recalculated.')
        else:
            logger.info('No sensor flip detected. No changes made to the data.')

        return df

    def check_upside_down_flip(self, df: pd.DataFrame) -> bool:
        valid_points = df[(df['inclination'] < 45) | (df['inclination'] > 135)]

        if valid_points.empty:
            logger.warning('Not enough data to check upside down flip. Skipping.')
            return False

        mdn = np.median(valid_points['x'])
        flip = True if (mdn < -0.1) else False

        if flip:
            logger.warning(f'Upside down flip detected (median x: {mdn:.2f}).')

        return flip

    @abstractmethod
    def check_inside_out_flip(self, df: pd.DataFrame) -> bool:
        pass

    @abstractmethod
    def rotate_by_reference_angle(self, df: pd.DataFrame, angle: float | np.ndarray) -> pd.DataFrame:
        pass

    @abstractmethod
    def calculate_reference_angle(self, df: pd.DataFrame) -> tuple[float | np.ndarray, Calculation]:
        pass

    def fix_bouts_orientation(self, df: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for start, end, non_wear, angle, calibration in bouts.to_numpy():
            if not non_wear:
                bout_df = df[start:end]
                df.loc[df.index.isin(bout_df.index)] = self.fix_sensor_orientation(bout_df)

        return df

    def rotate_bouts_by_reference_angles(
        self, df: pd.DataFrame, bouts: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        bouts = bouts.copy()
        angles = []
        prev_angle_value = None

        for start, end, non_wear, calibration, angle in bouts.to_numpy():
            bout_df = df[start:end]

            if angle:
                angle.status = AngleStatus.PROPAGATED
                prev_angle_value = angle.value

            elif isinstance(calibration, Calibration):
                calibration_period = calibration.get_calibration_period(bout_df)
                value, status = self.calculate_reference_angle(calibration_period)
                status = AngleStatus.CALIBRATION if status == Calculation.AUTOMATIC else AngleStatus.DEFAULT
                expires = calibration.expires or None
                angle = Angle(value, expires, status)
                prev_angle_value = angle.value

            else:
                if non_wear and prev_angle_value:
                    value, expires, status = prev_angle_value, None, AngleStatus.DEFAULT
                else:
                    value, status = self.calculate_reference_angle(bout_df)
                    expires = None

                angle = Angle(value, expires, status)
                prev_angle_value = angle.value

            angles.append(angle)
            df.loc[bout_df.index] = self.rotate_by_reference_angle(bout_df, angle.value)

        bouts['angle'] = pd.Series(angles, dtype=object)

        return df, bouts

    def fix_bouts(
        self,
        activities: pd.Series,
        activity: str,
        bouts_length: int,
    ) -> pd.Series:
        df = activities.to_frame('activity')
        df['bout'] = (df['activity'] != df['activity'].shift()).cumsum()

        specific_activity = df.loc[df['activity'] == activity, 'bout']
        bout_sizes = specific_activity.value_counts()
        short_bouts = bout_sizes[bout_sizes <= bouts_length].index.values

        df['new_activity'] = df['activity']
        df.loc[df['bout'].isin(short_bouts), 'new_activity'] = np.nan

        half = len(short_bouts) // 2
        half = max(half, 1)

        df['activity'] = df['new_activity'].ffill(limit=half).bfill(limit=half)

        return df['activity']
