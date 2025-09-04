import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .references import References
from .sensor import Calculation, Sensor

logger = logging.getLogger(__name__)


@dataclass
class Trunk(Sensor):
    config: dict[str, Any]

    def _reference_angle_string(self, angle: np.ndarray) -> str:
        return f'[{angle[0]:.2f}, {angle[1]:.2f}, {angle[2]:.2f}]'

    def calculate_reference_angle(self, df: pd.DataFrame) -> tuple[np.ndarray, Calculation]:
        default_angle = np.array([27, 27, 0])
        angle_status = Calculation.DEFAULT

        walk = df[(~df['non-wear']) & (df['activity'] == 'walk')]

        if not walk.empty:
            y = np.median(walk['direction']) - 6
            z = np.median(walk['side_tilt'])
            x = np.degrees(np.arccos(np.radians(y)) * np.cos(np.radians(z)))

            reference_angle = np.array([x, y, z])
            angle_status = Calculation.AUTOMATIC

            logger.info(f'Reference angle calculated: {self._reference_angle_string(reference_angle)} degrees.')
        else:
            reference_angle = default_angle
            logger.info(
                f'No walking data available. Using default reference angle: {self._reference_angle_string(reference_angle)} degrees.'
            )

        return np.float32(np.radians(reference_angle)), angle_status  # type: ignore

    def rotate_by_reference_angle(self, df: pd.DataFrame, angle: np.ndarray) -> pd.DataFrame:  # type: ignore
        df = df.copy()
        x, y, z = angle

        cos_y, sin_y = np.cos(y), np.sin(y)
        cos_z, sin_z = np.cos(z), np.sin(z)

        rotation_y = np.array(
            [
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y],
            ]
        )
        rotation_z = np.array(
            [
                [cos_z, sin_z, 0],
                [-sin_z, cos_z, 0],
                [0, 0, 1],
            ]
        )
        rotation_matrix = np.matmul(rotation_y, rotation_z)  # type: np.ndarray

        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].dot(rotation_matrix).astype(np.float32)
        df[['inclination', 'side_tilt', 'direction']] = self.get_angles(df)

        return df

    def check_inside_out_flip(self, df: pd.DataFrame) -> bool:
        valid_points = df[(df['inclination'] < 45) | (df['inclination'] > 135)]  # Standing

        if valid_points.empty:
            logger.warning('Not enough data to check inside out flip. Skipping.')
            return False

        mdn = np.median(valid_points['direction'])
        flip = False if (mdn > 0.1) else True

        if flip:
            logger.warning(f'Inside out flip detected (median z: {mdn:.2f}).')

        return flip

    def get_backwards(
        self,
        df: pd.DataFrame,
        inclination_angle: float,
        **kwargs,
    ) -> pd.Series:
        backwards = (df['direction'] < -inclination_angle) | (df['side_tilt'].abs() > inclination_angle)
        backwards.name = 'backwards'

        return backwards

    def fix_lie(
        self,
        df: pd.DataFrame,
        orientation_angle: float,
        **kwargs,
    ) -> pd.Series:
        valid = df[['activity']].copy()
        mask = ~df['non-wear']  # Only consider wear time
        valid.loc[mask & (valid['activity'] == 'lie'), 'activity'] = 'sit'  # Change every lie to sit
        valid.loc[mask & (valid['activity'] == 'sit') & (df['inclination'] > orientation_angle), 'activity'] = 'lie'
        valid.loc[mask & (valid['activity'] == 'sit') & (df['backwards']), 'activity'] = 'lie'

        return valid['activity']

    def fix_sit(
        self,
        df: pd.DataFrame,
        orientation_angle: float,
        inclination_angle: float,
        **kwargs,
    ) -> pd.Series:
        valid = df[['activity']].copy()
        mask = (~df['non-wear']) & (valid['activity'] == 'lie')  # Only consider wear time and lie activity
        valid.loc[mask & (df['direction'] > 0) & (df['thigh_direction'] > inclination_angle), 'activity'] = 'sit'

        valid.loc[
            mask
            & ((df['inclination'] - df['direction']).abs() < 10)
            & (df['inclination'] < orientation_angle)
            & (df['direction'] < orientation_angle),
            'activity',
        ] = 'sit'

        return valid['activity']

    def compute_activities(
        self,
        df: pd.DataFrame,
        activities: pd.DataFrame,
        references: References,
    ) -> pd.DataFrame:
        thigh_config = self.config['thigh']
        config = self.config['trunk']
        bouts_length = {activity[0]: activity[1]['bout'] for activity in thigh_config.items()}

        activities = activities.copy()
        df = df.copy()

        # Only keep overlapping data (thigh and trunk)
        df = df.join(activities, how='inner')

        df[['inclination', 'side_tilt', 'direction']] = self.get_angles(df)
        df['non-wear'] = self.get_non_wear(df)
        bouts = references.get_bouts(df['non-wear'], 'trunk')

        if self.orientation:
            df = self.fix_bouts_orientation(df, bouts)

        df, bouts = self.rotate_bouts_by_reference_angles(df, bouts)

        df['backwards'] = self.get_backwards(df, **config['lie'])
        df['activity'] = self.fix_lie(df, **config['lie'])
        df['activity'] = self.fix_sit(df, **config['lie'])

        for activity in ['sit', 'lie']:
            df['activity'] = self.fix_bouts(df['activity'], activity, bouts_length[activity])

        df.loc[df['non-wear'], 'inclination'] = np.nan
        df.loc[df['non-wear'], 'direction'] = np.nan

        df.rename(
            columns={
                'inclination': 'trunk_inclination',
                'side_tilt': 'trunk_side_tilt',
                'direction': 'trunk_direction',
            },
            inplace=True,
        )

        activities = activities.join(df[['trunk_inclination', 'trunk_side_tilt', 'trunk_direction']], how='left')
        activities.loc[activities.index.isin(df.index), 'activity'] = df['activity']

        references.update_angle(bouts, 'trunk')

        return activities
