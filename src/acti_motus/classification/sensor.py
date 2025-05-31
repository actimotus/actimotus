import logging
from abc import ABC
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# NON-WEAR BOUTS SETTINGS
SHORT_OFF_BOUTS = 600
LONG_OFF_BOUTS = 5400
ON_BOUTS = 60
DEGREES_TOLERANCE = 5

logger = logging.getLogger(__name__)


@dataclass
class Sensor(ABC):
    vendor: Literal['Sens', 'Other']

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

    def _get_small_bout_max(self, bout, sd_sum):
        start, end, _ = bout
        return sd_sum[start:end].max() > 0.5

    def _fix_off_bouts(self, df: pd.DataFrame) -> pd.Series:
        sd_mean = df[['sd_x', 'sd_y', 'sd_z']].mean(axis=1)
        sd_sum = df[['sd_x', 'sd_y', 'sd_z']].sum(axis=1)
        non_wear = sd_mean < 0.01
        non_wear.name = 'non_wear'

        bouts = (non_wear != non_wear.shift()).cumsum()
        bout_sizes = bouts[non_wear].value_counts()

        large_bouts = bout_sizes[bout_sizes > LONG_OFF_BOUTS].index.values

        small_bouts = bout_sizes[(bout_sizes <= LONG_OFF_BOUTS) & (bout_sizes > SHORT_OFF_BOUTS)]
        small_bouts = bouts[bouts.isin(small_bouts.index)].drop_duplicates(keep='first').reset_index(drop=False)
        small_bouts.rename(columns={'datetime': 'end', 'non_wear': 'bout'}, inplace=True)
        small_bouts.insert(0, 'start', small_bouts['end'] - pd.Timedelta(seconds=5))

        small_bouts['non_wear'] = small_bouts.apply(self._get_small_bout_max, args=(sd_sum,), axis=1)
        small_bouts = small_bouts.loc[small_bouts['non_wear'], 'bout'].values

        non_wear = bouts.isin(large_bouts) | bouts.isin(small_bouts)

        return non_wear

    def _fix_on_bouts(
        self,
        non_wear: pd.DataFrame,
    ) -> pd.Series:
        bouts = (non_wear != non_wear.shift()).cumsum()
        bout_sizes = bouts[~non_wear].value_counts()
        short_on_bouts = bout_sizes[bout_sizes < ON_BOUTS].index.values

        non_wear.loc[bouts.isin(short_on_bouts)] = True

        return non_wear

    def _get_angle_mean(self, df: pd.DataFrame) -> pd.Series:
        mean = df.mean(axis=0)

        rule_1 = (abs(mean - [90, 0, 90]) < DEGREES_TOLERANCE).all()
        rule_2 = (abs(mean - [90, 0, -90]) < DEGREES_TOLERANCE).all()

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
        non_wear = bouts.isin(large_bouts) | bouts.isin(other_bouts)

        return non_wear

    def get_non_wear(self, df: pd.DataFrame) -> pd.DataFrame:
        non_wear = self._fix_off_bouts(df)
        non_wear = self._fix_on_bouts(non_wear)
        non_wear = self._fix_off_bouts_angles(non_wear, df)

        ratio = non_wear.value_counts(normalize=True)
        total_time = (df.index[-1] - df.index[0]).floor('s')
        non_wear_time = pd.Timedelta(seconds=ratio[True] * total_time.total_seconds()).floor('s')
        non_wear_percentage = ratio[True] * 100

        logger.info(
            f'Non-wear detection: {non_wear_time} ({non_wear_percentage:.2f}%) out of {total_time} classified as non-wear time.'
        )

        return non_wear
