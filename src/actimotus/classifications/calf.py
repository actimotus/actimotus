import logging
from dataclasses import dataclass

import pandas as pd

from .references import References
from .sensor import Sensor

logger = logging.getLogger(__name__)


@dataclass
class Calf(Sensor):
    def get_kneeling(self, df: pd.DataFrame) -> pd.Series:
        kneel = (
            ~df['non-wear']
            & (df['inclination'] >= 84)
            & (df['direction'] > 45)
            & (df['thigh_direction'] > -20)
            & (df['thigh_side_tilt'].abs() < 30)
            & (~df['activity'].isin(['non-wear', 'lie']))
        )

        return kneel

    def get_squatting(self, df: pd.DataFrame) -> pd.Series:
        squat = (
            ~df['non-wear']
            & (df['inclination'].between(32, 84, inclusive='neither'))
            & (df['inclination'] >= -1.5 * df['thigh_inclination'] + 187)
            & (df['direction'] > 0)
            & (~df['activity'].isin(['non-wear', 'lie']))
        )
        return squat

    def compute_activities(
        self,
        df: pd.DataFrame,
        activities: pd.DataFrame,
    ) -> pd.DataFrame:
        activities = activities.copy()
        df = df.copy()

        activities['activity'] = activities['activity'].cat.add_categories(['kneel', 'squat'])
        references = References()

        # Only keep overlapping data (thigh and calf)
        df = df.join(activities, how='inner')

        df[['inclination', 'side_tilt', 'direction']] = self.get_angles(df)
        df['non-wear'] = self.get_non_wear(df)
        bouts = references.get_bouts(df['non-wear'])

        if self.orientation:
            df = self.fix_bouts_orientation(df, bouts)

        df['kneel'] = self.get_kneeling(df)
        df['squat'] = self.get_squatting(df)

        df.loc[df['kneel'], 'activity'] = 'kneel'
        df.loc[df['squat'], 'activity'] = 'squat'

        activities.loc[activities.index.isin(df.index), 'activity'] = df['activity']

        return activities

    def check_inside_out_flip(self, df: pd.DataFrame) -> bool:
        return False

    def rotate_by_reference_angle(self):
        raise NotImplementedError('Calf does not support rotating by reference angle.')

    def calculate_reference_angle(self):
        raise NotImplementedError('Calf does not support calculating reference angles.')
