from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from .settings import ACTIVITIES


@dataclass
class Exposures:
    window: str | timedelta = '1d'

    def __post_init__(self):
        if isinstance(self.window, str):
            self.window = pd.Timedelta(self.window).to_pytimedelta()

    def _get_exposure(self, df: pd.DataFrame, valid: pd.Series, function: str) -> pd.Timedelta | int:
        if function == 'time':
            result = df.loc[valid, 'activity'].count()
            result = pd.Timedelta(result, unit='s')
        elif function == 'count':
            transitions = valid & ~(valid.shift(-1, fill_value=False))
            result = transitions.sum()
        else:
            raise ValueError(f'Unknown function: {function}')

        return result

    def get_bending(
        self,
        df: pd.DataFrame,
        lower: int,
        upper: int,
    ) -> pd.Series:
        valid = (
            df['activity'].isin(['stand', 'shuffle', 'walk', 'run', 'stairs'])
            & (df['trunk_direction'] > 0)
            & (df['trunk_inclination'].between(lower, upper, inclusive='both'))
        )

        return valid

    def get_arm_lifting(
        self,
        df: pd.DataFrame,
        lower: int,
        upper: int,
    ) -> pd.Series:
        valid = df['activity'].isin(['stand', 'shuffle', 'walk']) & (
            df['arm_inclination'].between(lower, upper, inclusive='both')
        )

        return valid

    def get_fast_walking(self, df: pd.DataFrame) -> pd.Series:
        return (df['activity'] == 'walk') & (df['steps'] >= 120 / 60)

    def get_slow_walking(self, df: pd.DataFrame) -> pd.Series:
        return (df['activity'] == 'walk') & (df['steps'] < 120 / 60)

    def _get_exposures(self, df: pd.DataFrame) -> pd.Series:
        exposure = {
            'wear': self._get_exposure(df, df['activity'] != 'non-wear', 'time'),
            'sedentary': self._get_exposure(df, df['activity'].isin(['sit', 'lie']), 'time'),
            'standing': self._get_exposure(df, df['activity'].isin(['stand', 'shuffle']), 'time'),
            'on_feet': self._get_exposure(
                df, df['activity'].isin(['stand', 'shuffle', 'walk', 'runk', 'stairs']), 'time'
            ),
            'sedentary_to_other': self._get_exposure(df, df['activity'].isin(['sit', 'lie']), 'count'),
            'lpa': self._get_exposure(
                df,
                df['activity'].isin(['stand', 'shuffle']) | self.get_slow_walking(df),
                'time',
            ),
            'mvpa': self._get_exposure(
                df,
                df['activity'].isin(['run', 'stairs', 'bicycle', 'row']) | self.get_fast_walking(df),
                'time',
            ),
        }

        if ('trunk_direction' in df.columns) and ('trunk_inclination' in df.columns):
            exposure['bending_30_60'] = self._get_exposure(df, self.get_bending(df, 30, 60), 'time')
            exposure['bending_60_90'] = self._get_exposure(df, self.get_bending(df, 60, 180), 'time')
            exposure['bending_45_180'] = self._get_exposure(df, self.get_bending(df, 45, 180), 'count')

        if 'arm_inclination' in df.columns:
            exposure['arm_lifting_30_60'] = self._get_exposure(df, self.get_arm_lifting(df, 30, 60), 'time')
            exposure['arm_lifting_60_90'] = self._get_exposure(df, self.get_arm_lifting(df, 60, 90), 'time')
            exposure['arm_lifting_90_180'] = self._get_exposure(df, self.get_arm_lifting(df, 90, 180), 'time')

        exposure = pd.Series(exposure)

        return exposure

    def _get_activities(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = ACTIVITIES.values()
        columns = [col for col in columns if col in df['activity'].cat.categories]
        columns.remove('non-wear')

        activities = (
            df['activity'].groupby([pd.Grouper(freq=self.window, sort=True), df['activity']], observed=False).count()  # type: ignore
        )
        activities = activities.apply(pd.Timedelta, unit='s').unstack()
        return activities[columns]

    def compute(self, df: pd.DataFrame, activities: bool = False) -> pd.DataFrame:
        exposure = df.groupby(pd.Grouper(freq=self.window, sort=True)).apply(self._get_exposures)  # type: ignore

        if activities:
            df = self._get_activities(df)
            exposure = pd.concat([exposure, df], axis=1)

        return exposure
