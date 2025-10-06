from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import altair as alt
import pandas as pd

from .settings import ACTIVITIES, FUSED_ACTIVITIES

alt.data_transformers.enable('vegafusion')


PLOT_LANG = {
    'activities': {
        'non-wear': {'text': 'Non-wear', 'color': '#BDBDBD'},
        'lie': {'text': 'Lying', 'color': '#42A5F5'},
        'sit': {'text': 'Sitting', 'color': '#1565C0'},
        'stand': {'text': 'Standing', 'color': '#26A69A'},
        'shuffle': {'text': 'Shuffling', 'color': '#00695C'},
        'walk': {'text': 'Walking', 'color': '#66BB6A'},
        'fast-walk': {'text': 'Fast walking', 'color': '#2E7D32'},
        'run': {'text': 'Running', 'color': '#FF7043'},
        'stairs': {'text': 'Stairs', 'color': '#D84315'},
        'bicycle': {'text': 'Bicycling', 'color': '#E53935'},
        'row': {'text': 'Rowing', 'color': '#EC407A'},
        'kneel': {'text': 'Kneeling', 'color': '#26C6DA'},
        'squat': {'text': 'Squatting', 'color': '#00838F'},
    },
    'title': '24/7 Movement Behaviour',
    'x': 'Time',
    'y': 'Day',
    'legend': 'Activity',
    'weekdays': {
        'Monday': 'Monday',
        'Tuesday': 'Tuesday',
        'Wednesday': 'Wednesday',
        'Thursday': 'Thursday',
        'Friday': 'Friday',
        'Saturday': 'Saturday',
        'Sunday': 'Sunday',
    },
}


PLOT_FUSED_LANG = {
    'activities': {
        'non-wear': {'text': 'Non-wear', 'color': '#BDBDBD'},
        'sedentary': {'text': 'Sedentary', 'color': '#42A5F5'},
        'stand': {'text': 'Standing', 'color': '#26A69A'},
        'walk': {'text': 'Walking', 'color': '#66BB6A'},
        'run': {'text': 'Running', 'color': '#FF7043'},
        'bicycle': {'text': 'Bicycling', 'color': '#EF5350'},
        'row': {'text': 'Rowing', 'color': '#AB47BC'},
    },
    'title': '24/7 Movement Behaviour',
    'x': 'Time',
    'y': 'Day',
    'legend': 'Activity',
    'weekdays': {
        'Monday': 'Monday',
        'Tuesday': 'Tuesday',
        'Wednesday': 'Wednesday',
        'Thursday': 'Thursday',
        'Friday': 'Friday',
        'Saturday': 'Saturday',
        'Sunday': 'Sunday',
    },
}


@dataclass
class Exposures:
    window: str | timedelta = '1d'
    fused: bool = False

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
            df['activity'].isin(['stand', 'shuffle', 'walk', 'fast-walk', 'run', 'stairs'])
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
        valid = df['activity'].isin(['stand', 'shuffle', 'walk', 'fast-walk']) & (
            df['arm_inclination'].between(lower, upper, inclusive='both')
        )

        return valid

    def _get_exposures(self, df: pd.DataFrame) -> pd.Series:
        exposure = {
            'wear': self._get_exposure(df, df['activity'] != 'non-wear', 'time'),
            'sedentary': self._get_exposure(df, df['activity'].isin(['sit', 'lie']), 'time'),
            'standing': self._get_exposure(df, df['activity'].isin(['stand', 'shuffle']), 'time'),
            'on_feet': self._get_exposure(
                df, df['activity'].isin(['stand', 'shuffle', 'walk', 'fast-walk', 'run', 'stairs']), 'time'
            ),
            'sedentary_to_other': self._get_exposure(df, df['activity'].isin(['sit', 'lie', 'kneel']), 'count'),
            'lpa': self._get_exposure(
                df,
                df['activity'].isin(['stand', 'shuffle', 'walk', 'squat']),
                'time',
            ),
            'mvpa': self._get_exposure(
                df,
                df['activity'].isin(['fast-walk', 'run', 'stairs', 'bicycle', 'row']),
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

    def _get_activities(self, activities: pd.Series, activity_types: list[str]) -> pd.DataFrame:
        columns = [col for col in activity_types if col in activities.unique() and col != 'non-wear']

        df = (
            activities.groupby([pd.Grouper(freq=self.window, sort=True), activities], observed=False).count()  # type: ignore
        )
        df = df.apply(pd.Timedelta, unit='s').unstack()
        return df[columns]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        exposure = df.groupby(pd.Grouper(freq=self.window, sort=True)).apply(self._get_exposures)  # type: ignore
        activities = self._get_activities(df['activity'], ACTIVITIES.values())  # type: ignore

        if not self.fused:
            exposure = pd.concat([exposure, activities], axis=1)

        valid = (activities['stand'] + activities['walk']) >= pd.Timedelta(minutes=15)

        exposure.insert(
            0,
            'valid',
            valid,
        )

        return exposure

    def _get_plot(self, activities: pd.Series, lang: dict) -> alt.Chart:
        df = activities.to_frame('activities')
        df.index = df.index.tz_localize(None)  # type: ignore

        start = df.index.min()
        end = df.index.max() + pd.Timedelta(days=1)
        full_idx = pd.date_range(start=start, end=end, freq='1s', normalize=True)
        df = df.reindex(full_idx, fill_value='non-wear')

        labels = {k: v['text'] for k, v in lang['activities'].items()}
        colors = [v['color'] for k, v in lang['activities'].items()]

        df['activities'] = df['activities'].astype(str).replace(labels)
        domain = labels.values()

        df['datetime'] = df.index
        df['y_label'] = (
            df['datetime'].dt.strftime('%d-%m-%Y') + ' (' + df['datetime'].dt.day_name().map(lang['weekdays']) + ')'
        )

        heatmap = (
            alt.Chart(df[:-1])
            .mark_rect()
            .encode(
                x=alt.X(
                    'hoursminutesseconds(datetime):T',
                    title=lang['x'],
                    axis=alt.Axis(
                        labelFontSize=12,
                        titleFontSize=14,
                        format='%H:%M',
                        ticks=True,
                    ),
                ),
                y=alt.Y(
                    'y_label:O',
                    title=lang['y'],
                    axis=alt.Axis(
                        labelFontSize=12,
                        titleFontSize=14,
                        ticks=True,
                        grid=True,
                        gridColor='gray',
                        gridOpacity=0.1,
                    ),
                ),
                color=alt.Color(
                    'activities:N',
                    title=lang['legend'],
                    scale=alt.Scale(domain=domain, range=colors),
                    legend=alt.Legend(labelFontSize=12, titleFontSize=14),
                ),
                # tooltip=['hoursminutesseconds(datetime):T', 'activities:N'],
            )
            .properties(
                width=960,
                height=270,
                title=alt.TitleParams(text=lang['title'], fontSize=16),
            )
        )

        return heatmap

    def plot(self, df: pd.DataFrame, language: dict[str, Any] | None = None) -> alt.Chart:
        activities = df['activity']

        if self.fused:
            activities = activities.astype(str).replace(FUSED_ACTIVITIES)

        if language is None:
            language = PLOT_FUSED_LANG if self.fused else PLOT_LANG

        return self._get_plot(activities, language)

    def quality_check(self, df: pd.DataFrame) -> pd.DataFrame:
        exposures = self.compute(df)
        valid_dates = exposures.loc[exposures['valid']].index.normalize()  # type: ignore
        df['valid'] = df.index.normalize().isin(valid_dates)  # type: ignore

        return df
