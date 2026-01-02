import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import altair as alt
import pandas as pd

from .settings import ACTIVITIES, FUSED_ACTIVITIES, PLOT, PLOT_FUSED

logger = logging.getLogger(__name__)

alt.data_transformers.enable('vegafusion')


@dataclass
class Exposures:
    """Aggregates activity data into summary exposure metrics.

    This class takes the 1-second activity epochs (produced by `Activities`) and
    calculates aggregate exposure metrics over a specified time window. Metrics
    include total time spent in specific postures (e.g., Sedentary, MVPA),
    and frequency of transitions (e.g., sit-to-stand) and also data quality check indicating invalid data.

    It supports generating results as raw DataFrame or visual plot.

    Attributes:
        window: The time window for aggregation. Accepts a `timedelta` object
            or a pandas-style string offset (e.g., `'1d'` for daily totals,
            `'1h'` for hourly). Defaults to daily aggregation.
        fused: If `True`, granular activity categories are merged into broader
            semantic groups before calculation. This simplifies the output by
            combining physiologically similar states.

            **Fusion Mappings:**

            * **Sedentary**: Combines *lie*, *sit*, and *kneel*.
            * **Standing**: Combines *stand*, *squat*, and *shuffle*.
            * **Walking**: Combines *walk*, *fast-walk*, and *stairs* climbing.

    Examples:
        Standard daily exposures with full granular categories:

        >>> exposures = Exposures()
        >>> # results = exposures.compute(activities)

        Weekly exposures with fused categories (grouping all walking types):

        >>> exposures = Exposures(window='7d', fused=True)
    """

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
            'sedentary': self._get_exposure(df, df['activity'].isin(['sit', 'lie', 'stand']), 'time'),
            'standing': self._get_exposure(df, df['activity'].isin(['stand', 'shuffle']), 'time'),
            'on_feet': self._get_exposure(
                df,
                df['activity'].isin(['stand', 'shuffle', 'walk', 'fast-walk', 'run', 'stairs', 'squat']),
                'time',
            ),
            'sedentary_to_other': self._get_exposure(df, df['activity'].isin(['sit', 'lie', 'kneel']), 'count'),
            'lpa': self._get_exposure(
                df,
                df['activity'].isin(['shuffle', 'walk', 'squat']),
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
        df = (
            activities.groupby([pd.Grouper(freq=self.window, sort=True), activities], observed=False).count()  # type: ignore
        )
        df = df.apply(pd.Timedelta, unit='s').unstack()

        # NOTE: This could be done in better way.
        for col in activity_types:
            if col not in df.columns:
                df[col] = pd.Timedelta(0)

        df.drop(columns=['non-wear'], inplace=True)

        return df

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates exposure metrics and validity flags for the given activity data.

        This method aggregates the input time-series based on the configured
        `window` size (e.g., daily). It computes durations for each activity
        category and determines if the monitoring period is considered "valid."

        **Validity Criteria:**
        A window is marked as `valid` (True) if the subject performed at least
        **10 minutes** of active movement (Stairs + Walk) within that period.

        Args:
            df: A DataFrame containing 1-second activity epochs. It must be
                indexed by a `DatetimeIndex` and contain an `'activity'` column.

        Returns:
            A DataFrame indexed by the time window (e.g., each day), containing:

            * **valid**: Boolean flag indicating if the window met the
                activity threshold.
            * **[Activity Names]**: Columns for each activity type (e.g., 'sit',
                'stand', 'walk'), containing the total duration (`timedelta`)
                spent in that state.
            * **[Fused Categories]**: If `fused=True`, contains broader
                categories like 'sedentary' instead of granular ones.

        Examples:
            >>> exposures = Exposures(window='1d', fused=False)
            >>> results = exposures.compute(activity_epochs_df)
        """
        exposure = df.groupby(pd.Grouper(freq=self.window, sort=True)).apply(self._get_exposures)  # type: ignore
        activities = self._get_activities(df['activity'], ACTIVITIES.values())  # type: ignore

        if not self.fused:
            exposure = pd.concat([exposure, activities], axis=1)

        valid = (activities['stairs'] + activities['walk']) >= pd.Timedelta(minutes=10)

        exposure.insert(
            0,
            'valid',
            valid,
        )

        return exposure

    def _get_plot(self, activities: pd.Series, lang: dict) -> alt.Chart:
        labels = {k: v['text'] for k, v in lang['activities'].items()}
        colors = [v['color'] for k, v in lang['activities'].items()]
        domain = list(labels.values())

        df = activities.to_frame('activity')

        start = df.index[0]
        end = df.index[-1]

        full_idx = pd.date_range(start=start, end=end, freq='1s', name='datetime')
        df = df.reindex(full_idx, fill_value='non-wear').reset_index()

        df['rle'] = (df['activity'] != df['activity'].shift()).cumsum()
        df['date'] = df['datetime'].dt.date  # type: ignore

        df = (
            df.groupby(['rle', 'date'])
            .agg(
                activity=('activity', 'first'),
                start_time=('datetime', 'first'),
                end_time=('datetime', 'last'),
            )
            .reset_index(drop=True)
        )

        df['start_time'] = df['start_time'].dt.tz_localize(None)
        df['end_time'] = df['end_time'].dt.tz_localize(None)

        df['duration'] = df['end_time'] - df['start_time']
        df['duration'] = df['duration'].dt.total_seconds() / 60.0  # duration in minutes # type: ignore
        df['activity'] = df['activity'].astype(str).replace(labels)
        df['y_label'] = (
            df['start_time'].dt.strftime('%d-%m-%Y') + ' (' + df['start_time'].dt.day_name().map(lang['weekdays']) + ')'  # type: ignore
        )

        heatmap = (
            alt.Chart(df)
            .mark_rect(opacity=1)
            .encode(
                x=alt.X(
                    'hoursminutesseconds(start_time):T',
                    title=lang['x'],
                    axis=alt.Axis(
                        labelFontSize=12,
                        titleFontSize=14,
                        format='%H:%M',
                        ticks=True,
                    ),
                ),
                x2=alt.X2('hoursminutesseconds(end_time):T'),
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
                    sort=None,
                ),
                color=alt.Color(
                    'activity:N',
                    title=lang['legend'],
                    scale=alt.Scale(domain=domain, range=colors),
                    legend=alt.Legend(labelFontSize=12, titleFontSize=14),
                ),
                tooltip=[
                    alt.Tooltip('hoursminutesseconds(start_time):T', title='Start Time', format='%H:%M'),
                    alt.Tooltip('activity:N', title='Activity'),
                    alt.Tooltip('duration:Q', title='Duration (min)', format='.2f'),
                ],
            )
            .properties(
                width=960,
                height=270,
                title=alt.TitleParams(text=lang['title'], fontSize=16),
            )
        )

        return heatmap

    def plot(self, df: pd.DataFrame, language: dict[str, Any] | None = None) -> alt.Chart:
        """Generates an interactive Gantt-style chart of the activity timeline.

        This method visualizes the 1-second activity epochs. If `fused` is enabled
        on this instance, the plot will automatically group similar activities
        (e.g., 'walk', 'stairs' -> 'Walking') and use the simplified color scheme.

        Args:
            df: The input DataFrame containing 1-second activity epochs. Must
                contain an `'activity'` column.
            language: A configuration dictionary to customize chart labels and
                colors (e.g., for localization). If `None`, defaults to the
                standard English configuration.

        Returns:
            An Altair Chart object representing the activity timeline. To display it in a notebook, simply let the object return or call `.display()`.

        Examples:
            Basic usage with default English labels:

            >>> chart = exposures.plot(df)
            >>> chart.save('timeline.html')
        """
        activities = df['activity']

        if self.fused:
            activities = activities.astype(str).replace(FUSED_ACTIVITIES)

        if language is None:
            language = PLOT_FUSED if self.fused else PLOT

        return self._get_plot(activities, language)

    def quality_check(self, df: pd.DataFrame) -> pd.DataFrame:
        exposures = self.compute(df)
        valid_dates = exposures.loc[exposures['valid']].index.normalize()  # type: ignore
        df['valid'] = df.index.normalize().isin(valid_dates)  # type: ignore

        return df

    @staticmethod
    def context(
        df: pd.DataFrame,
        intervals: pd.DataFrame,
        context: str,
        activities: list[str] | None = None,
    ) -> pd.Series:
        df = df[['activity']].copy()
        df[context] = False

        for start, end in intervals.itertuples(index=False):
            logic = (df.index >= start) & (df.index < end)
            logic = logic & (df['activity'].isin(activities) if activities is not None else logic)
            df.loc[logic, context] = True

        return df[context]
