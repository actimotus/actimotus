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
        window: The time window for aggregation, as a pandas offset string.
            Use uppercase calendar aliases for day/week windows (`'1D'` for
            daily totals, `'7D'` for weekly); sub-day windows like `'1h'` also
            work. Defaults to daily (`'1D'`). Day/week windows bucket on **local
            calendar days**, so across a DST transition a fall-back day is a
            genuine 25-hour window and a spring-forward day a 23-hour window
            (labelled at local midnight) — this is correct, not a defect.
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

        >>> exposures = Exposures(window='7D', fused=True)
    """

    window: str = '1D'
    fused: bool = False

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
        # FIXME: Most probably this should include all activities expect lying?
        valid = df['activity'].isin(['stand', 'shuffle', 'walk', 'fast-walk']) & (
            df['arm_inclination'].between(lower, upper, inclusive='both')
        )

        return valid

    def _get_exposures(self, df: pd.DataFrame) -> pd.Series:
        sedentary = ['sit', 'lie', 'kneel']

        exposure = {
            'wear': self._get_exposure(df, df['activity'] != 'non-wear', 'time'),
            'sedentary': self._get_exposure(df, df['activity'].isin(sedentary), 'time'),
            'standing': self._get_exposure(df, df['activity'].isin(['stand', 'shuffle']), 'time'),
            'on_feet': self._get_exposure(
                df,
                df['activity'].isin(['stand', 'shuffle', 'walk', 'fast-walk', 'run', 'stairs', 'squat']),
                'time',
            ),
            'sedentary_to_other': self._get_exposure(df, df['activity'].isin(sedentary), 'count'),
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
            >>> exposures = Exposures(window='1D', fused=False)
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
        df['end_time'] += timedelta(seconds=1)  # make end_time exclusive

        df['start_time'] = df['start_time'].dt.tz_localize(None)  # type: ignore
        df['end_time'] = df['end_time'].dt.tz_localize(None)  # type: ignore

        df['duration'] = df['end_time'] - df['start_time']
        df['duration'] = df['duration'].dt.total_seconds() / 60.0  # duration in minutes # type: ignore
        df['end_time'] = df['end_time'] - timedelta(seconds=1)  # make end_time inclusive for plotting
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
    def _validate_diary(diary: pd.DataFrame) -> None:
        """Validate a diary of context intervals.

        The diary must have columns ``start``, ``end``, ``context`` (and an
        optional ``activities`` column). ``start``/``end`` must be timezone-aware
        datetimes without ``NaT`` and every ``end`` must be strictly after its
        ``start``. Each ``context`` must be a non-empty string. Each ``activities``
        cell must be missing (``None``/``NaN``/``pd.NA``) or a list of known
        ``ACTIVITIES`` labels.

        Args:
            diary: DataFrame with columns ``start``, ``end``, ``context`` and an
                optional per-row ``activities`` list.

        Raises:
            ValueError: If required columns are missing; ``start``/``end`` are not
                timezone-aware or contain ``NaT``; any interval has ``end <= start``;
                a ``context`` is null, non-string, or empty; or an ``activities``
                cell is neither missing nor a list of known ``ACTIVITIES`` labels.
        """
        required = {'start', 'end', 'context'}
        missing = required - set(diary.columns)
        if missing:
            raise ValueError(f'Diary missing required columns: {sorted(missing)}')

        for column in ('start', 'end'):
            if not isinstance(diary[column].dtype, pd.DatetimeTZDtype):
                raise ValueError(
                    f"Diary column '{column}' must be timezone-aware datetimes."
                )
            if diary[column].isna().any():
                raise ValueError(
                    f"Diary column '{column}' contains NaT (missing timestamps)."
                )

        if (diary['end'] <= diary['start']).any():
            raise ValueError("Diary has rows where 'end' is not after 'start'.")

        for context in diary['context']:
            if not isinstance(context, str) or not context.strip():
                raise ValueError(
                    f'Diary has an invalid context value: {context!r}. '
                    'Context must be a non-empty string.'
                )

        if 'activities' in diary.columns:
            known = set(ACTIVITIES.values())
            for activities in diary['activities']:
                if isinstance(activities, list):
                    for label in activities:
                        if not isinstance(label, str):
                            raise ValueError(
                                f'Diary activities must be strings; got {label!r}.'
                            )
                        if label not in known:
                            raise ValueError(
                                f'Diary activities contains unknown label {label!r}. '
                                f'Known labels: {sorted(known)}.'
                            )
                elif pd.api.types.is_scalar(activities) and pd.isna(activities):
                    continue  # missing == no gate
                else:
                    raise ValueError(
                        f'Diary activities must be a list of labels or missing; '
                        f'got {activities!r}.'
                    )

    @staticmethod
    def _context_mask(df: pd.DataFrame, intervals: pd.DataFrame) -> pd.Series:
        """Boolean mask for one context across its (possibly multiple) intervals.

        An epoch is ``True`` when its timestamp falls inside any interval's
        half-open ``[start, end)`` window and, if that interval row carries a
        non-empty ``activities`` list, the epoch's ``activity`` is in it. Rows for
        the same context union together.

        Args:
            df: Activity DataFrame, timezone-aware DatetimeIndex, ``activity`` column.
            intervals: The diary rows for a single context (``start``, ``end`` and
                optional per-row ``activities``).

        Returns:
            Boolean ``pd.Series`` aligned to ``df.index``.
        """
        mask = pd.Series(False, index=df.index)
        has_activities = 'activities' in intervals.columns

        for row in intervals.itertuples(index=False):
            in_interval = (df.index >= row.start) & (df.index < row.end)

            activities = row.activities if has_activities else None
            if not (
                pd.api.types.is_scalar(activities) and pd.isna(activities)
            ) and len(activities) > 0:
                in_interval = in_interval & df['activity'].isin(activities).to_numpy()

            mask = mask | in_interval

        return mask

    @staticmethod
    def context(df: pd.DataFrame, diary: pd.DataFrame) -> pd.DataFrame:
        """Annotate the activity DataFrame with diary contexts.

        For each distinct ``context`` in the diary, adds a boolean
        ``context__<name>`` column that is ``True`` for epochs inside any of that
        context's intervals (optionally gated by each interval's ``activities``).
        Contexts may overlap; multiple intervals for one context union into a
        single column. Surrounding whitespace in context names is stripped, so
        ``' work '`` and ``'work'`` collapse into one column. The input frame is not
        mutated — a copy is returned.

        Args:
            df: Activity DataFrame, timezone-aware DatetimeIndex, ``activity`` column.
            diary: Clean diary with columns ``start``, ``end``, ``context`` and an
                optional per-row ``activities`` list, timezone-aware and in the same
                timezone as ``df``'s index.

        Returns:
            A copy of ``df`` with one ``context__<name>`` boolean column per context.

        Raises:
            ValueError: If the diary is invalid (see :meth:`_validate_diary`); ``df``
                has no ``activity`` column or a timezone-naive index; the diary
                ``start``/``end`` zones differ from the index zone; or ``df`` already
                has a ``context__<name>`` column that this call would create.
        """
        Exposures._validate_diary(diary)

        if 'activity' not in df.columns:
            raise ValueError("Activity DataFrame must have an 'activity' column.")

        if df.index.tz is None:
            raise ValueError('Activity DataFrame index must be timezone-aware.')

        index_tz = str(df.index.tz)
        for column in ('start', 'end'):
            if str(diary[column].dt.tz) != index_tz:
                raise ValueError(
                    f"Diary '{column}' timezone ({diary[column].dt.tz}) does not "
                    f'match activity index timezone ({df.index.tz}).'
                )

        diary = diary.copy()
        diary['context'] = diary['context'].str.strip()

        new_columns = {f'context__{context}' for context in diary['context'].unique()}
        collisions = new_columns & set(df.columns)
        if collisions:
            raise ValueError(
                f'Activity DataFrame already has context columns: {sorted(collisions)}.'
            )

        df = df.copy()
        for context, intervals in diary.groupby('context', sort=False):
            df[f'context__{context}'] = Exposures._context_mask(df, intervals)

        return df
