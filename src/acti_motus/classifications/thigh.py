import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.signal import medfilt

from .references import References
from .sensor import Calculation, Sensor

logger = logging.getLogger(__name__)


@dataclass
class Thigh(Sensor):
    system_frequency: int
    vendor: Literal['Sens', 'Other']
    config: dict[str, Any]
    # rotate: bool = False # TODO: Implement rotation logic first.

    def check_inside_out_flip(self, df: pd.DataFrame) -> bool:
        # TODO: Check that this works correctly.
        rows_per_hour = self.system_frequency * 60 * 2
        window = rows_per_hour * 3
        step = rows_per_hour
        min_periods = rows_per_hour

        inclination = df['inclination']
        z = df['z']

        outliers = inclination.rolling(window=window, step=step, min_periods=min_periods).quantile(0.02) > 45
        outliers_mask = pd.Series(df.index.map(outliers), index=df.index, dtype='boolean').ffill()
        valid_points = z.loc[~outliers_mask & (inclination > 45)]

        if valid_points.empty:
            logger.warning('Not enough data to check inside out flip. Skipping.')
            return False

        mdn = np.median(valid_points)

        flip = True if (mdn > 0.1) else False

        if flip:
            logger.warning(f'Inside out flip detected (median z: {mdn:.2f}).')

        return flip

    def calculate_reference_angle(self, df: pd.DataFrame) -> tuple[float, Calculation]:
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
                reference_angle = default_angle
                logger.warning(
                    f'Reference angle {reference_angle:.2f} degrees is outside the threshold range. Using default reference angle: {reference_angle:.2f} degrees.'
                )

            else:
                angle_status = Calculation.AUTOMATIC
                logger.info(f'Reference angle calculated: {reference_angle:.2f} degrees.')
        else:
            reference_angle = default_angle
            logger.warning(f'No valid walk data found. Using default reference angle: {reference_angle:.2f} degrees.')

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
            + (2 * self.system_frequency * sq_x)
            + (2 * sin * df['x'] * df['sum_z'])
            + (-2 * sin * cos * df['sum_dot_xz'])
            + (-2 * cos * df['x'] * df['sum_x'])
        )
        sd.loc[sd['terms_x'] <= 0, 'terms_x'] = 0
        sd['sd_x'] = np.sqrt(1 / (2 * self.system_frequency - 1) * sd['terms_x'])

        sd['terms_z'] = (
            (sq_sin * df['sq_sum_x'])
            + (sq_cos * df['sq_sum_z'])
            + (2 * self.system_frequency * sq_z)
            + (2 * sin * cos * df['sum_dot_xz'])
            + (-2 * sin * df['z'] * df['sum_x'])
            + (-2 * cos * df['z'] * df['sum_z'])
        )
        sd.loc[sd['terms_z'] <= 0, 'terms_z'] = 0
        sd['sd_z'] = np.sqrt(1 / (2 * self.system_frequency - 1) * sd['terms_z'])

        sd['sd_y'] = df['sd_y']

        return sd[['sd_x', 'sd_y', 'sd_z']].astype(np.float32)

    def rotate_by_reference_angle(self, df: pd.DataFrame, angle: float) -> pd.DataFrame:  # type: ignore
        df = df.copy()
        angle = np.float32(angle)  # type: ignore
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

    def _downsample_sd_correction_for_sens(
        self,
        df: pd.DataFrame,
        sampling_frequency: float,
        tolerance: float = 1,
    ) -> None:
        values, correction = None, None

        if math.isclose(sampling_frequency, 25, abs_tol=tolerance):
            values = (0.18, 1.03)
            correction = 25

        elif math.isclose(sampling_frequency, 12.5, abs_tol=tolerance):
            values = (0.02, 1.14)
            correction = 12.5

        if values:
            df[['sd_x', 'sd_y', 'sd_z']] = df[['sd_x', 'sd_y', 'sd_z']].apply(
                lambda x: values[0] * x**2 + values[1] * x
            )
            logger.info(
                f'Applied SD correction [{values[0]} * x^2 + {values[1]} * x] which is specific for Sens sensors at {correction:.1f} Hz ({sampling_frequency:.2f} Hz).'
            )
        else:
            logger.warning(f'No correction applied for sampling frequency {sampling_frequency:.2f} Hz on axes.')

    def _median_filter(
        self,
        valid: pd.Series,
        bouts_length: int,
    ) -> pd.Series:
        length = 2 * bouts_length - 1
        filtered = medfilt(valid.astype(int), length)
        filtered = medfilt(filtered, length)

        valid = pd.Series(filtered.astype(bool), index=valid.index, name='median_filtered')

        return valid

    def get_row(
        self,
        df: pd.DataFrame,
        bout: int,
        movement_threshold: float,
        inclination_angle: float,
        **kwargs,
    ) -> pd.Series:
        # NOTE: Do we miss the "mean anterior-posterior angle is greater than 40°"?
        valid = (inclination_angle < df['inclination']) & (movement_threshold < df['sd_x'])
        valid = self._median_filter(valid, bout)
        valid.name = 'row'

        return valid

    def get_bicycle(
        self,
        df: pd.DataFrame,
        bout: int,
        movement_threshold: float,
        inclination_angle: float,
        anterior_posterior_angle: float,
        direction_threshold: float,
        **kwargs,
    ) -> pd.Series:
        valid = (
            ((anterior_posterior_angle) < df['direction'])
            & (df['inclination'] < inclination_angle)
            & (movement_threshold < df['sd_x'])
        )

        valid = pd.Series(medfilt(valid.astype(int), 9), index=df.index)
        valid = valid & ((df['hl_ratio'] < 0.5) | (df['direction'] > direction_threshold))

        valid = self._median_filter(valid, bout)
        valid.name = 'bicycle'

        return valid

    def _get_stairs_threshold(
        self,
        df: pd.DataFrame,
        run_threshold: float,
        anterior_posterior_angle: float,
        stairs_threshold: float,
    ) -> float:
        valid = df['sd_x'].between(0.25, run_threshold, inclusive='neither') & (
            df['direction'] < anterior_posterior_angle
        )

        valid = df.loc[valid, 'direction']

        # FIXME: Maybe get different threshold if no valid data is found.
        if valid.empty:
            logger.warning('No valid data found for stairs threshold calculation. Using default stairs threshold.')
            valid = stairs_threshold
        else:
            valid = stairs_threshold + np.median(valid)  # type: ignore
            valid = valid.item()

        return valid

    def get_stairs(
        self,
        df: pd.DataFrame,
        bout: int,
        inclination_angle: float,
        movement_threshold: float,
        run_threshold: float,
        direction_threshold: float,
        stairs_threshold: float,
        anterior_posterior_angle: float,
        **kwargs,
    ) -> tuple[pd.Series, float]:
        stairs_threshold = self._get_stairs_threshold(df, run_threshold, anterior_posterior_angle, stairs_threshold)

        valid = (
            (stairs_threshold < df['direction'])
            & (df['direction'] < direction_threshold)
            & (movement_threshold < df['sd_x'])
            & (df['sd_x'] < run_threshold)
            & (df['inclination'] < inclination_angle)
        )

        valid = self._median_filter(valid, bout)
        valid.name = 'stairs'

        return valid, stairs_threshold

    def get_run(
        self,
        df: pd.DataFrame,
        bout: int,
        inclination_angle: float,
        run_threshold: float,
        **kwargs,  # Because there is extra parameter, so it will be ignored.
    ) -> pd.Series:
        valid = (df['sd_x'] > run_threshold) & (df['inclination'] < inclination_angle)

        valid = self._median_filter(valid, bout)
        valid.name = 'run'

        return valid

    def get_walk(
        self,
        df: pd.DataFrame,
        bout: int,
        stairs_threshold: float,
        inclination_angle: float,
        movement_threshold: float,
        run_threshold: float,
        **kwargs,
    ) -> pd.Series:
        valid = (
            (movement_threshold < df['sd_x'])
            & (df['sd_x'] < run_threshold)
            & (df['direction'] < stairs_threshold)
            & (df['inclination'] < inclination_angle)
        )

        valid = self._median_filter(valid, bout)
        valid.name = 'walk'

        return valid

    def get_stand(
        self,
        df: pd.DataFrame,
        bout: int,
        inclination_angle: float,
        movement_threshold: float,
        **kwargs,
    ) -> pd.Series:
        sd_max = np.max(df[['sd_x', 'sd_y', 'sd_z']], axis=1)
        valid = (df['inclination'] < inclination_angle) & (sd_max < movement_threshold)

        valid = self._median_filter(valid, bout)
        valid.name = 'stand'

        return valid

    def get_sit(
        self,
        df: pd.DataFrame,
        bout: int,
        inclination_angle: float,
        **kwargs,
    ) -> pd.Series:
        valid = df['inclination'] > inclination_angle
        valid = self._median_filter(valid, bout)
        valid.name = 'sit'

        return valid

    def _get_activity_column(self, df: pd.DataFrame) -> pd.Series:
        # Order matters here
        categories = [
            'row',
            'bicycle',
            'stairs',
            'run',
            'walk',
            'stand',
            'sit',
        ]

        df = df[categories].copy()
        df['shuffle'] = True  # If nothing else, it is shuffle

        categories.append('shuffle')
        activity = df[categories].idxmax(axis=1)
        categories = categories + ['lie', 'non-wear']
        activity = activity.astype(pd.CategoricalDtype(categories=categories))
        activity.name = 'activity'

        return activity

    def _fix_activities_bouts(self, df: pd.DataFrame, bouts: dict[str, int]) -> pd.Series:
        activities = df['activity']

        # Order matters here - shuffle position changed
        for activity in [
            'row',
            'bicycle',
            'stairs',
            'run',
            'walk',
            'shuffle',
            'stand',
            'sit',
        ]:
            activities = self.fix_bouts(activities, activity, bouts[activity])

        return activities

    def _get_rotational_crossing_points(
        self,
        df: pd.DataFrame,
        orientation_threshold: float,
    ) -> pd.DataFrame:
        noise_margin = 0.05

        thigh_angle = np.arcsin(df['y'] / np.sqrt(np.square(df['y']) + np.square(df['z'])))  # type: pd.Series # type:ignore
        thigh_angle = np.absolute(np.degrees(thigh_angle))  # type: pd.Series # type:ignore

        low = (thigh_angle < orientation_threshold).diff()
        high = (thigh_angle >= orientation_threshold).diff()

        noise = thigh_angle.diff().abs()  # type: float # type: ignore
        noise = noise >= noise_margin

        low = low & noise
        high = high & noise

        return pd.DataFrame({'low': low, 'high': high}, index=df.index)

    def get_lie(
        self,
        df: pd.DataFrame,
        bout: int,
        orientation_angle: float,
        **kwargs,
    ) -> pd.Series:
        df = df[['activity', 'y', 'z']].copy()
        df[['low', 'high']] = self._get_rotational_crossing_points(df, orientation_angle)
        df['lie'] = pd.Series(False, index=df.index, dtype=bool)
        df['bout'] = (df['activity'] != df['activity'].shift()).cumsum()

        specific_activity = df.loc[
            df['activity'] == 'sit', 'bout'
        ]  # NOTE: Activity "sit" needs to be already in the activities dataframe.
        bout_sizes = specific_activity.value_counts()
        specific_bouts = bout_sizes[bout_sizes > bout].index.values  # FIXME: Try > or >=

        df['specific'] = False
        df.loc[df['bout'].isin(specific_bouts), 'specific'] = True

        updated_bouts = df[df['specific']].groupby('bout').aggregate({'low': 'any', 'high': 'any'})
        updated_bouts = updated_bouts[updated_bouts['low'] & updated_bouts['high']]
        df['lie'] = False
        df.loc[df['bout'].isin(updated_bouts.index), 'lie'] = True

        return df['lie']

    def get_steps(self, df: pd.DataFrame) -> pd.Series:
        df = df[['activity', 'walk_feature', 'run_feature']].copy()
        scale = self.system_frequency / 2 * np.linspace(0, 1, 256)

        df['steps'] = 0
        df.loc[df['activity'].isin(['walk', 'stairs']), 'steps'] = df['walk_feature']
        df.loc[df['activity'] == 'run', 'steps'] = df['run_feature']
        df['steps'] = scale[df['steps']]
        df['steps'] = medfilt(df['steps'], 3)

        return df['steps'].astype(np.float32)

    def compute_activities(
        self,
        df: pd.DataFrame,
        references: References,
    ) -> pd.DataFrame:
        config = self.config['thigh']

        df = df.copy()
        sf = df['sf'].mode().values[0].item()

        df[['inclination', 'side_tilt', 'direction']] = self.get_angles(df)
        non_wear = self.get_non_wear(df)
        bouts = references.get_bouts(non_wear, 'thigh')

        if self.orientation:
            df = self.fix_bouts_orientation(df, bouts)

        df, bouts = self.rotate_bouts_by_reference_angles(df, bouts)

        if self.vendor.lower() == 'sens':
            self._downsample_sd_correction_for_sens(df, sf)

        df['row'] = self.get_row(df, **config['row'])
        df['bicycle'] = self.get_bicycle(df, **config['bicycle'])
        df['stairs'], stairs_threshold = self.get_stairs(df, **config['stairs'])
        df['run'] = self.get_run(df, **config['run'])
        df['walk'] = self.get_walk(df, **config['walk'], stairs_threshold=stairs_threshold)
        df['stand'] = self.get_stand(df, **config['stand'])
        df['sit'] = self.get_sit(df, **config['sit'])

        df['activity'] = self._get_activity_column(df)
        bouts_length = {activity[0]: activity[1]['bout'] for activity in config.items()}
        df['activity'] = self._fix_activities_bouts(df, bouts_length)

        df['lie'] = self.get_lie(df, **config['lie'])
        df.loc[df['lie'], 'activity'] = 'lie'

        for activity in ['sit', 'lie']:
            df['activity'] = self.fix_bouts(df['activity'], activity, bouts_length[activity])

        df.loc[non_wear, 'activity'] = 'non-wear'
        del non_wear

        df['steps'] = self.get_steps(df)

        df.loc[(df['activity'] == 'walk') & (df['steps'] > config['run']['step_frequency']), 'activity'] = 'run'
        for activity in ['run', 'walk']:
            df['activity'] = self.fix_bouts(df['activity'], activity, bouts_length[activity])

        df.loc[df['activity'] == 'non-wear', 'direction'] = np.nan
        df.rename(
            columns={
                'inclination': 'thigh_inclination',
                'side_tilt': 'thigh_side_tilt',
                'direction': 'thigh_direction',
            },
            inplace=True,
        )

        references.update_angle(bouts, 'thigh')

        return df[['activity', 'steps', 'thigh_inclination', 'thigh_side_tilt', 'thigh_direction']]
