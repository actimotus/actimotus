import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from .classifications import Arm, Calf, References, Thigh, Trunk
from .iterators import DataFrameIterator
from .logger import traceable_logging
from .settings import ACTIVITIES, FEATURES, SENS__ACTIVITY_VALUES, SENS__FLOAT_FACTOR

logger = logging.getLogger(__name__)


@dataclass
class Activities:
    vendor: Literal['Sens', 'Other'] = 'Other'
    orientation: bool = True
    chunks: bool = False
    size: timedelta = '1d'
    overlap: timedelta = '15min'

    def __post_init__(self):
        if isinstance(self.size, str):
            self.size = pd.Timedelta(self.size).to_pytimedelta()

        if isinstance(self.overlap, str):
            self.overlap = pd.Timedelta(self.overlap).to_pytimedelta()

    def _detect_chunk(
        self,
        thigh: pd.DataFrame,
        *,
        trunk: pd.DataFrame | None = None,
        calf: pd.DataFrame | None = None,
        arm: pd.DataFrame | None = None,
        references: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, References]:
        not_overlaps = thigh[~thigh['overlap']]
        start, end = not_overlaps.index[0], not_overlaps.index[-1]

        if trunk is not None:
            trunk = trunk.loc[(trunk.index >= start) & (trunk.index < end)]

        if calf is not None:
            calf = calf.loc[(calf.index >= start) & (calf.index < end)]

        if arm is not None:
            arm = arm.loc[(arm.index >= start) & (arm.index < end)]

        df, references = self._detect(
            thigh=thigh,
            trunk=trunk,
            calf=calf,
            arm=arm,
            references=references,
        )

        df = df.loc[(df.index >= start) & (df.index < end)]

        return df, references

    def _detect_chunks(
        self,
        thigh: pd.DataFrame,
        *,
        trunk: pd.DataFrame | None = None,
        calf: pd.DataFrame | None = None,
        arm: pd.DataFrame | None = None,
        references: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, References]:
        chunks = DataFrameIterator(thigh, size=self.size, overlap=self.overlap)

        output = []

        for chunk in chunks:
            df, references = self._detect_chunk(
                chunk,
                trunk=trunk,
                calf=calf,
                arm=arm,
                references=references,
            )
            output.append(df)

        output = pd.concat(output)
        output.sort_index(inplace=True)

        return output, references

    def _detect(
        self,
        thigh: pd.DataFrame,
        references: References,
        *,
        trunk: pd.DataFrame | None = None,
        calf: pd.DataFrame | None = None,
        arm: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, References]:
        if thigh.empty:
            raise ValueError('Thigh data is empty. Please provide valid thigh data.')

        activities = Thigh(vendor=self.vendor, orientation=self.orientation).detect_activities(
            thigh, references=references
        )
        logger.info('Detected activities for thigh.')

        if isinstance(trunk, pd.DataFrame) and not trunk.empty:
            activities = Trunk(orientation=self.orientation).detect_activities(trunk, activities, references=references)
            logger.info('Detected activities for trunk.')

        if isinstance(calf, pd.DataFrame) and not calf.empty:
            activities = Calf(orientation=self.orientation).detect_activities(calf, activities)
            logger.info('Detected activities for calf.')

        if isinstance(arm, pd.DataFrame) and not arm.empty:
            activities = activities.join(Arm(orientation=self.orientation).detect_activities(arm), how='left')
            logger.info('Detected activities for arm.')

        references.remove_outdated(activities.index[-1])

        return activities, references

    @traceable_logging
    def detect(
        self,
        thigh: pd.DataFrame,
        *,
        trunk: pd.DataFrame | None = None,
        calf: pd.DataFrame | None = None,
        arm: pd.DataFrame | None = None,
        references: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, References]:
        references = References.from_dict(references)  # type: References
        references.remove_outdated(thigh.index[0])

        if self.chunks:
            activities, references = self._detect_chunks(
                thigh=thigh,
                trunk=trunk,
                calf=calf,
                arm=arm,
                references=references,
            )
        else:
            activities, references = self._detect(
                thigh=thigh,
                trunk=trunk,
                calf=calf,
                arm=arm,
                references=references,
            )

        return activities, references.to_dict()

    def _map_activities(
        self,
        series: pd.Series,
        input: Literal['text', 'numeric'] = 'text',
    ) -> pd.Series:
        match input:
            case 'text':
                code_to_activity = {v: k for k, v in ACTIVITIES.items()}
                series = series.map(code_to_activity)
            case 'numeric':
                series = series.map(ACTIVITIES)
            case _:
                raise ValueError("Input must be either 'text' or 'numeric'.")

        if series.isna().all():
            raise ValueError('No valid activities found in the series. Please check the input data.')

        return series

    def to_sens(
        self,
        df: pd.DataFrame,
        references: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = SENS__ACTIVITY_VALUES
        origin = df
        df = pd.DataFrame(index=origin.index)

        # "activity" is not part of SENS__ACTIVITY_VALUES, because it is always present in the dataframe
        for value in values:
            if value not in origin.columns:
                df[value] = 0  # If column is missing, fill with 0. Usually because not all sensors are present
            else:
                df[value] = origin[value]

        df[values] = df[values].fillna(0)  # Fill NaN values with 0
        df['activity'] = self._map_activities(origin['activity'], 'text').astype(np.int16)  # From the origin dataframe

        df[values] = (df[values] * SENS__FLOAT_FACTOR).astype(np.int32)
        df['verbose'] = 0
        df['verbose'] = df['verbose'].astype(np.int32)

        df.index = df.index.astype(np.int64) // 10**6  # Time in milliseconds

        return (
            df.index.values,
            df['activity'].values,
            df[values].values,
            df['verbose'].values,
            references,
        )

    def _parse_sensor_features(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
    ) -> pd.DataFrame:
        df = pd.DataFrame(data, index=timestamps, columns=FEATURES)
        df.index = pd.to_datetime(df.index, unit='ms')  # type: ignore
        df.index.name = 'datetime'  # type: ignore

        df[FEATURES] = (df[FEATURES] / SENS__FLOAT_FACTOR).astype(np.float32)

        return df

    def _raw_from_sens(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        thigh, trunk, calf, arm = None, None, None, None

        if timestamps[0] is not None and data[0] is not None:
            thigh = self._parse_sensor_features(timestamps[0], data[0][:, 1:])

        if timestamps[1] is not None and data[1] is not None:
            trunk = self._parse_sensor_features(timestamps[1], data[1][:, 1:])

        if timestamps[2] is not None and data[2] is not None:
            calf = self._parse_sensor_features(timestamps[2], data[2][:, 1:])

        if timestamps[3] is not None and data[3] is not None:
            arm = self._parse_sensor_features(timestamps[3], data[3][:, 1:])

        return thigh, trunk, calf, arm

    def detect_sens(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
        references: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        thigh, trunk, calf, arm = self._raw_from_sens(timestamps, data)

        df, references = self.detect(thigh=thigh, trunk=trunk, calf=calf, arm=arm, references=references)

        return self.to_sens(df, references)
