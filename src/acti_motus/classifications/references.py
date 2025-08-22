import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Literal, Self

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SECOND = timedelta(seconds=1)
MIN_CALIBRATION_SECONDS = 10


class AngleStatus(Enum):
    PROPAGATED = 'propagated'
    CALIBRATION = 'calibration'
    DEFAULT = 'default'


@dataclass
class Angle:
    value: float | list[float] | np.ndarray
    expires: datetime | None = None
    status: AngleStatus | None = None

    def __post_init__(self) -> Self:
        """Convert expires to appropriate types."""
        self.expires = pd.to_datetime(self.expires) if self.expires else None

        return self

    def _updated_bouts_by_angle(self, bouts: pd.DataFrame) -> pd.DataFrame:
        if not self.expires:
            return bouts

        updated_bouts = []
        angle = self

        for start, end, non_wear in bouts.to_numpy():
            if non_wear:
                angle = None
                updated_bouts.append({'start': start, 'end': end, 'non_wear': True, 'angle': angle})

            else:
                # If the angle expires in the bout, split the bout, else keep it as is.
                if self.expires > start and self.expires < end:
                    updated_bouts.append({'start': start, 'end': self.expires, 'non_wear': False, 'angle': angle})
                    angle = None
                    updated_bouts.append(
                        {'start': self.expires + SECOND, 'end': end, 'non_wear': False, 'angle': angle}
                    )

                else:
                    updated_bouts.append({'start': start, 'end': end, 'non_wear': False, 'angle': angle})

        return pd.DataFrame(updated_bouts)

    def is_expired(self, date: datetime) -> bool:
        """Check if the angle is expired based on the given date."""
        if self.expires is None:
            return False
        return self.expires < date

    def fix_expiration(self, calibrations: list['Calibration']) -> None:
        for calibration in calibrations:
            if self.expires and calibration.start < self.expires:
                self.expires = calibration.start - SECOND
                logger.info(f'Angle expiration adjusted to {self.expires} due to calibration overlap.')
                break


@dataclass
class Calibration:
    start: datetime
    end: datetime
    ttl: timedelta | None = None
    expires: datetime | None = None

    def __post_init__(self):
        """Convert start, end, ttl, and expires to appropriate types."""
        self.start = pd.to_datetime(self.start)
        self.end = pd.to_datetime(self.end)
        self.ttl = pd.Timedelta(self.ttl) if self.ttl else None

        self.expires = pd.to_datetime(self.expires) if self.expires else None
        self.expires = self.end + self.ttl if self.ttl and not self.expires else self.expires

    def is_outdated(self, date: datetime) -> bool:
        """Check if the calibration is outdated based on the given date."""
        return self.end < date

    def get_overlap_in_seconds(self, start: datetime, end: datetime):
        """Calculate the overlap in seconds between the calibration and a given interval."""
        overlap_start = max(self.start, start)
        overlap_end = min(self.end, end)

        if overlap_start < overlap_end:
            return (overlap_end - overlap_start).total_seconds()
        return 0

    def get_calibration_period(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.start : self.end]


@dataclass
class References:
    calibrations: list[Calibration] | None = None
    thigh: Angle | None = None
    trunk: Angle | None = None

    def __post_init__(self) -> Self:
        """Initialize the References object."""
        if self.calibrations:
            self.calibrations = [Calibration(**cal) if isinstance(cal, dict) else cal for cal in self.calibrations]
            self.sort_calibrations()
            self.fix_calibration_overlaps()

        """Initialize the Angles object."""
        if isinstance(self.thigh, dict):
            self.thigh = Angle(**self.thigh)

            if self.calibrations:
                self.thigh.fix_expiration(self.calibrations)

        if isinstance(self.trunk, dict):
            self.trunk = Angle(**self.trunk)

            if self.calibrations:
                self.trunk.fix_expiration(self.calibrations)

        return self

    @classmethod
    def from_dict(cls, references: dict[str, Any] | None) -> Self:
        """Create a References object from a dictionary."""
        if references is None:
            logger.info('No references provided, returning empty References object.')
            return cls()

        return cls(**references)

    def to_dict(self) -> dict[str, Any]:
        """Convert the References object to a dictionary."""
        references = {}

        if self.calibrations:
            references['calibrations'] = [cal.__dict__ for cal in self.calibrations]
        if self.thigh:
            references['thigh'] = self.thigh.__dict__
        if self.trunk:
            references['trunk'] = self.trunk.__dict__

        return references

    def sort_calibrations(self):
        """Sort calibrations by start time."""
        if self.calibrations:
            self.calibrations.sort(key=lambda x: x.start)

    def fix_calibration_overlaps(self):
        """Fix overlapping calibration intervals."""
        second = timedelta(seconds=1)
        for index, interval in enumerate(self.calibrations):
            next_interval = self.calibrations[index + 1] if index + 1 < len(self.calibrations) else None

            if next_interval:
                if interval.expires > next_interval.start:
                    interval.expires = next_interval.start - second

    def remove_outdated_calibrations(self, date: datetime) -> None:
        """Remove calibrations that are outdated based on the dataframe start time."""
        if self.calibrations:
            for calibration in self.calibrations:
                if calibration.is_outdated(date):
                    logger.info(
                        f'Calibration {calibration.start} - {calibration.end} is outdated and has been removed.'
                    )
                    self.calibrations.remove(calibration)

    def remove_expired_angles(self, date: datetime) -> None:
        """Remove angles that have expired based on the dataframe start time."""
        if self.thigh and self.thigh.is_expired(date):
            self.thigh = None
            logger.info('Thigh angle reference has expired and has been removed.')

        if self.trunk and self.trunk.is_expired(date):
            self.trunk = None
            logger.info('Trunk angle reference has expired and has been removed.')

    def remove_outdated(self, date: datetime) -> None:
        """Remove outdated references based on the dataframe start time."""
        self.remove_outdated_calibrations(date)
        self.remove_expired_angles(date)

    def _update_bout_by_calibrations(self, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
        calibration_bouts = []
        for calibration in self.calibrations:
            # Check if the calibration period is within the wear bout.
            if calibration.get_overlap_in_seconds(start, end) > MIN_CALIBRATION_SECONDS:
                bout_start, bout_end = start, end

                # If calibration starts after the wear bout starts, adjust the start time.
                if calibration.start > start:
                    bout_start = calibration.start

                # If the calibration angle expires before the wear bout ends, adjust the end time.
                if calibration.expires and calibration.expires < end:
                    bout_end = calibration.expires

                calibration_bouts.append(
                    {
                        'start': bout_start,
                        'end': bout_end,
                        'non_wear': False,
                        'calibration': calibration,
                    }
                )

        return calibration_bouts

    def _fill_bout_gaps(
        self,
        calibration_bouts: list[dict[str, Any]],
        start: pd.Timestamp,
        end: pd.Timestamp,
        angle: Angle | None,
    ) -> list[dict[str, Any]]:
        new_bouts = []
        new_start = start
        for calibration_bout in calibration_bouts:
            if new_start < calibration_bout['start']:
                new_bouts.append(
                    {
                        'start': new_start,
                        'end': calibration_bout['start'] - SECOND,
                        'non_wear': False,
                        'angle': angle,
                    }
                )

            new_start = calibration_bout['end'] + SECOND

        if new_start < end:
            new_bouts.append(
                {
                    'start': new_start,
                    'end': end,
                    'non_wear': False,
                }
            )

        new_bouts.extend(calibration_bouts)
        new_bouts.sort(key=lambda x: x['start'])

        return new_bouts

    def _updated_bouts_with_calibrations(self, bouts: pd.DataFrame) -> pd.DataFrame:
        if 'angle' not in bouts.columns:
            bouts['angle'] = None

        updated_bouts = []

        for start, end, non_wear, angle in bouts.to_numpy():
            if non_wear:
                updated_bouts.append({'start': start, 'end': end, 'non_wear': True, 'angle': angle})

            else:
                if self.calibrations:
                    calibration_bouts = self._update_bout_by_calibrations(start, end)
                    if calibration_bouts:
                        new_bouts = self._fill_bout_gaps(calibration_bouts, start, end, angle)
                        updated_bouts.extend(new_bouts)
                    else:
                        updated_bouts.append({'start': start, 'end': end, 'non_wear': False, 'angle': angle})
                else:
                    updated_bouts.append({'start': start, 'end': end, 'non_wear': False, 'angle': angle})

        updated_bouts = pd.DataFrame(updated_bouts)

        return updated_bouts

    def _get_non_wear_bouts(self, non_wear: pd.Series) -> pd.DataFrame:
        bouts = (non_wear != non_wear.shift()).cumsum().to_frame(name='bouts')
        bouts['non_wear'] = non_wear
        bouts = (
            bouts.reset_index()
            .groupby('bouts')
            .agg(
                start=('datetime', 'first'),
                end=('datetime', 'last'),
                non_wear=('non_wear', 'first'),
            )
        )

        return bouts

    def get_bouts(self, non_wear: pd.Series, angle: Literal['thigh', 'trunk'] | None = None) -> pd.DataFrame:
        bouts = self._get_non_wear_bouts(non_wear)

        if angle:
            angle = getattr(self, angle, None)
            bouts = angle._updated_bouts_by_angle(bouts) if isinstance(angle, Angle) else bouts

        bouts = self._updated_bouts_with_calibrations(bouts) if self.calibrations else bouts

        for col in ['calibration', 'angle']:
            if col not in bouts.columns:
                bouts[col] = None

            bouts[col] = bouts[col].replace(np.nan, None)

        return bouts[['start', 'end', 'non_wear', 'calibration', 'angle']]

    def update_angle(self, df: pd.DataFrame, angle: Literal['thigh', 'trunk']) -> None:
        last_angle = df['angle'].values[-1]

        if not isinstance(last_angle, Angle) or (
            last_angle.status not in [AngleStatus.PROPAGATED, AngleStatus.CALIBRATION]
        ):
            last_angle = None

        setattr(self, angle, last_angle)
