import logging
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from .classifications import Arm, Calf, References, Thigh, Trunk

logger = logging.getLogger(__name__)


@dataclass
class Activities:
    vendor: Literal['Sens', 'Other'] = 'Other'

    def detect(
        self,
        thigh: pd.DataFrame,
        *,
        trunk: pd.DataFrame | None = None,
        calf: pd.DataFrame | None = None,
        arm: pd.DataFrame | None = None,
        references: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, References]:
        if thigh.empty:
            raise ValueError('Thigh data is empty. Please provide valid thigh data.')

        references = references or References()  # type: References
        references.remove_outdated(thigh.index[0])

        activities = Thigh(self.vendor).detect_activities(thigh, references=references)
        logger.info('Detected activities for thigh.')

        if isinstance(trunk, pd.DataFrame) and not trunk.empty:
            activities = Trunk().detect_activities(trunk, activities, references=references)
            logger.info('Detected activities for trunk.')

        if isinstance(calf, pd.DataFrame) and not calf.empty:
            activities = Calf().detect_activities(calf, activities)
            logger.info('Detected activities for calf.')

        if isinstance(arm, pd.DataFrame) and not arm.empty:
            activities = activities.join(Arm().detect_activities(arm), how='left')
            logger.info('Detected activities for arm.')

        references.remove_outdated(activities.index[-1])

        return activities, references
