import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .references import References
from .sensor import Sensor

logger = logging.getLogger(__name__)


@dataclass
class Arm(Sensor):
    def compute_activities(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df.copy()

        references = References()

        df[['inclination', 'side_tilt', 'direction']] = self.get_angles(df)

        # TODO: Is non-wear detection correct for arm? Produces non-wear times even if the arm is worn most probably.
        df['non-wear'] = self.get_non_wear(df)
        bouts = references.get_bouts(df['non-wear'])

        if self.orientation:
            df = self.fix_bouts_orientation(df, bouts)

        df.loc[df['non-wear'], 'inclination'] = np.nan
        df.rename(columns={'inclination': 'arm_inclination'}, inplace=True)

        return pd.DataFrame(df['arm_inclination'])

    def check_inside_out_flip(self, df: pd.DataFrame) -> bool:
        return False

    def rotate_by_reference_angle(self):
        raise NotImplementedError('Calf does not support rotating by reference angle.')

    def calculate_reference_angle(self):
        raise NotImplementedError('Calf does not support calculating reference angles.')
