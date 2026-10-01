"""`Exposures(intensity=...)`: which activities count as sedentary, light and moderate-to-vigorous.

Every activity must be in exactly one of `sedentary`, `lpa`, `mvpa` and `none`, so the four bands
add up to the whole recording. A mapping that breaks this is refused at construction, before it can
give wrong minutes.
"""

import pandas as pd
import pytest

from actimotus import Exposures
from actimotus.settings import ACTIVITIES, INTENSITY

TZ = 'Europe/Copenhagen'


def _day(minutes: dict[str, int]) -> pd.DataFrame:
    labels = [activity for activity, n in minutes.items() for _ in range(n * 60)]
    index = pd.date_range('2024-09-02 08:00:00', periods=len(labels), freq='1s', tz=TZ, name='datetime')
    return pd.DataFrame({'activity': pd.Categorical(labels, categories=list(ACTIVITIES.values()))}, index=index)


def test_the_default_bands_cover_every_activity_once():
    listed = [a for band in INTENSITY.values() for a in band]
    assert sorted(listed) == sorted(ACTIVITIES.values())


def test_the_bands_add_up_to_the_recording():
    out = Exposures().compute(_day({a: 1 for a in ACTIVITIES.values()})).iloc[0]
    none = len(INTENSITY['none'])
    assert out['sedentary'] + out['lpa'] + out['mvpa'] == pd.Timedelta(minutes=len(ACTIVITIES) - none)


def test_a_custom_mapping_moves_standing_into_sedentary():
    intensity = {**INTENSITY, 'sedentary': [*INTENSITY['sedentary'], 'stand'], 'none': ['non-wear']}
    day = _day({'sit': 2, 'stand': 3, 'walk': 1})
    assert Exposures().compute(day).iloc[0]['sedentary'] == pd.Timedelta(minutes=2)
    out = Exposures(intensity=intensity).compute(day).iloc[0]
    assert out['sedentary'] == pd.Timedelta(minutes=5)
    assert out['mvpa'] == pd.Timedelta(minutes=1)


def test_sedentary_transitions_follow_the_mapping():
    intensity = {**INTENSITY, 'sedentary': [*INTENSITY['sedentary'], 'stand'], 'none': ['non-wear']}
    labels = ['sit'] * 60 + ['walk'] * 60 + ['stand'] * 60 + ['walk'] * 60
    index = pd.date_range('2024-09-02 08:00:00', periods=len(labels), freq='1s', tz=TZ, name='datetime')
    day = pd.DataFrame({'activity': pd.Categorical(labels, categories=list(ACTIVITIES.values()))}, index=index)
    assert Exposures().compute(day).iloc[0]['sedentary_to_other'] == 1
    assert Exposures(intensity=intensity).compute(day).iloc[0]['sedentary_to_other'] == 2


@pytest.mark.parametrize(
    'intensity, message',
    [
        ({**INTENSITY, 'none': ['non-wear']}, 'missing'),
        ({**INTENSITY, 'lpa': [*INTENSITY['lpa'], 'walk']}, 'two intensity bands'),
        ({**INTENSITY, 'none': [*INTENSITY['none'], 'swim']}, 'unknown activity'),
        ({k: v for k, v in INTENSITY.items() if k != 'none'}, 'exactly the keys'),
        ({**INTENSITY, 'light': []}, 'exactly the keys'),
        ({**INTENSITY, 'none': 'non-wear'}, 'list of activity names'),
    ],
    ids=['missing', 'duplicate', 'unknown', 'no-none', 'extra-key', 'string'],
)
def test_a_broken_mapping_is_refused(intensity, message):
    with pytest.raises(ValueError, match=message):
        Exposures(intensity=intensity)


def test_the_default_is_not_shared():
    exposures = Exposures()
    exposures.intensity['sedentary'].append('stand')
    assert 'stand' not in INTENSITY['sedentary']
