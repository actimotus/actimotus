"""A sitting bout that rolls past the threshold becomes lying -- in either direction.

`thigh_angle` is the roll of the thigh about its own long axis, taken as an absolute value
so a roll to the left and a roll to the right both read positive. Crossing the threshold is
what separates lying from sitting: the thigh is horizontal in both postures, so inclination
cannot tell them apart, but the front of the thigh faces up when you sit and to the side
when you lie on your side.

Lying down and staying put is one crossing. Requiring an up-crossing AND a down-crossing
would leave that bout classified as sitting, which is the regression these tests pin.
"""

import numpy as np
import pandas as pd
import pytest

from actimotus.classifications.thigh import Thigh

ORIENTATION_ANGLE = 65
BOUT = 1


@pytest.fixture
def thigh() -> Thigh:
    return Thigh(orientation=False, system_frequency=30, vendor='Other', config={})


def _sitting_bout(roll_degrees) -> pd.DataFrame:
    """One unbroken `sit` bout whose thigh roll follows the given angles.

    y/z are laid out so that |degrees(arcsin(y / hypot(y, z)))| reproduces each angle.
    """
    radians = np.radians(np.asarray(roll_degrees, dtype=float))

    return pd.DataFrame(
        {
            'activity': 'sit',
            'y': np.sin(radians),
            'z': np.cos(radians),
        },
        index=pd.date_range('2026-01-01', periods=len(radians), freq='1s'),
    )


def _is_lie(thigh: Thigh, roll_degrees) -> bool:
    lie = thigh.get_lie(_sitting_bout(roll_degrees), bout=BOUT, orientation_angle=ORIENTATION_ANGLE)
    return bool(lie.all())


def test_rolling_onto_the_side_and_staying_is_lying(thigh):
    """The regression. One upward crossing, never coming back -- someone lay down."""
    assert _is_lie(thigh, [10, 10, 10, 80, 80, 80, 80])


def test_rolling_back_up_and_staying_is_lying(thigh):
    """The mirror image: the bout starts rolled over and unrolls. Still one crossing."""
    assert _is_lie(thigh, [80, 80, 80, 10, 10, 10, 10])


def test_rolling_over_and_back_is_lying(thigh):
    """Turning over during the night -- crossings in both directions."""
    assert _is_lie(thigh, [10, 80, 80, 20, 20, 80, 80])


def test_a_roll_to_either_side_counts(thigh):
    """`thigh_angle` is an absolute value, so left and right are the same evidence."""
    left = _sitting_bout([10, 10, 80, 80])
    right = left.copy()
    right['y'] = -right['y']

    for frame in (left, right):
        lie = thigh.get_lie(frame, bout=BOUT, orientation_angle=ORIENTATION_ANGLE)
        assert lie.all()


def test_sitting_upright_throughout_is_not_lying(thigh):
    """Never crosses the threshold, so there is no evidence of a roll."""
    assert not _is_lie(thigh, [5, 12, 20, 8, 15, 11, 9])


def test_lying_still_past_the_threshold_throughout_is_not_detected(thigh):
    """A known limitation, pinned so a future change to it is deliberate.

    With no crossing inside the bout there is nothing for this rule to see -- someone who
    was already rolled over before the bout began is not reclassified.
    """
    assert not _is_lie(thigh, [80, 80, 80, 80, 80, 80, 80])
