"""The daily `valid` flag is walk-only: `walk >= 5 min`.

Previously `valid = (walk + stairs) >= 10 min`. Stairs on a thigh sensor is easily
confused with walking (a mounting/reference-angle-sensitive fore-aft split), so it
can mask a day where real walking was suppressed by an orientation artifact — the
walk+stairs sum stays high while walk itself is ~0. A functional wearer walks at
least a few minutes a day, so `walk >= 5 min` is a stricter, harder-to-fool
data-quality floor.
"""

import pandas as pd

from actimotus.exposures import Exposures

TZ = 'Europe/Copenhagen'


def _acts(**seconds) -> pd.DataFrame:
    """A 1 s activity frame: one row == one second of the named activity."""
    labels = []
    for act, n in seconds.items():
        labels += [act] * n
    index = pd.date_range('2024-09-02 00:00:00', periods=len(labels), freq='1s', tz=TZ, name='datetime')
    return pd.DataFrame({'activity': labels}, index=index)


def test_valid_true_with_5min_walk_and_no_stairs():
    # 6 min walk, no stairs: old rule (walk+stairs >= 10 min) = invalid; walk-only = valid.
    res = Exposures(window='1D').compute(_acts(walk=6 * 60, sit=200))
    assert res['valid'].all()


def test_valid_at_exactly_5min_walk():
    res = Exposures(window='1D').compute(_acts(walk=5 * 60, sit=200))
    assert res['valid'].all()


def test_invalid_when_walk_under_5min_despite_stairs():
    # 4 min walk + 20 min stairs: old rule = valid (24 >= 10); walk-only = invalid.
    # Stairs must not rescue a day where walking was suppressed.
    res = Exposures(window='1D').compute(_acts(walk=4 * 60, stairs=20 * 60, sit=200))
    assert not res['valid'].any()
