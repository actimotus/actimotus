"""Directed rotational crossings: `low` and `high` must not be the same series."""
import numpy as np
import pandas as pd
from actimotus.classifications.thigh import Thigh

# `_get_rotational_crossing_points` never touches `self`, and Thigh's constructor
# requires sensor configuration this test has no opinion about -- so it is called
# unbound, which keeps the test about the crossing logic and nothing else.
_crossings = Thigh._get_rotational_crossing_points


def _df(angles_deg):
    """y/z laid out so |degrees(arcsin(y/hypot(y,z)))| equals the given angles."""
    a = np.radians(np.asarray(angles_deg, dtype=float))
    return pd.DataFrame({"y": np.sin(a), "z": np.cos(a)},
                        index=pd.date_range("2018-01-01", periods=len(a), freq="1s"))


def test_low_and_high_are_not_the_same_series():
    """The regression this fixes: .diff() of a boolean and of its complement are
    identical, which made the `low & high` conjunction in get_lie vacuous."""
    out = _crossings(None, _df([10, 80, 10, 80, 10]), 65)
    assert not out["low"].equals(out["high"])


def test_high_marks_upward_crossings_and_low_marks_downward():
    out = _crossings(None, _df([10, 80, 10]), 65)
    assert out["high"].tolist() == [False, True, False]
    assert out["low"].tolist() == [False, False, True]


def test_a_single_upward_crossing_yields_no_downward_crossing():
    """A bout that only rotates up has not rotated down and back; get_lie's
    conjunction must be able to see the difference."""
    out = _crossings(None, _df([10, 10, 80, 80]), 65)
    assert out["high"].any()
    assert not out["low"].any()


def test_movement_below_the_noise_margin_is_not_a_crossing():
    out = _crossings(None, _df([64.99, 65.0]), 65)
    assert not out["high"].any()
