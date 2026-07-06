"""`get_row` must not fire on an inverted thigh inclination.

`inclination = arccos(x)` runs 0-180deg: standing ~0deg, thigh horizontal ~90deg,
inverted long-axis (x<0) ~90-180deg. Real rowing keeps the thigh at/near horizontal
(~90-100deg); it is never inverted. The rule has a lower inclination bound (87.5deg)
but historically no upper bound, so an upside-down device (or feet-up lying) with
leg motion was misclassified as `row` and folded into MVPA. These tests pin the
upper bound: `87.5deg < inclination < inclination_upper`.
"""

import pandas as pd

from actimotus.classifications.thigh import Thigh

TZ = 'Europe/Copenhagen'
ROW_KW = dict(bout=1, movement_threshold=0.075, inclination_angle=87.5)


def _thigh() -> Thigh:
    # get_row uses only self._median_filter; instance state is irrelevant here.
    return Thigh(system_frequency=12, vendor='Sens', config={}, orientation=False)


def _feat(inclination: float, sd_x: float, n: int = 40) -> pd.DataFrame:
    index = pd.date_range('2024-09-02 07:00:00', periods=n, freq='1s', tz=TZ, name='datetime')
    return pd.DataFrame({'inclination': float(inclination), 'sd_x': float(sd_x)}, index=index)


def test_row_excludes_inverted_inclination():
    # Inverted device: thigh reads 150deg with clear motion. Old rule tagged this
    # `row`; with the upper bound it must not.
    out = _thigh().get_row(_feat(150, 0.2), inclination_upper=110.0, **ROW_KW)
    assert not out.any()


def test_row_keeps_genuine_horizontal_rowing():
    # Thigh at ~horizontal (92deg) with motion is genuine rowing and must survive.
    out = _thigh().get_row(_feat(92, 0.2), inclination_upper=110.0, **ROW_KW)
    assert out.all()


def test_row_upper_bound_is_exclusive_at_the_threshold():
    # Just inside stays row; at/above the bound is excluded.
    assert _thigh().get_row(_feat(109, 0.2), inclination_upper=110.0, **ROW_KW).all()
    assert not _thigh().get_row(_feat(111, 0.2), inclination_upper=110.0, **ROW_KW).any()


def test_row_upper_bound_defaults_to_no_op():
    # Backward compatibility: without inclination_upper the old behaviour holds, so
    # an inverted window is still flagged (default upper bound is a no-op).
    out = _thigh().get_row(_feat(150, 0.2), **ROW_KW)
    assert out.all()
