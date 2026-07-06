"""DST calendar-day grouping for `Exposures`.

The window must bucket on LOCAL calendar days, so a DST fall-back day is one 25 h
window and a spring-forward day one 23 h window, with every bin label at local
midnight. This is a pandas-3.0 regression: `to_offset(timedelta(days=1))` is a
calendar `<Day>` on pandas 2.2/2.3 but a fixed `<24 * Hours>` tick on 3.0+, so
these tests only exercise the bug under pandas >= 3.0 (they pass trivially below).
"""

import warnings

import pandas as pd

from actimotus.exposures import Exposures

TZ = 'Europe/Copenhagen'  # same CET/CEST transitions as Europe/Prague


def _walk_frame(start: str, end: str, tz: str = TZ) -> pd.DataFrame:
    """A fully-worn 1 s 'walk' frame; one row == one second of exposure."""
    index = pd.date_range(start, end, freq='1s', tz=tz, name='datetime')
    return pd.DataFrame({'activity': 'walk'}, index=index)


def test_fall_back_day_is_one_25h_window_at_midnight():
    # 2025-10-26 is the 25 h fall-back day in Europe/Copenhagen.
    df = _walk_frame('2025-10-26 00:00:00', '2025-10-27 23:59:59')

    res = Exposures(window='1D').compute(df)

    assert list(res.index.strftime('%Y-%m-%d')) == ['2025-10-26', '2025-10-27']
    assert (res.index.strftime('%H:%M') == '00:00').all()
    assert res.iloc[0]['walk'] == pd.Timedelta(hours=25)
    assert res.iloc[1]['walk'] == pd.Timedelta(hours=24)


def test_spring_forward_day_is_one_23h_window_at_midnight():
    # 2026-03-29 is the 23 h spring-forward day in Europe/Copenhagen.
    df = _walk_frame('2026-03-29 00:00:00', '2026-03-30 23:59:59')

    res = Exposures(window='1D').compute(df)

    assert list(res.index.strftime('%Y-%m-%d')) == ['2026-03-29', '2026-03-30']
    assert (res.index.strftime('%H:%M') == '00:00').all()
    assert res.iloc[0]['walk'] == pd.Timedelta(hours=23)
    assert res.iloc[1]['walk'] == pd.Timedelta(hours=24)


def test_non_dst_week_is_unchanged():
    # A plain stretch with no transition: every day is 24 h, at midnight.
    df = _walk_frame('2025-11-10 00:00:00', '2025-11-12 23:59:59')

    res = Exposures(window='1D').compute(df)

    assert list(res.index.strftime('%Y-%m-%d')) == ['2025-11-10', '2025-11-11', '2025-11-12']
    assert (res.index.strftime('%H:%M') == '00:00').all()
    assert (res['walk'] == pd.Timedelta(hours=24)).all()


def test_weekly_window_does_not_drift_across_dst():
    # Weekly bins spanning the fall-back must still land on local midnight.
    idx = pd.date_range('2025-10-20', '2025-11-02 23:59:00', freq='1min', tz=TZ, name='datetime')
    df = pd.DataFrame({'activity': 'walk'}, index=idx)

    res = Exposures(window='7D').compute(df)

    assert (res.index.strftime('%H:%M') == '00:00').all()


def test_default_window_emits_no_pandas_deprecation_warning():
    # The default window must use the non-deprecated 'D' alias, not lowercase 'd'.
    df = _walk_frame('2025-11-10 00:00:00', '2025-11-10 00:59:59')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Exposures().compute(df)

    assert not any("'d' is deprecated" in str(w.message) for w in caught)
