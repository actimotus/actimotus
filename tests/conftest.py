import pandas as pd
import pytest

TZ = 'Europe/Copenhagen'


@pytest.fixture
def activities():
    """Ten 1-second epochs, 2024-09-02 07:00:00..07:00:09 (Europe/Copenhagen)."""
    index = pd.date_range(
        '2024-09-02 07:00:00', periods=10, freq='1s', tz=TZ, name='datetime'
    )
    labels = ['walk', 'walk', 'sit', 'sit', 'lie', 'lie', 'stand', 'walk', 'sit', 'lie']
    return pd.DataFrame({'activity': labels}, index=index)


def make_diary(rows, tz=TZ):
    """Build a clean diary from (start, end, context, activities) tuples.

    `start`/`end` are naive strings localized to `tz` (pass tz=None to keep naive).
    """
    diary = pd.DataFrame(rows, columns=['start', 'end', 'context', 'activities'])
    diary['start'] = pd.to_datetime(diary['start'])
    diary['end'] = pd.to_datetime(diary['end'])
    if tz is not None:
        diary['start'] = diary['start'].dt.tz_localize(tz)
        diary['end'] = diary['end'].dt.tz_localize(tz)
    return diary


@pytest.fixture
def diary_factory():
    return make_diary
