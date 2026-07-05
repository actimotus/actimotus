import pandas as pd
import pytest

from actimotus import Exposures


class TestValidateDiary:
    def test_missing_column_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', None)])
        diary = diary.drop(columns=['context'])
        with pytest.raises(ValueError, match='missing required columns'):
            Exposures._validate_diary(diary)

    def test_end_not_after_start_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:05', '2024-09-02 07:00', 'work', None)])
        with pytest.raises(ValueError, match="'end'"):
            Exposures._validate_diary(diary)

    def test_naive_timestamps_raise(self, diary_factory):
        diary = diary_factory(
            [('2024-09-02 07:00', '2024-09-02 07:05', 'work', None)], tz=None
        )
        with pytest.raises(ValueError, match='timezone-aware'):
            Exposures._validate_diary(diary)

    def test_valid_diary_passes(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', None)])
        assert Exposures._validate_diary(diary) is None

    def test_equal_timestamps_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:00', 'work', None)])
        with pytest.raises(ValueError, match="'end'"):
            Exposures._validate_diary(diary)

    def test_missing_activities_column_passes(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', None)])
        diary = diary.drop(columns=['activities'])
        assert Exposures._validate_diary(diary) is None
