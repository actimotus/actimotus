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


class TestContextMask:
    def test_interval_only(self, activities, diary_factory):
        # 07:00:02 .. 07:00:05 -> epochs at :02, :03, :04 (half-open)
        diary = diary_factory([('2024-09-02 07:00:02', '2024-09-02 07:00:05', 'work', None)])
        mask = Exposures._context_mask(activities, diary)
        assert list(mask) == [False, False, True, True, True, False, False, False, False, False]

    def test_activity_gate(self, activities, diary_factory):
        # Same window, but only count 'sit' epochs: :02 and :03 are sit, :04 is lie
        diary = diary_factory([('2024-09-02 07:00:02', '2024-09-02 07:00:05', 'work', ['sit'])])
        mask = Exposures._context_mask(activities, diary)
        assert list(mask) == [False, False, True, True, False, False, False, False, False, False]

    def test_empty_activity_list_is_pure_interval(self, activities, diary_factory):
        diary = diary_factory([('2024-09-02 07:00:02', '2024-09-02 07:00:05', 'work', [])])
        mask = Exposures._context_mask(activities, diary)
        assert list(mask) == [False, False, True, True, True, False, False, False, False, False]

    def test_multiple_intervals_union(self, activities, diary_factory):
        diary = diary_factory([
            ('2024-09-02 07:00:00', '2024-09-02 07:00:02', 'sleep', None),
            ('2024-09-02 07:00:08', '2024-09-02 07:00:10', 'sleep', None),
        ])
        mask = Exposures._context_mask(activities, diary)
        assert list(mask) == [True, True, False, False, False, False, False, False, True, True]

    def test_half_open_end_excluded(self, activities, diary_factory):
        # Interval end at :05 must exclude the :05 epoch
        diary = diary_factory([('2024-09-02 07:00:03', '2024-09-02 07:00:05', 'work', None)])
        mask = Exposures._context_mask(activities, diary)
        assert mask.iloc[5] == False
        assert mask.iloc[3] == True
        assert mask.iloc[4] == True

    def test_empty_intervals_all_false(self, activities, diary_factory):
        diary = diary_factory([])
        mask = Exposures._context_mask(activities, diary)
        assert list(mask) == [False] * 10
        assert mask.dtype == bool

    def test_nan_activities_is_pure_interval(self, activities, diary_factory):
        diary = diary_factory([('2024-09-02 07:00:02', '2024-09-02 07:00:05', 'work', float('nan'))])
        mask = Exposures._context_mask(activities, diary)
        assert list(mask) == [False, False, True, True, True, False, False, False, False, False]


class TestContext:
    def test_adds_prefixed_boolean_columns(self, activities, diary_factory):
        diary = diary_factory([
            ('2024-09-02 07:00:00', '2024-09-02 07:00:02', 'sleep', ['lie', 'sit']),
            ('2024-09-02 07:00:02', '2024-09-02 07:00:05', 'work', None),
        ])
        result = Exposures.context(activities, diary)
        assert 'context__sleep' in result.columns
        assert 'context__work' in result.columns
        # :00 and :01 are 'walk' -> gated out of sleep
        assert list(result['context__sleep']) == [False] * 10
        assert list(result['context__work']) == \
            [False, False, True, True, True, False, False, False, False, False]

    def test_overlapping_contexts_both_true(self, activities, diary_factory):
        diary = diary_factory([
            ('2024-09-02 07:00:00', '2024-09-02 07:00:10', 'work-day', None),
            ('2024-09-02 07:00:02', '2024-09-02 07:00:04', 'commute', None),
        ])
        result = Exposures.context(activities, diary)
        assert result['context__work-day'].iloc[2] == True
        assert result['context__commute'].iloc[2] == True

    def test_multiple_intervals_one_column(self, activities, diary_factory):
        diary = diary_factory([
            ('2024-09-02 07:00:00', '2024-09-02 07:00:02', 'sleep', None),
            ('2024-09-02 07:00:08', '2024-09-02 07:00:10', 'sleep', None),
        ])
        result = Exposures.context(activities, diary)
        sleep_cols = [c for c in result.columns if c.startswith('context__sleep')]
        assert sleep_cols == ['context__sleep']
        assert list(result['context__sleep']) == \
            [True, True, False, False, False, False, False, False, True, True]

    def test_does_not_mutate_input(self, activities, diary_factory):
        diary = diary_factory([('2024-09-02 07:00:00', '2024-09-02 07:00:05', 'work', None)])
        original_columns = list(activities.columns)
        Exposures.context(activities, diary)
        assert list(activities.columns) == original_columns

    def test_empty_diary_returns_copy_unchanged(self, activities, diary_factory):
        diary = diary_factory([('2024-09-02 07:00:00', '2024-09-02 07:00:05', 'work', None)])
        diary = diary.iloc[0:0]  # zero rows, dtypes preserved
        result = Exposures.context(activities, diary)
        assert list(result.columns) == list(activities.columns)
        assert result is not activities

    def test_timezone_mismatch_raises(self, activities, diary_factory):
        diary = diary_factory(
            [('2024-09-02 07:00:00', '2024-09-02 07:00:05', 'work', None)],
            tz='America/New_York',
        )
        with pytest.raises(ValueError, match='timezone'):
            Exposures.context(activities, diary)
