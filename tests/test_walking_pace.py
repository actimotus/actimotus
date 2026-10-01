"""Three walking paces, `slow-walk`, `walk` and `fast-walk`, and every place that reads them.

Both presets split walking by its step rate: `slow-walk` below 100 steps a minute, `walk` from 100
to below 115, `fast-walk` from 115. DEFAULT judges each second, LEGACY the minute around it.
`slow-walk` is light; `walk` and `fast-walk` are moderate.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from actimotus.activities import Activities
from actimotus.classifications.thigh import Thigh
from actimotus.classifications.trunk import Trunk
from actimotus.exposures import Exposures
from actimotus.settings import (
    ACTIVITIES,
    CONFIG,
    FUSED_ACTIVITIES,
    LEGACY_CONFIG,
    PLOT,
    PLOT_FUSED,
)

TZ = 'Europe/Copenhagen'
WALKING = ['slow-walk', 'walk', 'fast-walk']
ORDER = [
    'non-wear', 'lie', 'sit', 'kneel', 'squat', 'stand', 'shuffle',
    'slow-walk', 'walk', 'fast-walk', 'run', 'stairs', 'bicycle', 'row',
]  # fmt: skip


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range('2024-09-02 07:00:00', periods=n, freq='1s', tz=TZ)


def _thigh(config=CONFIG) -> Thigh:
    return Thigh(system_frequency=30, vendor='Other', config=config, orientation=False)


def _walking(spm: Sequence[float], activity: str = 'walk') -> pd.DataFrame:
    categories = ['row', 'bicycle', 'stairs', 'run', 'walk', 'stand', 'sit', 'shuffle', 'lie', 'non-wear']
    return pd.DataFrame(
        {
            'activity': pd.Categorical([activity] * len(spm), categories=categories),
            'walk_feature': np.asarray(spm, dtype=np.float32) / 60,
            'run_feature': np.zeros(len(spm), dtype=np.float32),
        },
        index=_index(len(spm)),
    )


class TestThePreset:
    def test_default_judges_each_second_at_100_and_115_steps_a_minute(self):
        assert CONFIG['thigh']['pace'] == {'slow': 100, 'fast': 115, 'window': 1}

    def test_legacy_has_the_same_paces_judged_over_a_minute(self):
        """LEGACY's light/moderate edge was 100 steps a minute over 60 s; it keeps both."""
        assert LEGACY_CONFIG['thigh']['pace'] == {'slow': 100, 'fast': 115, 'window': 60}

    @pytest.mark.parametrize('config', [CONFIG, LEGACY_CONFIG], ids=['DEFAULT', 'LEGACY'])
    def test_no_preset_turns_walking_into_running_by_its_step_rate(self, config):
        assert 'steps' not in config['thigh']['run']

    @pytest.mark.parametrize('config', [CONFIG, LEGACY_CONFIG], ids=['DEFAULT', 'LEGACY'])
    def test_no_preset_keeps_the_old_fast_walk_keys(self, config):
        assert 'fast-walk' not in config['thigh']
        assert 'slow-walk' not in config['thigh']

    def test_a_config_without_pace_is_refused_with_a_reason(self):
        old = {'fast-walk': {'bout': 15, 'steps': 27}}

        with pytest.raises(ValueError, match="'pace'"):
            Thigh.pace_settings(old)

    def test_the_old_rule_is_gone(self):
        assert not hasattr(Thigh, 'get_fast_walking_and_running')
        assert not hasattr(Thigh, 'get_steps')


class TestThreePaces:
    @pytest.mark.parametrize(
        ('spm', 'expected'),
        [
            (60, 'slow-walk'),
            (99.9, 'slow-walk'),
            (100.1, 'walk'),
            (114.9, 'walk'),
            (115.1, 'fast-walk'),
            (140, 'fast-walk'),
        ],
    )
    def test_each_second_takes_the_pace_of_its_step_rate(self, spm: float, expected: str):
        """Each edge is tested 0.1 either side: `walk_feature` is float32, so 100 itself reads 99.99999."""
        df = _walking([spm] * 5)

        _thigh().get_walking_pace(df, slow=100, fast=115)

        assert df['activity'].iloc[2] == expected

    def test_there_is_no_window(self):
        """One second at its own pace is kept: the rule was tuned with no window (W = 1)."""
        df = _walking([90] * 4 + [108] * 3 + [90] * 4)

        _thigh().get_walking_pace(df, slow=100, fast=115)

        assert df['activity'].astype(str).tolist() == ['slow-walk'] * 4 + ['walk'] * 3 + ['slow-walk'] * 4

    def test_a_single_wrong_second_is_removed_by_the_median_of_three(self):
        df = _walking([108, 108, 50, 108, 108])

        _thigh().get_walking_pace(df, slow=100, fast=115)

        assert set(df['activity'].astype(str)) == {'walk'}

    @pytest.mark.parametrize('activity', ['run', 'stairs', 'stand', 'sit'])
    def test_only_walking_is_split(self, activity: str):
        df = _walking([60] * 5, activity=activity)

        _thigh().get_walking_pace(df, slow=100, fast=115)

        assert set(df['activity'].astype(str)) == {activity}

    def test_fast_walking_stays_walking_however_fast(self):
        """No step-rate rule turns walking into running."""
        df = _walking([190] * 5)

        _thigh().get_walking_pace(df, slow=100, fast=115)

        assert set(df['activity'].astype(str)) == {'fast-walk'}


class TestTheWindow:
    def test_a_minute_window_judges_a_short_burst_by_the_minute_around_it(self):
        """Five fast seconds inside slow walking: fast on their own, slow over a minute."""
        spm = [95] * 60 + [130] * 5 + [95] * 60

        alone, minute = _walking(spm), _walking(spm)
        _thigh().get_walking_pace(alone, slow=100, fast=115, window=1)
        _thigh().get_walking_pace(minute, slow=100, fast=115, window=60)

        assert alone['activity'].iloc[62] == 'fast-walk'
        assert set(minute['activity'].astype(str)) == {'slow-walk'}

    def test_the_window_averages_walking_seconds_only(self):
        """Sitting has a step rate of 0; counted, it would drag the walking beside it to slow."""
        df = _walking([0] * 60 + [108] * 30)
        df.iloc[:60, df.columns.get_loc('activity')] = 'sit'

        _thigh().get_walking_pace(df, slow=100, fast=115, window=60)

        assert set(df['activity'].iloc[60:].astype(str)) == {'walk'}


class TestTheClassIsKnownEverywhere:
    def test_the_codes_run_from_still_to_active(self):
        assert list(ACTIVITIES.values()) == ORDER
        assert list(ACTIVITIES) == list(range(len(ORDER)))

    def test_the_legend_has_the_same_order(self):
        assert list(PLOT['activities']) == ORDER

    def test_every_walking_pace_fuses_to_walk(self):
        assert all(FUSED_ACTIVITIES.get(pace, pace) == 'walk' for pace in WALKING)

    def test_every_activity_has_a_plot_entry(self):
        assert set(ACTIVITIES.values()) <= set(PLOT['activities'])

    def test_every_fused_activity_has_a_fused_plot_entry(self):
        fused = {FUSED_ACTIVITIES.get(a, a) for a in ACTIVITIES.values()}
        assert fused <= set(PLOT_FUSED['activities'])

    def test_the_sens_code_goes_both_ways(self):
        series = pd.Series(WALKING, index=_index(3))
        model = Activities()

        codes = model._map_activities(series, 'text')
        back = model._map_activities(codes, 'numeric')

        assert codes.tolist() == [7, 8, 9]
        assert back.tolist() == WALKING


class TestExposures:
    def _day(self, minutes: dict[str, int]) -> pd.DataFrame:
        labels = [a for a, m in minutes.items() for _ in range(m * 60)]
        return pd.DataFrame({'activity': labels}, index=_index(len(labels)))

    def test_slow_walk_is_light_and_the_other_paces_are_moderate(self):
        out = Exposures().compute(self._day({'slow-walk': 3, 'walk': 4, 'fast-walk': 5}))

        assert out['lpa'].iloc[0] == pd.Timedelta(minutes=3)
        assert out['mvpa'].iloc[0] == pd.Timedelta(minutes=9)

    def test_every_pace_is_on_feet(self):
        out = Exposures().compute(self._day({'slow-walk': 3, 'walk': 4, 'fast-walk': 5}))

        assert out['on_feet'].iloc[0] == pd.Timedelta(minutes=12)

    def test_a_day_is_valid_on_walking_at_any_pace(self):
        """Older adults walk slowly: a day of only `slow-walk` is still a day of walking."""
        out = Exposures().compute(self._day({'slow-walk': 5}))

        assert bool(out['valid'].iloc[0])

    def test_the_activity_columns_follow_the_codes(self):
        out = Exposures().compute(self._day({'run': 1, 'fast-walk': 1, 'sit': 1, 'slow-walk': 1}))

        assert [c for c in out.columns if c in ORDER] == [a for a in ORDER if a != 'non-wear']

    def test_slow_walk_has_its_own_column(self):
        out = Exposures().compute(self._day({'slow-walk': 2, 'walk': 1}))

        assert out['slow-walk'].iloc[0] == pd.Timedelta(minutes=2)

    def test_fused_exposures_count_every_pace_as_walk(self):
        day = self._day({'slow-walk': 2, 'walk': 3, 'fast-walk': 4})

        plot = Exposures(fused=True).plot(day)

        assert plot is not None
        assert day['activity'].replace(FUSED_ACTIVITIES).eq('walk').all()

    def test_the_plot_draws_slow_walk(self):
        assert Exposures().plot(self._day({'slow-walk': 1, 'walk': 1})) is not None

    @pytest.mark.parametrize('pace', WALKING)
    def test_bending_counts_every_pace(self, pace: str):
        df = pd.DataFrame(
            {'activity': [pace] * 3, 'trunk_direction': [10.0] * 3, 'trunk_inclination': [45.0] * 3},
            index=_index(3),
        )

        assert Exposures().get_bending(df, 30, 60).all()

    @pytest.mark.parametrize('pace', WALKING)
    def test_arm_lifting_counts_every_pace(self, pace: str):
        df = pd.DataFrame({'activity': [pace] * 3, 'arm_inclination': [45.0] * 3}, index=_index(3))

        assert Exposures().get_arm_lifting(df, 30, 60).all()


class TestTrunk:
    def test_the_reference_angle_is_taken_from_walking_at_any_pace(self):
        """The trunk's reference angle comes from walking seconds. Without `slow-walk` in that set, an
        older adult who walks only slowly would get the default angle."""
        n = 30
        df = pd.DataFrame(
            {
                'activity': ['slow-walk'] * n,
                'non-wear': [False] * n,
                'direction': [20.0] * n,
                'side_tilt': [0.0] * n,
            },
            index=_index(n),
        )

        _, status = Trunk(orientation=False, config=CONFIG).calculate_reference_angle(df)

        assert status.name == 'AUTOMATIC'
