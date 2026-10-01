# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Version numbers have the form MAJOR.MINOR.PATCH, but a minor version can contain breaking changes.
Each breaking change is marked **Breaking**.

## [2.4.0] - 2026-10-01

### Added
- **Walking is split into three paces in both presets**, by its step rate: `slow-walk` below 100 steps a minute (new class), `walk` from 100 to below 115, `fast-walk` from 115. `slow-walk` is walking below 4 km/h, the Compendium's edge between light and moderate. **`DEFAULT` judges each second on its own** (after a median of three seconds). On three walking-speed datasets (128 people, 117 of them scored) it scores 0.8365 macro F1 over the three paces and 0.8732 on light against moderate, where 2.3.3 scored 0.4062. **`LEGACY` judges each walking second by the mean step rate of the walking seconds in the minute around it**, the closest match to its 2.3.3 rule (two classes split at 100 steps a minute over 60 s blocks). The thigh config key is `'pace': {'slow': 100, 'fast': 115, 'window': 1}` (`window` 60 in `LEGACY`).
- `slow-walk` is in the fused map (to `walk`), both plots, every exposure that counts walking, and the trunk's reference angle.
- **The intensity bands are configurable.** `settings.INTENSITY` lists which activities count as `sedentary`, `lpa` and `mvpa`, and which count in no band (`none`: non-wear and stand). `Exposures(intensity=...)` takes a mapping of the same shape, for example to count standing as sedentary. Every activity must be in exactly one of the four bands, so they add up to the whole recording; a mapping that leaves one out, lists one twice or names an unknown one is refused with a `ValueError`. The default gives the same exposures as before.

### Changed
- **Breaking: the activity codes are renumbered** from still to active: 0 non-wear, 1 lie, 2 sit, 3 kneel, 4 squat, 5 stand, 6 shuffle, 7 slow-walk, 8 walk, 9 fast-walk, 10 run, 11 stairs, 12 bicycle, 13 row. A code stored by 2.3.3 or earlier means a different class here, and the SENS export uses the new codes.
- **Breaking: `walk` is now moderate, so `Exposures` counts it in `mvpa`** (it was in `lpa`), and `slow-walk` in `lpa`. **Output from 2.3.3 or earlier uses `walk` for light walking, and `Exposures` now counts it as `mvpa` without an error.** Compute exposures for such output with 2.3.3.
- **Breaking: no preset turns walking into running by its step rate** (`run` has no `steps` key). With a correct step rate that rule only turned fast child walking into running; `run_threshold` already catches every run.
- **Breaking: the 2.3.3 `fast-walk` rule is removed**, with `Thigh.get_steps` and the internal `steps` column. A thigh config without a `pace` entry is refused with a `ValueError` that says so.
- The `Exposures` activity columns are in the code order, from still to active, in place of an order set by chance.
- The daily `valid` flag counts walking at any pace (`slow-walk` + `walk` + `fast-walk` >= 5 min).
- **Breaking:** `walk_feature` and `run_feature` are now step rates in Hz. Before, both were FFT bin numbers (0-239), and `Thigh.get_steps` changed them to Hz. The column names are the same, but the values are not. **Features stored by 2.3.3 or earlier must not be classified by this version, nor the reverse:** a bin number read as Hz, or Hz read as a bin number, gives wrong walking classes without an error.
- `walk_feature` is now read from the thigh's angle from vertical (new `actimotus.cadence` module) in place of a 1.5-2.5 Hz band-pass on the long axis. The band-pass was right only between 90 and 150 steps a minute: below that it read about 1.5 times the true rate, above it about half. Against counted steps (84 treadmill stages, 21 adults) the new feature puts 98.7% of seconds within 10%.
- `run_feature` is now read the same way as `walk_feature`, from the thigh's angle from vertical, but searched only from 120 to 264 steps a minute (`cadence.RUN_STRIDE_BAND`). The shipped band-pass saw only about 130-206 steps a minute. Against the back sensor on child running, the new feature halves 0.2% of seconds and is about 6% low above 200 steps a minute, where the band-pass was about 9% low. On adults the two agree.
- Step features are transformed in blocks: a 24 h recording at 30 Hz now needs about 145 MB in place of about 980 MB.

### Removed
- **Breaking:** the `steps` column is no longer in the thigh output of `get_activities`, and it is no longer computed. In the SENS export, the `steps` slot stays in place and is always 0, so the columns after it do not move.

### Fixed
- **Walking no longer reads as `stairs` on a recording with little brisk walking.** The line between `walk` and `stairs` is the median `direction` of a pool of walking seconds, plus `stairs_threshold`. That pool took only seconds with `sd_x` above 0.25 and had no upright test, so slow walking was left out and lying with leg movement was let in. On a recording with little brisk walking the median then fell, and walking read as `stairs`. The pool is now the seconds `get_walk` and `get_stairs` judge: `sd_x` above the preset's `movement_threshold`, below `run_threshold`, and `inclination` below the preset's `inclination_angle`. On three walking-speed datasets (128 people, no stairs), walking read as `stairs` falls from 22.6% to 0.0% on one of them and is unchanged on the other two. On HARTH (127 people, free-living, with stair labels) walking read as walking rises from 88.3% to 89.3%, and stairs read as `stairs` from 70.6% to 72.1% (`DEFAULT`).
- **A recording with no walking to measure no longer reads its walking as `stairs`.** On an empty pool the stairs threshold fell back to `stairs_threshold` alone (5 deg in `DEFAULT`, 4 in `LEGACY`), far below walking, and on one older adult in HARTH 380 of 489 walking seconds read as `stairs`. It now falls back to `stairs_threshold` + 10 deg, the median walking direction measured on 255 people (10.5 deg), and it does so on any pool under 10 s: from 10 s up, a person's own median is closer to their full-recording value than the fallback is.
- The default chunk `size` of `Activities` and `DataFrameIterator` is `'1D'` in place of `'1d'`, which pandas 3 deprecates. The same 24 hours.
- Every step rate read about 6% low: the FFT bin was scaled as 0.0588 Hz where it is 0.0625 Hz. Both features are now computed by the new estimator, which has no such scale.

## [2.3.3] - 2026-08-15

### Fixed
- Orientation correction no longer corrupts `sd_x`/`sd_z` for a sensor detected as both upside down and inside out. That branch negates both `x` and `z`, which leaves the cross term `sum_dot_xz` unchanged, but it was negated anyway — and `Thigh._rotate_sd` uses it to rebuild the axis standard deviations, so the wrong sign reached every activity gate. **Only affects `orientation=True` with both flips detected**; the default `orientation=False` is unaffected.

### Changed
- Thigh rotational crossings are now computed as directed up/down crossings. The previous form took `.diff()` of a boolean and of its own negation, which in pandas is XOR and so produced two identical series. `get_lie` combines the two directions with **or**, so rolling the thigh past the orientation threshold in either direction still converts a sitting bout to lying, exactly as before. **No behaviour change:** per-second activity labels should be identical to 2.3.2.

## [2.3.2] - 2026-07-07

### Added
- Diary context mapping: `Exposures.context(df, diary)` annotates the 1-second activity series with boolean `context__<name>` columns derived from a diary of `[start, end, context, activities]` intervals. Supports overlapping contexts, per-interval activity gating, and multiple intervals per context (unioned into one column).

### Changed
- The daily `valid` flag is now `walk >= 5 min` (was `walk + stairs >= 10 min`). Stairs on a thigh sensor is a mounting/reference-angle-sensitive split of walking, so the walk+stairs sum can stay above threshold on a day where genuine walking was suppressed by an orientation artifact (the movement leaks into false `stairs`). Walk-only is a stricter, harder-to-fool data-quality floor. **Behaviour change:** windows previously marked valid on stairs alone are now invalid.
- `Exposures.window` is now typed `str` (was `str | timedelta`) and defaults to `'1D'` (was `'1d'`); use uppercase pandas offset aliases (`'1D'`, `'7D'`) — lowercase `'d'` is deprecated in pandas 3.0.
- `Exposures.context` now takes a full diary — `context(df, diary)` — and returns a copy of the activity DataFrame with one `context__<name>` column per context, replacing the earlier experimental single-context `(df, intervals, context, activities)` signature.
- Diary validation is strict: `Exposures.context` raises clear `ValueError`s for `NaT` timestamps, null/non-string/empty context values, malformed or unknown-label `activities`, a missing `activity` column, timezone mismatches between the diary and the activity index, and pre-existing `context__` column collisions. Surrounding whitespace in context names is normalized.
- Datetime-to-integer conversions in `Activities` and `Features` are now resolution-agnostic, correct for non-nanosecond `datetime64` indices (e.g. `[ms]` parquet under pandas ≥ 2).

### Fixed
- Thigh `row` (rowing) is no longer emitted for an inverted device or feet-up lying. `get_row` had a lower inclination bound (`87.5°`) but no upper bound, so `inclination = arccos(x)` values from `~90°` up to `180°` (i.e. `x` negative, device upside-down) with any leg motion were misclassified as rowing — the only class with a lower but no upper inclination bound — and folded into MVPA. A configurable `inclination_upper` (default `110.0` in the shipped config) now caps it; excluded windows fall through to `sit`/`lie` (every class above `sit` is gated below `87.5°`, so no MVPA leak). Backwards compatible: the `get_row` parameter defaults to `180.0` (a no-op) when a caller does not supply it.
- `Exposures` daily/weekly windows now bucket on **local calendar days** across DST transitions (a fall-back day is one 25-hour window, spring-forward one 23-hour window, all labelled at local midnight). Previously the window string was coerced to a `timedelta` in `__post_init__`; under pandas ≥ 3.0 that resolves to a fixed 24-hour tick, which drifted daily boundaries off local midnight and duplicated the fall-back date. The string is now passed straight to `pd.Grouper` (a calendar `<Day>` offset on all supported pandas versions).
- Declared the missing `scipy` runtime dependency (imported by `features` and `classifications.thigh`). Fresh installs previously relied on `scipy` arriving transitively and failed to `import actimotus` without it.
- Sampling-frequency detection and SENS timestamp export were off by a factor of 10³–10⁶ when the datetime index used a non-nanosecond resolution; conversions now use `.as_unit('ms')` / `.dt.total_seconds()`.

## [2.3.1] - 2026-02-03

### Changed
- Trunk reference angle calculation: Updated the calculation logic to prevent errors when values fall outside the valid arccos domain. Inputs are now strictly clipped to the [-1, 1] range (radians) to ensure numerical stability.
- Refined activity mapping: Updated the fused activities mapping logic. `Standing`: No longer categorized as sedentary or LPA; it is now tracked as a standalone category. `Kneeling`: Now mapped as sedentary.

### Fixed
- Exposures plot: Improved handling of timeline data to ensure consistent rendering and scaling.
- Project maintenance: Cleaned and optimized `pyproject.toml` and `.gitignore`.

## [2.3.0] - 2026-01-03

### Added
- New activity type: fast-walking.
- Timeline visualization plot for Exposures.
- Option to fuse activity types into merged exposures.
- Initial data quality checks for Activities/Exposures (flags invalid data when combined duration of climbing + walking is less than 10 minutes in a specific window).
- Support for custom configuration of activity detection thresholds.
- Experimental: Context initialization for Exposures (diary handling).
- Initial project documentation.

### Changed
- Renamed package from `acti-motus` to `actimotus`.
- Updated Exposures generation logic to include fast-walking and other new categories.
- Improved feature extraction robustness regarding data gaps (handling missing data in raw accelerometer time series).
- Updated default configuration thresholds for activities.
- Adjusted orientation correction: Non-wear data is no longer flipped when flipping detection is enabled.
- Implemented new custom gravitational calibration algorithm.
- Updated docstrings.

### Removed
- Dependency: `scikit-digital-health` library.

## [2.2.0] - 2025-09-10

### Added
- Auto-calibration support using the Scikit Digital Health library.
- Parser for Sens binary files.
- Configuration option to set custom activity thresholds (for thigh and trunk sensors).

### Changed
- Unified terminology: consistently use "compute" instead of generate/extract/etc.
- Updated wear-time detection algorithm (ongoing debugging).
- Renamed activity "move" → "shuffle".
- Updated default activity thresholds based on recent validation studies (use LEGACY_CONFIG for Acti4 threshold compatibility).
- Improved flipping detection functions to handle edge cases more robustly by refining detection thresholds.

### Fixed
- Corrected bug where non-wear time was not counted properly in the Exposures report.
- Added default stairs threshold when no data-based threshold is available.
- Fixed return values for inside-out flip detection for trunk sensors.
- Corrected chunking procedure: only acceleration axes are propagated (removed unintended overlapping column).
- Improved inside-out flipping detection for thigh sensors, reducing false positives and increasing accuracy.

### Removed
- Default logger.
- Multithreaded processing.

[2.4.0]: https://github.com/actimotus/actimotus/releases/tag/v2.4.0
[2.3.3]: https://github.com/actimotus/actimotus/releases/tag/v2.3.3
[2.3.2]: https://github.com/actimotus/actimotus/releases/tag/v2.3.2
[2.3.1]: https://github.com/actimotus/actimotus/releases/tag/v2.3.1
[2.3.0]: https://github.com/actimotus/actimotus/releases/tag/v2.3.0
[2.2.0]: https://github.com/actimotus/actimotus/releases/tag/v2.2.0
