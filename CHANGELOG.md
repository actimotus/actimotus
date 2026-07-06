# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `Exposures` daily/weekly windows now bucket on **local calendar days** across DST transitions (a fall-back day is one 25-hour window, spring-forward one 23-hour window, all labelled at local midnight). Previously the window string was coerced to a `timedelta` in `__post_init__`; under pandas ≥ 3.0 that resolves to a fixed 24-hour tick, which drifted daily boundaries off local midnight and duplicated the fall-back date. The string is now passed straight to `pd.Grouper` (a calendar `<Day>` offset on all supported pandas versions).
- Declared the missing `scipy` runtime dependency (imported by `features` and `classifications.thigh`). Fresh installs previously relied on `scipy` arriving transitively and failed to `import actimotus` without it.

### Changed
- `Exposures.window` is now typed `str` (was `str | timedelta`) and defaults to `'1D'` (was `'1d'`); use uppercase pandas offset aliases (`'1D'`, `'7D'`) — lowercase `'d'` is deprecated in pandas 3.0.

## [2.3.2] - 2026-07-06

### Added
- Diary context mapping: `Exposures.context(df, diary)` annotates the 1-second activity series with boolean `context__<name>` columns derived from a diary of `[start, end, context, activities]` intervals. Supports overlapping contexts, per-interval activity gating, and multiple intervals per context (unioned into one column).

### Changed
- `Exposures.context` now takes a full diary — `context(df, diary)` — and returns a copy of the activity DataFrame with one `context__<name>` column per context, replacing the earlier experimental single-context `(df, intervals, context, activities)` signature.
- Diary validation is strict: `Exposures.context` raises clear `ValueError`s for `NaT` timestamps, null/non-string/empty context values, malformed or unknown-label `activities`, a missing `activity` column, timezone mismatches between the diary and the activity index, and pre-existing `context__` column collisions. Surrounding whitespace in context names is normalized.
- Datetime-to-integer conversions in `Activities` and `Features` are now resolution-agnostic, correct for non-nanosecond `datetime64` indices (e.g. `[ms]` parquet under pandas ≥ 2).

### Fixed
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

[Unreleased]: https://github.com/acti-motus/acti-motus/compare/v2.3.0...HEAD
[2.3.0]: https://github.com/acti-motus/acti-motus/releases/tag/v2.3.0
[2.2.0]: https://github.com/acti-motus/acti-motus/releases/tag/v2.2.0
