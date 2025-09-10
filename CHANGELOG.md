# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/acti-motus/acti-motus/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/acti-motus/acti-motus/releases/tag/v2.2.0
