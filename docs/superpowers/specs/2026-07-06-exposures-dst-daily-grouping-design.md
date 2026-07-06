# Design: Fix `Exposures` daily/weekly grouping across DST (pandas 3.0 regression)

**Date:** 2026-07-06 · **Scope:** `src/actimotus/exposures.py` only · **Type:** correctness bugfix
**Brief:** `docs/superpowers/dst-daily-grouping-bug.md`

## Problem

`Exposures(window='1d')` does not bucket on local calendar days across a DST transition **on
pandas ≥ 3.0**. `__post_init__` coerces the window string (`'1d'`) into a `timedelta(days=1)`
before handing it to `pd.Grouper`. pandas resolves a frequency **string** to a calendar `<Day>`
offset (DST-aware) but resolves a **timedelta** differently depending on version:

| pandas | `to_offset(timedelta(days=1))` | DST behaviour of the coerced grouper |
|---|---|---|
| 2.2.3, 2.3.3 | `<Day>` (calendar) | correct — fall-back day = one 25 h bin |
| 3.0.0, 3.0.3 | `<24 * Hours>` (fixed tick) | **drifts** — two Oct-26 rows, later rows at 23:00 |

So the coercion is a latent no-op on 2.2–2.3 but an active bug on 3.0+. Symptoms on 3.0.x:

- daily boundaries drift off local midnight (23:00 after fall-back, 01:00 after spring-forward);
- the DST fall-back date gets **two rows** (both labelled the same calendar date), and the true
  25-hour day is never one window.

Weekly (`window='7d'`) is affected identically. The failure is silent.

## Evidence

- **pandas offset resolution.** `pandas.tseries.frequencies.to_offset(pd.Timedelta("1d").
  to_pytimedelta())` → `<Day>` on 2.2.3/2.3.3, `<24 * Hours>` on 3.0.0/3.0.3. The string `'1d'`
  → `<Day>` on all four.
- **`Exposures` under pandas 3.0.3, real `veronika-phd` subject 21** (spans the fall-back):
  `2025-10-26` is duplicated (00:00+02:00 and 23:00+01:00); 4 off-midnight labels. Under 2.3.3
  the same call is correct (Oct-26 = 25 h, no duplicate). The consumer (`veronika-phd`) runs
  pandas 3.0.3, so its daily exposures are affected in production.
- `self.window` is used **only** as a `Grouper` freq (`exposures.py:139, :182`) — no arithmetic
  on the window length anywhere (grep-confirmed), so removing the coercion is length-agnostic.

## Fix (`src/actimotus/exposures.py`)

1. **Delete `__post_init__`** (lines 55–57). The string then flows untouched into
   `pd.Grouper(freq=self.window, sort=True)`, producing a calendar `<Day>` / `<7*Day>` offset on
   **all** supported pandas versions (verified 2.3.3 and 3.0.3).
2. **Narrow the type** (line 52): `window: str | timedelta = '1d'` → `window: str = '1D'`.
   - Passthrough is the whole fix. Narrowing to `str` makes the contract match reality (no caller
     passes a `timedelta`; a `timedelta` would silently reintroduce the fixed-tick bug on 3.0).
   - Default `'1d'` → `'1D'`: lowercase `'d'` is deprecated on pandas 3.0 and emits a
     `Pandas4Warning` when passed to `Grouper`. `'1D'`/`'7D'` (uppercase) are **warning-free on
     both 2.3.3 and 3.0.3** and future-proof for pandas 4.0. Keep `from datetime import
     timedelta` — still used at lines 223/230 in `_get_plot`.
3. **Update the `window` attribute docstring**: it takes a pandas offset string; use uppercase
   day/week aliases (`'1D'`, `'7D'`); these bucket on **local calendar days**, so across DST a
   fall-back day is a genuine **25 h** window and a spring-forward day a **23 h** window — correct,
   not a defect. Update the `'7d'` docstring example to `'7D'`.

## What does not change (verified)

- `_get_exposure` counts rows as seconds (`pd.Timedelta(count, unit='s')`) and the `valid` flag is
  an absolute `≥ 10 min` threshold — no "per-24 h" denominator, so variable-length DST days flow
  through correctly.
- `quality_check` (line 320) calls `.index.normalize()`; with calendar grouping the labels are
  already at local midnight, so it is a no-op.
- Scope stays on `Exposures`. The other `to_pytimedelta()` sites (calibration, step bouts,
  memory-chunk boundaries, TTLs, signal sampling) are output-invariant or genuine fixed durations
  — see the brief's audit. Do **not** touch the sub-minute ones.

## Tests (TDD — new file `tests/test_exposures_dst.py`)

**Version caveat:** the bug only manifests on pandas ≥ 3.0. The DST regression tests therefore
**pass trivially on 2.3.3 and only exercise the bug under pandas ≥ 3.0**. RED must be demonstrated
under 3.0.x (`uv run --with 'pandas==3.0.3' pytest tests/test_exposures_dst.py`). GREEN is verified
under **both** 2.3.3 (project env) and 3.0.3.

Follow `conftest.py` conventions; timezone `Europe/Copenhagen` (same CET/CEST transitions as
Prague — fall-back 2025-10-26, spring-forward 2026-03-29). Frames use **1-second epochs** (the
library counts one row as one second, so a fully-populated calendar day yields a duration equal to
the day's true wall-clock length).

1. **Fall-back (2025-10-26).** Frame spanning the transition → exactly one row per calendar date
   (no duplicate 2025-10-26); Oct-26 activity duration totals **25 h**; every index label at local
   **00:00**.
2. **Spring-forward (2026-03-29).** → a single **23 h** day; labels at local midnight.
3. **Regression (non-DST week).** A plain week → all 24 h days; labels at midnight; no drift.
4. **Weekly across a transition (`window='7D'`).** → 7 whole calendar days per bin; no drift.
5. **No deprecation warning.** `Exposures().compute(...)` on any tz-aware frame emits no
   `Pandas4Warning`/`FutureWarning` from the default window (guards the `'1d'`→`'1D'` choice).

## Housekeeping

- Add a `CHANGELOG.md` entry under **Fixed** (bugfix → patch bump per the versioning convention;
  note it is a pandas-3.0 regression). Release `chore` commit follows the normal flow.
- **Recommend** raising the acti-motus test matrix to include pandas 3.0 — the bug slipped because
  CI/dev runs 2.3.x, where it is invisible. (Flagged to the maintainer; out of scope for the code
  fix itself.)

## Repo conventions

- Tests: `pytest`, in `tests/`, fixtures in `conftest.py`. Run with `uv run pytest`.
- Lint/format: `ruff`, line-length 120, Google-style docstrings, target py311.
