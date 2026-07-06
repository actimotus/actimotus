# Bug brief: `Exposures` daily/weekly grouping drifts across DST

**Filed:** 2026-07-06 · **Severity:** correctness (silent) · **Scope:** `Exposures` only (audited)
**Reporter context:** found while building 24-hour movement-behaviour compositions in the
`veronika-phd` project (accelerometer thigh data, Europe/Prague, recordings spanning the
Oct-2025 fall-back and Mar-2026 spring-forward).

> This is a self-contained handoff. Recommended workflow: **superpowers brainstorming →
> writing-plans → TDD implementation → requesting-code-review** (in a fresh thread).

---

## ✅ CONFIRMED (2026-07-06) — real bug, but pandas-version-dependent

The bug is **real on pandas ≥ 3.0** and **absent on pandas 2.2–2.3**. Full evidence and the fix
in `docs/superpowers/specs/2026-07-06-exposures-dst-daily-grouping-design.md`. Key points:

- **pandas 3.0 changed the offset resolution.** On 2.2.3/2.3.3, `to_offset(timedelta(days=1))`
  returns `<Day>` (calendar-aware), so the `__post_init__` coercion is harmless there. On
  **3.0.0/3.0.3 it returns `<24 * Hours>`** (a fixed tick) — so the coercion downgrades
  calendar-day grouping exactly as this brief describes.
- Confirmed through `Exposures` under **pandas 3.0.3** on real `veronika-phd` data (subject 21):
  `2025-10-26` appears **twice** (00:00+02:00 and 23:00+01:00), post-transition rows drift to
  23:00. Under 2.3.3 the same call is correct — which is why an initial pass on the 2.3.3 env
  wrongly concluded "not a bug." **The bug is invisible below pandas 3.0.**
- acti-motus requires `pandas>=2.2.3` (range includes 3.0) and the primary consumer runs 3.0.3,
  so the fix is needed. **Testing caveat: a DST regression test only fails under pandas ≥ 3.0.**

**Fix:** stop coercing the string; pass it to `Grouper` (→ `<Day>` on all versions); narrow
`window` to `str`; change the default `'1d'` → `'1D'` (lowercase `'d'` warns on pandas 3.0).

## TL;DR

`Exposures(window='1d')` does **not** bucket on local calendar days across a DST transition.
Its daily windows are fixed 24-hour ticks anchored in absolute (UTC) time, so once the local
UTC offset changes:

- the daily boundaries drift off local midnight (to 23:00 after fall-back, 01:00 after
  spring-forward), and
- the DST fall-back date gets **two rows** (both labelled the same calendar date), while the
  true 25-hour day is never represented as one 25-hour window.

pandas itself is **not** at fault — `pd.Grouper(freq='1d')` on a tz-aware index groups by
calendar day correctly (a fall-back day → one 25-hour bin). The bug is that `Exposures`
converts the window **string** into a fixed `timedelta` before handing it to the grouper,
which downgrades calendar-day semantics to fixed-24h-tick semantics.

## Root cause (one place)

`src/actimotus/exposures.py`, `Exposures.__init__`:

```python
if isinstance(self.window, str):
    self.window = pd.Timedelta(self.window).to_pytimedelta()   # '1d' -> timedelta(days=1)
```

Then the (now `timedelta`) window is passed to the grouper:

```python
# exposures.py:139 and :182
... pd.Grouper(freq=self.window, sort=True) ...
```

**Why that breaks it:** pandas resolves a frequency **string** differently from a **timedelta
object**:

| passed to `pd.Grouper(freq=...)` | `to_offset` result | DST behaviour |
|---|---|---|
| `'1d'` / `'1D'` / `'D'` (string)  | `<Day>` offset (calendar) | correct — fall-back day = one 25 h bin |
| `timedelta(days=1)` (object)      | `<24 * Hours>` **Tick** (fixed) | **drifts** — anchored to absolute UTC, two labels on the DST date |

Because `__init__` coerces the string to a `timedelta`, `Grouper` receives the object form and
uses the fixed-tick path.

## Reproduction (pandas-level, proves pandas is fine)

```python
import pandas as pd
idx = pd.date_range("2025-10-24", "2025-10-28", freq="1min", tz="Europe/Prague")
s = pd.Series(1, index=idx)                       # Oct-26 is the 25 h fall-back day

s.groupby(pd.Grouper(freq="1d")).size()           # Oct-26 -> ONE bin, 1500 min (25 h)  ✅
s.groupby(pd.Grouper(freq=pd.Timedelta("1d").to_pytimedelta())).size()
                                                  # Oct-26 -> TWO bins, 1440+1440 min   ❌ (the bug)
```

## Reproduction (through `Exposures`, the actual symptom)

Run `Exposures(window='1d', fused=False).compute(activities)` on a tz-aware
(`Europe/Prague`) 1 s activity frame that spans 2025-10-26. Observed:

```
datetime                     wear
2025-10-26 00:00:00+02:00    24 h     <- labelled Oct 26
2025-10-26 23:00:00+01:00    24 h     <- ALSO labelled Oct 26 (really the Oct-27 window)
2025-10-27 23:00:00+01:00    ...      <- every later day now anchored at 23:00 local
```

Expected: a single `2025-10-26 00:00+02:00` row of **25 h**, and all later rows at local
midnight.

## Impact

- Any consumer doing **per-calendar-day** analysis (day-typing against a shift roster,
  night-vs-day splits, matching to diaries) gets days shifted ±1 h for the entire
  post-transition portion of a recording.
- The DST date is duplicated (fall-back) and no window ever equals the true 25 h / 23 h day.
- In the source dataset that surfaced this: **26 of 74 subjects** had off-midnight day
  labels; 8 had duplicated fall-back dates. Silent — nothing errors.
- Weekly (`window='7d'`) is affected the same way.

## Codebase audit — is the pattern elsewhere? (done 2026-07-06)

The `pd.Timedelta(...).to_pytimedelta()` coercion of a frequency string appears in several
places. Only **one** is a correctness bug; the rest are benign, for concrete reasons:

| Site | Window | Verdict |
|---|---|---|
| `exposures.py:57` → grouper `:139,:182` | `'1d'` / `'7d'` (daily/weekly) | **BUG** — calendar semantics required |
| `calibration.py:26` → `resample(:31)` | `'10s'` (default) | benign — sub-minute; fixed-tick == calendar at this scale |
| `classifications/thigh.py:455` grouper | `'{fast-walk bout}s'` (seconds) | benign — sub-minute step-bout window |
| `iterators.py:13-14`, `features.py:70,73`, `activities.py:79,82` | chunk `size`/`overlap` (`'7d'`/`'1d'`) | benign — memory-chunk boundaries with **trimmed overlap**; output is boundary-invariant |
| `references.py:85` `ttl` | cache TTL | correct — a TTL is a genuine fixed duration |
| `features.py:479,526`, `activities.py:316` `as_unit('ms').astype(int64)` | signal processing | correct — sampling is absolute-time by definition |

**Conclusion:** scope the fix to `Exposures`. The brainstorm should still *confirm* the chunk
sites are output-invariant (they should be, thanks to overlap trimming) and decide whether to
normalise the coercion pattern for consistency — but they are not correctness bugs. Do **not**
"fix" the sub-minute windows (calibration, step bouts): fixed-tick is correct there.

## Fix direction (for the brainstorm to refine, not prescriptive)

Core idea: give the grouper a **calendar-day offset**, not a fixed tick.

- In `Exposures`, `self.window` is only ever used as a `Grouper` `freq` (lines 139, 182) — grep
  confirms no arithmetic use of the window length. So the simplest fix is to **stop coercing
  the string to a `timedelta`** and pass the string through to `Grouper` (which then yields a
  `<Day>`/`<7*Day>` offset).
- Contract to decide: the annotation is `window: str | timedelta`. If a caller passes a
  `timedelta` object, `Grouper` will still use fixed-tick semantics. Options: (a) document that
  the string form is required for calendar-day semantics; (b) map a whole-day timedelta to the
  equivalent offset alias; (c) warn. Pick one explicitly.
- **Downstream consequence to verify:** with calendar-day grouping, a DST fall-back day is a
  genuine **25 h** window and spring-forward a **23 h** window. Confirm no exposure metric uses
  the window *length* as a fixed 24 h denominator (durations and the `valid = walk+stairs ≥
  10 min` flag are unaffected; check `_get_exposure`/`_get_exposures` for any implicit "per 24 h"
  assumption). Variable-length DST days are the **correct** result and should be documented.

## Test plan (write the failing test first — TDD)

Add to `tests/` (pytest; there is a `conftest.py`). Suggested `test_exposures_dst.py`:

1. **Fall-back:** build a tz-aware `Europe/Prague` 1 s (or 1 min) activity frame spanning
   2025-10-26. Assert after `Exposures(window='1d').compute(...)`:
   - exactly **one** row per calendar date (no duplicate 2025-10-26),
   - the 2025-10-26 window's total duration = **25 h**,
   - every bin label is at local **midnight** (`.dt.strftime('%H:%M') == '00:00'`).
2. **Spring-forward:** same over 2026-03-29 → single **23 h** day, labels at midnight.
3. **Regression (non-DST):** a plain week with no transition is unchanged (all 24 h days).
4. Optional: `window='7d'` spanning a transition → 7 whole calendar days, no drift.

All of (1)–(2) fail on current `main` and pass after the fix.

## Repo conventions

- Tests: `pytest` (config in `pyproject.toml`), tests live in `tests/`, fixtures in
  `conftest.py`. Run with `uv run pytest`.
- Lint/format: `ruff`, line-length **120**, Google-style docstrings, target **py311**.
- Working tree was **clean** as of this brief (no uncommitted changes to preserve).

## Downstream note (`veronika-phd`)

`veronika-phd` currently **works around** this by building its daily composition table from a
Prague-calendar-date `groupby` of `activities.parquet` (validated to yield a correct 25 h
Oct-26). Once this is fixed upstream, that project can optionally rely on `Exposures` daily
output again — but its own groupby is also fine to keep. No coordination required; the fix is
purely an improvement.
