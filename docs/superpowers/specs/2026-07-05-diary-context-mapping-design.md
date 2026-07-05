# Diary Context Mapping — Design

**Date:** 2026-07-05
**Status:** Approved (brainstorm), pending implementation plan
**Related:** GitHub issue [#6](https://github.com/actimotus/actimotus/issues/6), prior attempt in `dev/diaries.ipynb`

## Purpose

Let users annotate the 1-second activity time series with **contexts** (domains)
drawn from a diary: self-reported time intervals such as `sleep`, `work`,
`commute`, or a whole-day `work-day`. Each context becomes a boolean column on
the activity DataFrame, so downstream analysis (e.g. `Exposures.compute`) can be
sliced by context — "how much MVPA during work?", "what activity happened during
the reported sleep window?".

Contexts are **not mutually exclusive**: a single epoch can belong to several at
once (e.g. `commute` ∧ `work-day`). The model must handle arbitrary overlap.

## Scope

**In scope:** validating a clean diary, and applying it to the activity
DataFrame to produce boolean context columns.

**Out of scope:** parsing the messy app-export CSV (`temp/export_diary_*.csv`
with `entry_type`/`HH:MM`/midnight-crossing). Converting that export into the
clean diary shape is a **separate upstream adapter** (a script or helper), not
part of this feature. This keeps the core matching logic clean and vendor-neutral.

**Explicitly not included (YAGNI):** relabeling / overriding the `activity`
column with a context value; a `dimension`/grouping layer over contexts;
context-stratified exposure summaries as a built-in. All of these remain trivial
one-liners downstream because the boolean columns sit next to the untouched
`activity` column.

## Data model

### Diary

A "flat named intervals" table — a `pd.DataFrame` with columns:

| column       | type                         | notes |
|--------------|------------------------------|-------|
| `start`      | tz-aware datetime            | interval start (inclusive) |
| `end`        | tz-aware datetime            | interval end (exclusive) |
| `context`    | str                          | context/domain name, e.g. `sleep`, `work` |
| `activities` | `list[str]` \| `None`        | **optional per-row** activity gate |

Example:

```
start                      end                        context   activities
2024-09-02 23:00:00+02:00  2024-09-03 06:30:00+02:00  sleep     ['lie', 'sit']
2024-09-02 08:00:00+02:00  2024-09-02 16:00:00+02:00  work      None
2024-09-02 07:15:00+02:00  2024-09-02 08:00:00+02:00  commute   None
2024-09-02 00:00:00+02:00  2024-09-03 00:00:00+02:00  work-day  None
```

- Overlap is allowed and expected (`commute` sits inside `work-day`).
- A whole-day context (`day_type` in issue #6) is just a context whose interval
  spans the day — no special-casing.
- `activities` is a property of the **row** (each interval may differ), not of
  the context. Consistency across rows of the same context is by convention only.
- **A context may appear in multiple rows** (e.g. two `sleep` periods in one day).
  These do not create multiple columns — they collapse into the single
  `context__sleep` column, `True` for any epoch inside *either* interval (union).

### Activity DataFrame (input, unchanged)

The output of `Activities.compute`: tz-aware `DatetimeIndex` named `datetime` at
1-second epochs, with an `activity` string column (plus steps/inclination columns).

### Output

A **copy** of the input activity DataFrame with one added boolean column per
distinct context name, prefixed `context__`:

```
datetime             activity  ...  context__sleep  context__work  context__commute  context__work-day
2024-09-02 07:30:00  walk           False           False          True              True
2024-09-02 08:15:00  sit            False           True           False             True
2024-09-02 23:30:00  lie            True            False          False             False
```

The `context__` prefix prevents collisions and makes the columns selectable via
`df.filter(like='context__')`. Overlap is naturally represented — an epoch can be
`True` in several `context__` columns.

## Components

### `validate_diary(diary) -> None` (raises on invalid)

Standalone, testable, and invoked automatically at the top of `context()`.
Checks (raises `ValueError` on failure):

1. Required columns present: `start`, `end`, `context`. `activities` optional; if
   absent, treated as all-`None`.
2. `start < end` for every row.
3. `start` and `end` are **tz-aware** (raise if naive).

The **same-zone** check is done in `context()` (which has the activity index):
diary zone must equal the activity index zone, else raise. Rule is strict — no
warnings, no localization, no cross-zone tolerance. Both tz-aware and identical
zone → proceed; anything else → error.

### `Exposures.context(df, diary) -> pd.DataFrame`  (public)

Replaces the current single-context `Exposures.context` (breaking change to a
documented method — acceptable).

1. Copy `df` (never mutate the caller's frame).
2. `validate_diary(diary)` (structure + tz-aware), then assert diary zone ==
   `df.index` zone, else raise.
3. Group `diary` by `context`.
4. For each context, call `_context_mask` over that context's rows and assign the
   result to `context__<name>`. Multiple rows for one context union into one column.
5. Return the enriched copy.

### `Exposures._context_mask(df, intervals) -> pd.Series`  (private)

Single-context matcher (generalizes today's `context` internals). `intervals` is
the diary sub-frame for one context (`start`, `end`, `activities`). Returns a
boolean `pd.Series` aligned to `df.index`:

- For each interval row: epoch is `True` when its timestamp is in `[start, end)`
  **and**, if that row's `activities` is a non-empty list, its `activity` is in
  that list. `None`/empty `activities` → pure interval membership.
- Union across the context's rows.

This is the existing loop, extended so the activity gate is read **per interval
row** instead of a single list for the whole call.

## Data flow

```
export_diary_*.csv  --(separate upstream adapter, out of scope)-->  clean diary [start,end,context,activities]
                                                                          |
Activities.compute(...) --> activity df (1s epochs, datetime index) ------+
                                                                          v
                                        Exposures.context(df, diary)  (copy + validate + loop)
                                                                          v
                                        df + context__<name> boolean columns
                                                                          v
                            downstream: Exposures.compute slices, activity∧context one-liners
```

## Behavior notes & edge cases

- **Interval matching** is half-open `[start, end)`, consistent with existing code.
- **Activity-gated context** (e.g. sleep as interval ∧ lie/sit) is expressed via
  the diary's `activities` column; it is **not** lossy relabeling. The `activity`
  column is untouched, so any activity-conditioned question remains recoverable
  downstream.
- **Empty diary** → returns a copy of `df` with no context columns added.
- **Context name collisions** with existing columns are avoided by the
  `context__` prefix.
- **Thigh-only sensor limitation** (kneel/squat undetectable) is a diary-authoring
  concern, not a code concern here.

## Testing

- `validate_diary`: missing columns; `start >= end`; naive timestamps raise.
- `context` tz: same-zone proceeds; different zone raises; diary-vs-index zone
  mismatch raises.
- `_context_mask`: single interval; **multiple intervals same context union into
  one column**; per-row `activities` gate vs `None`; overlapping intervals;
  half-open boundary (epoch exactly at `end` excluded).
- `context`: multi-context diary → correct `context__*` columns; overlap
  reflected in multiple columns; input frame **not mutated** (copy semantics);
  empty diary.

## Open items for the implementation plan

- Where the upstream `export_diary_*.csv` adapter should live (likely `dev/` or a
  small documented helper) — tracked separately from this feature.
