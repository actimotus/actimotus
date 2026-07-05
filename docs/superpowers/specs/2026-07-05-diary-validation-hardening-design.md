# Diary Validation Hardening — Design

**Date:** 2026-07-05
**Status:** Approved (brainstorm), pending implementation plan
**Builds on:** `docs/superpowers/specs/2026-07-05-diary-context-mapping-design.md`

## Purpose

The diary context-mapping feature assumes a clean diary supplied by the caller.
Today's `_validate_diary` catches only missing columns, non-tz-aware timestamps,
and `end <= start`. Several malformed inputs slip through and cause **silent wrong
results or opaque crashes** instead of a clear error at the boundary:

- **Null `context`** → `groupby('context')` silently drops the row; the column
  vanishes with no error.
- **`NaT` in `start`/`end`** → passes the dtype and `end > start` checks, then
  silently matches *nothing* during masking (comparisons with `NaT` are always
  `False`).
- **`pd.NA` / non-list `activities`** → crashes deep inside `_context_mask` with a
  `TypeError` on `len(...)`.
- **Non-string / empty `context`** → produces a nonsense column like `context__`
  or `context__123`.
- **Typo'd activity label** (e.g. `'sleeping'`) → silently matches nothing.

Cleaning the diary remains the caller's responsibility (out of scope). This work
makes `_validate_diary`/`context` the **boundary guard** that converts every such
problem into a clear `ValueError`.

## Principle

**Strict: raise on everything.** Every detected problem is a `ValueError` with a
specific, actionable message. Consistent with the existing strict timezone rule.
No warnings. The **one** normalization applied is stripping surrounding whitespace
from `context` names (see `context()` below) — a padded name is a formatting
nuisance, not an error, and stripping also lets `' work '` and `'work'` collapse
into a single column instead of splitting.

## Architecture

Keep the existing diary/df split:

- **`Exposures._validate_diary(diary)`** — diary-only structural checks (needs no
  `df`). Gains the new NaT / context / activities checks, including validating
  activity labels against `settings.ACTIVITIES` (already imported in
  `exposures.py`).
- **`Exposures.context(df, diary)`** — the checks that relate the diary to the
  `df` (activity column present, timezone match on both `start` and `end`, no
  column collision), added before the annotation loop.
- **`Exposures._context_mask`** — its "no gate" detection is generalized from
  `float`-NaN-only to **any scalar NA** (`pd.api.types.is_scalar(x) and
  pd.isna(x)`), matching what `_validate_diary` now accepts. Without this, a
  validation-approved `pd.NA` activities cell would reach `len(pd.NA)` and raise
  `TypeError`, breaking the ValueError-only contract. Its defensive
  `None`/`NaN`/`pd.NA`/empty-list handling keeps it robust and independently
  testable.

Rejected alternatives: passing `df` into `_validate_diary` (muddies the "validate
the diary alone" boundary, makes it un-callable without a df); a separate
`_validate_context_call` function (unneeded surface for a small feature).

## Checks

### `_validate_diary(diary)` — diary-only

In addition to the existing checks (required columns `start`/`end`/`context`;
`start`/`end` are `pd.DatetimeTZDtype`; every `end > start`), raise `ValueError`
on:

1. **`NaT` in `start` or `end`** — any missing timestamp. Message names the column.
2. **`context` invalid** — a value that is null (`NaN`/`None`), not a `str`, or
   empty/whitespace-only after `str.strip()`.
3. **`activities` malformed** — for each row, the cell must be one of:
   - **"no gate":** `None`, float `NaN`, or `pd.NA`, **or** an empty list `[]`.
   - **a gate:** a non-empty `list` whose every element is a `str` present in the
     `ACTIVITIES` values (the 13 canonical labels: `non-wear, lie, sit, stand,
     shuffle, walk, run, stairs, bicycle, row, kneel, squat, fast-walk`).

   Anything else raises: `pd.NA` is fine (no gate), but a bare string, a number, a
   tuple/set, a list containing a non-string, or a list containing an unknown
   label all raise. The message names the offending value/label.

Note on missing detection: a cell is "missing" (no gate) when it is not a list and
`pd.isna(cell)` is `True` (covers `None`, `NaN`, `pd.NA`). Lists are checked as
gates. This ordering avoids calling `pd.isna` on a list (which returns an array).

### `context(df, diary)` — diary↔df relationship

Before `df.copy()` and the loop, in addition to the existing index-tz-aware and
`start`-zone-match checks, raise `ValueError` on:

4. **`df` has no `activity` column** — required for both masking and the gate.
5. **`end` timezone mismatch** — extend the existing `start`-vs-index zone check so
   both `start` and `end` zones must equal the index zone. (String comparison of
   tz as today.)
6. **Column collision** — `df` already contains a column named `context__<name>`
   for some (normalized) context in the diary. Prevents silently overwriting
   caller data.

**Normalization (not a check):** after validation, strip surrounding whitespace
from the context values before grouping — e.g. `diary = diary.copy();
diary['context'] = diary['context'].str.strip()`. This is done on the local copy
only (the caller's frame is never mutated). Because grouping and the
`context__<name>` column names then use the stripped values, `' work '` yields
`context__work`, and `' work '` + `'work'` collapse into one column. The collision
check (6) runs against these normalized names.

## Behavior notes & edge cases

- **Empty list `activities` (`[]`)** stays valid = no gate, matching the existing
  `_context_mask` behavior and its tests.
- **Empty diary** (zero rows, correct dtypes) still passes — every per-row check
  is vacuous, and `context()` returns the copy unchanged. Preserve this.
- **`_context_mask` NA handling is generalized** from `isinstance(activities,
  float) and pd.isna(...)` to `pd.api.types.is_scalar(activities) and
  pd.isna(activities)`, so `pd.NA` (which validation accepts as "no gate") is
  handled by the helper too rather than crashing on `len(pd.NA)`.
- Messages should be specific enough to fix the input (name the column, the row's
  value, or the unknown label).

## Testing

New tests extend `tests/test_exposures_context.py`.

**`TestValidateDiary`:**
- `NaT` in `start` raises; `NaT` in `end` raises.
- null `context` raises; non-string `context` (e.g. `123`) raises; empty/whitespace
  `context` raises.
- `activities` as `pd.NA` → passes (no gate); as a bare string → raises; as a
  number → raises; as a list with a non-string element → raises; as a list with an
  unknown label (`'sleeping'`) → raises; as a list of valid labels → passes.
- Existing valid-diary and `[]`-as-no-gate cases still pass.

**`TestContext`:**
- `df` without an `activity` column raises.
- diary whose `end` is in a different tz than the index raises.
- `df` already containing a `context__work` column raises when the diary has a
  `work` context.
- **whitespace normalization:** a diary context `' work '` produces a
  `context__work` column (not `context__ work `); a diary mixing `' work '` and
  `'work'` yields a single `context__work` column (their intervals unioned).
- Existing behavior tests (columns added, overlap, union, copy, empty-diary,
  tz-mismatch, naive-index) still pass.

## Open items for the implementation plan

None — the check-set and placement are fully specified.
