# Diary Context Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Annotate the 1-second activity DataFrame with boolean `context__<name>` columns derived from a clean diary of time intervals.

**Architecture:** Three static methods on `Exposures`: `_validate_diary` (structure + tz-aware checks), `_context_mask` (private; boolean Series for one context, per-row activity gate, half-open matching, union across the context's rows), and `context` (public; copies the frame, validates, asserts the diary zone matches the activity-index zone, loops the distinct contexts, and adds one `context__<name>` column each). Replaces the old single-context `Exposures.context`.

**Tech Stack:** Python, pandas, pytest. Run tests with `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-07-05-diary-context-mapping-design.md`

---

## File Structure

- Modify: `src/actimotus/exposures.py` — replace the existing `context` staticmethod (lines 325-340) with `__validate_diary`, `_context_mask`, and the new `context`.
- Create: `tests/__init__.py` — empty; makes `tests/` a package.
- Create: `tests/conftest.py` — shared fixtures (`activities` frame, `make_diary` helper).
- Create: `tests/test_exposures_context.py` — all tests for the three methods.
- Modify: `docs/references/exposures.md` — add `_validate_diary` to the documented members.

There is no `tests/` directory yet; `pyproject.toml` already sets `testpaths = ["tests"]`, `python_files = ["test_*.py"]`. This plan creates the first tests.

---

## Task 1: Test scaffolding and fixtures

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create the tests package marker**

Create `tests/__init__.py` as an empty file (0 bytes).

- [ ] **Step 2: Create shared fixtures**

Create `tests/conftest.py`:

```python
import pandas as pd
import pytest

TZ = 'Europe/Copenhagen'


@pytest.fixture
def activities():
    """Ten 1-second epochs, 2024-09-02 07:00:00..07:00:09 (Europe/Copenhagen)."""
    index = pd.date_range(
        '2024-09-02 07:00:00', periods=10, freq='1s', tz=TZ, name='datetime'
    )
    labels = ['walk', 'walk', 'sit', 'sit', 'lie', 'lie', 'stand', 'walk', 'sit', 'lie']
    return pd.DataFrame({'activity': labels}, index=index)


def make_diary(rows, tz=TZ):
    """Build a clean diary from (start, end, context, activities) tuples.

    `start`/`end` are naive strings localized to `tz` (pass tz=None to keep naive).
    """
    diary = pd.DataFrame(rows, columns=['start', 'end', 'context', 'activities'])
    diary['start'] = pd.to_datetime(diary['start'])
    diary['end'] = pd.to_datetime(diary['end'])
    if tz is not None:
        diary['start'] = diary['start'].dt.tz_localize(tz)
        diary['end'] = diary['end'].dt.tz_localize(tz)
    return diary


@pytest.fixture
def diary_factory():
    return make_diary
```

- [ ] **Step 3: Verify pytest collects the empty suite**

Run: `uv run pytest tests/ -q`
Expected: `no tests ran` (exit code 5) — confirms collection works and fixtures import cleanly with no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add tests package and diary/activities fixtures"
```

---

## Task 2: `_validate_diary`

**Files:**
- Modify: `src/actimotus/exposures.py` (add `_validate_diary` staticmethod inside `class Exposures`)
- Test: `tests/test_exposures_context.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_exposures_context.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_exposures_context.py::TestValidateDiary -q`
Expected: FAIL — `AttributeError: type object 'Exposures' has no attribute '_validate_diary'`.

- [ ] **Step 3: Implement `_validate_diary`**

In `src/actimotus/exposures.py`, inside `class Exposures`, replace the existing `context` staticmethod (currently lines 325-340) with the following three methods. This step adds `_validate_diary`; Steps in Tasks 3 and 4 add the other two (place all three together where the old `context` was):

```python
    @staticmethod
    def _validate_diary(diary: pd.DataFrame) -> None:
        """Validate a diary of context intervals.

        The diary must have columns ``start``, ``end``, ``context`` (and an
        optional ``activities`` column). ``start``/``end`` must be timezone-aware
        datetimes and every ``end`` must be strictly after its ``start``.

        Args:
            diary: DataFrame with columns ``start``, ``end``, ``context`` and an
                optional per-row ``activities`` list.

        Raises:
            ValueError: If required columns are missing, any interval has
                ``end <= start``, or ``start``/``end`` are not timezone-aware.
        """
        required = {'start', 'end', 'context'}
        missing = required - set(diary.columns)
        if missing:
            raise ValueError(f'Diary missing required columns: {sorted(missing)}')

        for column in ('start', 'end'):
            if not isinstance(diary[column].dtype, pd.DatetimeTZDtype):
                raise ValueError(
                    f"Diary column '{column}' must be timezone-aware datetimes."
                )

        if (diary['end'] <= diary['start']).any():
            raise ValueError("Diary has rows where 'end' is not after 'start'.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_exposures_context.py::TestValidateDiary -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/actimotus/exposures.py tests/test_exposures_context.py
git commit -m "feat: add Exposures._validate_diary"
```

---

## Task 3: `_context_mask` (private single-context matcher)

**Files:**
- Modify: `src/actimotus/exposures.py` (add `_context_mask` staticmethod after `_validate_diary`)
- Test: `tests/test_exposures_context.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_exposures_context.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_exposures_context.py::TestContextMask -q`
Expected: FAIL — `AttributeError: type object 'Exposures' has no attribute '_context_mask'`.

- [ ] **Step 3: Implement `_context_mask`**

In `src/actimotus/exposures.py`, add this staticmethod directly after `_validate_diary`:

```python
    @staticmethod
    def _context_mask(df: pd.DataFrame, intervals: pd.DataFrame) -> pd.Series:
        """Boolean mask for one context across its (possibly multiple) intervals.

        An epoch is ``True`` when its timestamp falls inside any interval's
        half-open ``[start, end)`` window and, if that interval row carries a
        non-empty ``activities`` list, the epoch's ``activity`` is in it. Rows for
        the same context union together.

        Args:
            df: Activity DataFrame, timezone-aware DatetimeIndex, ``activity`` column.
            intervals: The diary rows for a single context (``start``, ``end`` and
                optional per-row ``activities``).

        Returns:
            Boolean ``pd.Series`` aligned to ``df.index``.
        """
        mask = pd.Series(False, index=df.index)
        has_activities = 'activities' in intervals.columns

        for row in intervals.itertuples(index=False):
            in_interval = (df.index >= row.start) & (df.index < row.end)

            activities = row.activities if has_activities else None
            if activities is not None and not (
                isinstance(activities, float) and pd.isna(activities)
            ) and len(activities) > 0:
                in_interval = in_interval & df['activity'].isin(activities).to_numpy()

            mask = mask | in_interval

        return mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_exposures_context.py::TestContextMask -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/actimotus/exposures.py tests/test_exposures_context.py
git commit -m "feat: add Exposures._context_mask single-context matcher"
```

---

## Task 4: `context` (public; replaces old single-context method)

**Files:**
- Modify: `src/actimotus/exposures.py` (add the new `context` staticmethod after `_context_mask`)
- Test: `tests/test_exposures_context.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_exposures_context.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_exposures_context.py::TestContext -q`
Expected: FAIL — `TypeError` (old `context` signature expects `intervals`/`context` args) or `AttributeError` depending on state; the new behavior is not implemented yet.

- [ ] **Step 3: Implement the new `context`**

In `src/actimotus/exposures.py`, add this staticmethod directly after `_context_mask`. (The old `context(df, intervals, context, activities=None)` body was already removed in Task 2, Step 3.)

```python
    @staticmethod
    def context(df: pd.DataFrame, diary: pd.DataFrame) -> pd.DataFrame:
        """Annotate the activity DataFrame with diary contexts.

        For each distinct ``context`` in the diary, adds a boolean
        ``context__<name>`` column that is ``True`` for epochs inside any of that
        context's intervals (optionally gated by each interval's ``activities``).
        Contexts may overlap; multiple intervals for one context union into a
        single column. The input frame is not mutated — a copy is returned.

        Args:
            df: Activity DataFrame, timezone-aware DatetimeIndex, ``activity`` column.
            diary: Clean diary with columns ``start``, ``end``, ``context`` and an
                optional per-row ``activities`` list, timezone-aware and in the same
                timezone as ``df``'s index.

        Returns:
            A copy of ``df`` with one ``context__<name>`` boolean column per context.

        Raises:
            ValueError: If the diary is invalid (see :meth:`_validate_diary`), the
                index is not timezone-aware, or the diary and index timezones differ.
        """
        Exposures._validate_diary(diary)

        if df.index.tz is None:
            raise ValueError('Activity DataFrame index must be timezone-aware.')
        if str(diary['start'].dt.tz) != str(df.index.tz):
            raise ValueError(
                f"Diary timezone ({diary['start'].dt.tz}) does not match activity "
                f'index timezone ({df.index.tz}).'
            )

        df = df.copy()
        for context, intervals in diary.groupby('context', sort=False):
            df[f'context__{context}'] = Exposures._context_mask(df, intervals)

        return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_exposures_context.py::TestContext -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest tests/ -q`
Expected: PASS (15 passed).

- [ ] **Step 6: Commit**

```bash
git add src/actimotus/exposures.py tests/test_exposures_context.py
git commit -m "feat: replace Exposures.context with diary-driven multi-context annotation"
```

---

## Task 5: Documentation

`docs/references/exposures.md` already lists `context` as a documented member, and
mkdocstrings pulls the updated docstring automatically. `_validate_diary` and
`_context_mask` are private and intentionally undocumented, so there is **no
members-list change** — this task only verifies the rendered docs stay clean.

**Files:**
- (No edits expected.) `docs/references/exposures.md`

- [ ] **Step 1: Verify docs build (if mkdocs is available)**

Run: `uv run mkdocs build --strict 2>&1 | tail -20`
Expected: build completes without errors referencing `exposures.md`, and the new
`context(df, diary)` signature/docstring renders. If mkdocs is not installed, skip.

- [ ] **Step 2: No commit needed** unless the build surfaced a docs fix. If it did,
commit only that fix:

```bash
git add docs/references/exposures.md
git commit -m "docs: refresh Exposures.context reference"
```

---

## Self-Review Notes

- **Spec coverage:** `_validate_diary` (Task 2), `_context_mask` with per-row gate + union + half-open (Task 3), `context` with copy/tz-assert/prefix columns/overlap/empty (Task 4), docs (Task 5). The out-of-scope export adapter is correctly excluded.
- **Copy semantics** verified by `test_does_not_mutate_input`.
- **Strict tz rule** verified by `test_naive_timestamps_raise` and `test_timezone_mismatch_raises`.
- **Union into one column** verified by `test_multiple_intervals_one_column`.
- **Method name consistency:** `_validate_diary`, `_context_mask`, `context` used identically across tasks and tests.

## Verification Checklist (post-implementation)

- [ ] `uv run pytest tests/ -q` → all green.
- [ ] `grep -rn "Exposures.context(" dev/ examples/` — no in-repo caller relies on the **old** `(df, intervals, context, activities)` signature (only `dev/diaries.ipynb`, which is scratch and may be updated separately).
