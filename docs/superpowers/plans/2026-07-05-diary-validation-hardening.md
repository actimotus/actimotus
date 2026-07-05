# Diary Validation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn malformed diary input from silent-wrong/crashing into clear `ValueError`s at the boundary, plus normalize surrounding whitespace in context names.

**Architecture:** Extend the two existing static methods on `Exposures` in `src/actimotus/exposures.py`. `_validate_diary(diary)` gains diary-only checks (NaT, context validity, activities well-formedness incl. labels ∈ `settings.ACTIVITIES`). `context(df, diary)` gains diary↔df checks (activity column present, `start`/`end` zones match the index, no `context__<name>` collision) and strips surrounding whitespace from context names on its local copy. `_context_mask` is left unchanged.

**Tech Stack:** Python, pandas, pytest. Run tests with `uv run pytest`. `ACTIVITIES` and `pd` are already imported in `exposures.py`; no new imports are needed.

**Spec:** `docs/superpowers/specs/2026-07-05-diary-validation-hardening-design.md`

---

## File Structure

- Modify: `src/actimotus/exposures.py` — replace the body of `_validate_diary` (currently lines 325-353) and `context` (currently lines 388-425). `_context_mask` (lines 355-386) is untouched.
- Modify: `tests/test_exposures_context.py` — add tests to the existing `TestValidateDiary` and `TestContext` classes.

The `activity`-label check uses `ACTIVITIES` (already imported: `from .settings import ACTIVITIES, FUSED_ACTIVITIES, PLOT, PLOT_FUSED`). Its values are the 13 canonical labels: `non-wear, lie, sit, stand, shuffle, walk, run, stairs, bicycle, row, kneel, squat, fast-walk`.

Baseline before this plan: the suite has **21 passing tests**.

---

## Task 1: Harden `_validate_diary` (NaT, context, activities)

**Files:**
- Modify: `src/actimotus/exposures.py` (replace `_validate_diary`, lines 325-353)
- Test: `tests/test_exposures_context.py` (add to class `TestValidateDiary`)

- [ ] **Step 1: Write the failing tests**

Append these methods to the existing `TestValidateDiary` class in `tests/test_exposures_context.py` (keep the existing tests):

```python
    def test_nat_in_start_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', None)])
        diary.loc[0, 'start'] = pd.NaT
        with pytest.raises(ValueError, match='NaT'):
            Exposures._validate_diary(diary)

    def test_nat_in_end_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', None)])
        diary.loc[0, 'end'] = pd.NaT
        with pytest.raises(ValueError, match='NaT'):
            Exposures._validate_diary(diary)

    def test_null_context_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', None)])
        diary.loc[0, 'context'] = None
        with pytest.raises(ValueError, match='context'):
            Exposures._validate_diary(diary)

    def test_non_string_context_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', None)])
        diary.loc[0, 'context'] = 123
        with pytest.raises(ValueError, match='context'):
            Exposures._validate_diary(diary)

    def test_empty_context_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', '   ', None)])
        with pytest.raises(ValueError, match='context'):
            Exposures._validate_diary(diary)

    def test_activities_pd_na_passes(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', pd.NA)])
        assert Exposures._validate_diary(diary) is None

    def test_activities_bare_string_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', 'sit')])
        with pytest.raises(ValueError, match='activities'):
            Exposures._validate_diary(diary)

    def test_activities_number_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', 5)])
        with pytest.raises(ValueError, match='activities'):
            Exposures._validate_diary(diary)

    def test_activities_non_string_element_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', ['sit', 5])])
        with pytest.raises(ValueError, match='activities'):
            Exposures._validate_diary(diary)

    def test_activities_unknown_label_raises(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', ['sleeping'])])
        with pytest.raises(ValueError, match='unknown'):
            Exposures._validate_diary(diary)

    def test_activities_valid_labels_passes(self, diary_factory):
        diary = diary_factory([('2024-09-02 07:00', '2024-09-02 07:05', 'work', ['lie', 'sit'])])
        assert Exposures._validate_diary(diary) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_exposures_context.py::TestValidateDiary -q`
Expected: FAIL — several of the new tests fail (no `NaT`/`context`/`activities` checks exist yet); e.g. `test_activities_bare_string_raises` currently raises no error, `test_nat_in_start_raises` does not raise.

- [ ] **Step 3: Replace `_validate_diary` with the hardened version**

In `src/actimotus/exposures.py`, replace the entire `_validate_diary` staticmethod (currently lines 325-353) with:

```python
    @staticmethod
    def _validate_diary(diary: pd.DataFrame) -> None:
        """Validate a diary of context intervals.

        The diary must have columns ``start``, ``end``, ``context`` (and an
        optional ``activities`` column). ``start``/``end`` must be timezone-aware
        datetimes without ``NaT`` and every ``end`` must be strictly after its
        ``start``. Each ``context`` must be a non-empty string. Each ``activities``
        cell must be missing (``None``/``NaN``/``pd.NA``) or a list of known
        ``ACTIVITIES`` labels.

        Args:
            diary: DataFrame with columns ``start``, ``end``, ``context`` and an
                optional per-row ``activities`` list.

        Raises:
            ValueError: If required columns are missing; ``start``/``end`` are not
                timezone-aware or contain ``NaT``; any interval has ``end <= start``;
                a ``context`` is null, non-string, or empty; or an ``activities``
                cell is neither missing nor a list of known ``ACTIVITIES`` labels.
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
            if diary[column].isna().any():
                raise ValueError(f"Diary column '{column}' contains NaT (missing timestamps).")

        if (diary['end'] <= diary['start']).any():
            raise ValueError("Diary has rows where 'end' is not after 'start'.")

        for context in diary['context']:
            if not isinstance(context, str) or not context.strip():
                raise ValueError(
                    f'Diary has an invalid context value: {context!r}. '
                    'Context must be a non-empty string.'
                )

        if 'activities' in diary.columns:
            known = set(ACTIVITIES.values())
            for activities in diary['activities']:
                if isinstance(activities, list):
                    for label in activities:
                        if not isinstance(label, str):
                            raise ValueError(
                                f'Diary activities must be strings; got {label!r}.'
                            )
                        if label not in known:
                            raise ValueError(
                                f'Diary activities contains unknown label {label!r}. '
                                f'Known labels: {sorted(known)}.'
                            )
                elif pd.api.types.is_scalar(activities) and pd.isna(activities):
                    continue  # missing == no gate
                else:
                    raise ValueError(
                        f'Diary activities must be a list of labels or missing; '
                        f'got {activities!r}.'
                    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_exposures_context.py::TestValidateDiary -q`
Expected: PASS. The class now has the original 6 plus 11 new tests = 17 passed.

- [ ] **Step 5: Commit**

```bash
git add src/actimotus/exposures.py tests/test_exposures_context.py
git commit -m "feat: harden _validate_diary against NaT, bad context, malformed activities"
```

---

## Task 2: Harden `context` (df checks) + normalize context whitespace

**Files:**
- Modify: `src/actimotus/exposures.py` (replace `context`, lines 388-425)
- Test: `tests/test_exposures_context.py` (add to class `TestContext`)

- [ ] **Step 1: Write the failing tests**

Append these methods to the existing `TestContext` class in `tests/test_exposures_context.py` (keep the existing tests):

```python
    def test_missing_activity_column_raises(self, activities, diary_factory):
        df = activities.drop(columns=['activity'])
        diary = diary_factory([('2024-09-02 07:00:00', '2024-09-02 07:00:05', 'work', None)])
        with pytest.raises(ValueError, match='activity'):
            Exposures.context(df, diary)

    def test_end_timezone_mismatch_raises(self, activities, diary_factory):
        diary = diary_factory([('2024-09-02 07:00:00', '2024-09-02 07:00:05', 'work', None)])
        diary['end'] = diary['end'].dt.tz_convert('America/New_York')
        with pytest.raises(ValueError, match='timezone'):
            Exposures.context(activities, diary)

    def test_existing_context_column_collision_raises(self, activities, diary_factory):
        df = activities.copy()
        df['context__work'] = False
        diary = diary_factory([('2024-09-02 07:00:00', '2024-09-02 07:00:05', 'work', None)])
        with pytest.raises(ValueError, match='context'):
            Exposures.context(df, diary)

    def test_whitespace_context_normalized(self, activities, diary_factory):
        diary = diary_factory([('2024-09-02 07:00:00', '2024-09-02 07:00:05', ' work ', None)])
        result = Exposures.context(activities, diary)
        assert 'context__work' in result.columns
        assert 'context__ work ' not in result.columns

    def test_whitespace_variants_merge_one_column(self, activities, diary_factory):
        diary = diary_factory([
            ('2024-09-02 07:00:00', '2024-09-02 07:00:02', ' work ', None),
            ('2024-09-02 07:00:08', '2024-09-02 07:00:10', 'work', None),
        ])
        result = Exposures.context(activities, diary)
        work_cols = [c for c in result.columns if c.startswith('context__work')]
        assert work_cols == ['context__work']
        assert list(result['context__work']) == \
            [True, True, False, False, False, False, False, False, True, True]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_exposures_context.py::TestContext -q`
Expected: FAIL — e.g. `test_missing_activity_column_raises` raises `KeyError` not `ValueError`, `test_whitespace_context_normalized` produces `context__ work ` instead of `context__work`, `test_existing_context_column_collision_raises` silently overwrites.

- [ ] **Step 3: Replace `context` with the hardened version**

In `src/actimotus/exposures.py`, replace the entire `context` staticmethod (currently lines 388-425) with:

```python
    @staticmethod
    def context(df: pd.DataFrame, diary: pd.DataFrame) -> pd.DataFrame:
        """Annotate the activity DataFrame with diary contexts.

        For each distinct ``context`` in the diary, adds a boolean
        ``context__<name>`` column that is ``True`` for epochs inside any of that
        context's intervals (optionally gated by each interval's ``activities``).
        Contexts may overlap; multiple intervals for one context union into a
        single column. Surrounding whitespace in context names is stripped, so
        ``' work '`` and ``'work'`` collapse into one column. The input frame is not
        mutated — a copy is returned.

        Args:
            df: Activity DataFrame, timezone-aware DatetimeIndex, ``activity`` column.
            diary: Clean diary with columns ``start``, ``end``, ``context`` and an
                optional per-row ``activities`` list, timezone-aware and in the same
                timezone as ``df``'s index.

        Returns:
            A copy of ``df`` with one ``context__<name>`` boolean column per context.

        Raises:
            ValueError: If the diary is invalid (see :meth:`_validate_diary`); ``df``
                has no ``activity`` column or a timezone-naive index; the diary
                ``start``/``end`` zones differ from the index zone; or ``df`` already
                has a ``context__<name>`` column that this call would create.
        """
        Exposures._validate_diary(diary)

        if 'activity' not in df.columns:
            raise ValueError("Activity DataFrame must have an 'activity' column.")

        if df.index.tz is None:
            raise ValueError('Activity DataFrame index must be timezone-aware.')

        index_tz = str(df.index.tz)
        for column in ('start', 'end'):
            if str(diary[column].dt.tz) != index_tz:
                raise ValueError(
                    f"Diary '{column}' timezone ({diary[column].dt.tz}) does not "
                    f'match activity index timezone ({df.index.tz}).'
                )

        diary = diary.copy()
        diary['context'] = diary['context'].str.strip()

        new_columns = {f'context__{context}' for context in diary['context'].unique()}
        collisions = new_columns & set(df.columns)
        if collisions:
            raise ValueError(
                f'Activity DataFrame already has context columns: {sorted(collisions)}.'
            )

        df = df.copy()
        for context, intervals in diary.groupby('context', sort=False):
            df[f'context__{context}'] = Exposures._context_mask(df, intervals)

        return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_exposures_context.py::TestContext -q`
Expected: PASS. The class now has its original 8 plus 5 new tests = 13 passed.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest tests/ -q`
Expected: PASS (37 passed: 17 `TestValidateDiary` + 7 `TestContextMask` + 13 `TestContext`).

- [ ] **Step 6: Commit**

```bash
git add src/actimotus/exposures.py tests/test_exposures_context.py
git commit -m "feat: harden context() with df guards and context whitespace normalization"
```

---

## Self-Review Notes

- **Spec coverage:** `_validate_diary` checks (NaT, context, activities incl. label validation) → Task 1. `context` checks (activity column, `start`+`end` zone match, collision) and whitespace normalization → Task 2. `_context_mask` deliberately unchanged. Empty-diary and `[]`-as-no-gate preserved (no test removed; existing tests still run).
- **Missing-detection ordering:** `_validate_diary` checks `isinstance(list)` before `pd.api.types.is_scalar(...) and pd.isna(...)`, so `pd.isna` is never called on a list/array. A bare string/number/tuple falls through to the final `raise`.
- **Copy semantics:** `context` strips on `diary.copy()` and annotates `df.copy()`; the caller's frames are never mutated (already covered by the existing `test_does_not_mutate_input`).
- **Type/name consistency:** `_validate_diary`, `_context_mask`, `context`, and the `context__<name>` prefix are used identically across tasks and tests.

## Verification Checklist (post-implementation)

- [ ] `uv run pytest tests/ -q` → 37 passed.
- [ ] `grep -n "is_scalar\|ACTIVITIES" src/actimotus/exposures.py` — confirm `ACTIVITIES` and `pd.api.types.is_scalar` are used and require no new imports.
