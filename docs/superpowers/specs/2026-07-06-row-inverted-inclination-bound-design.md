# Design: bound `row` detection to non-inverted thigh inclination

- **Date:** 2026-07-06
- **Status:** proposed
- **Component:** `actimotus.classifications.thigh` (`get_row`), `actimotus.settings`
- **Motivation:** false `row` (rowing) detections from an inverted device or feet-up
  lying, which inflate MVPA. Recurring across studies; observed strongly in a Czech
  nurse cohort (no rowers) processed by the `veronika-phd` consumer.

## Problem

`row` is emitted for windows that are not rowing. In a thigh-worn cohort with no
rowers, subjects accumulate up to ~1 h/day of `row`, which is folded into **MVPA**
(`exposures.py`), the opposite of the truth (the person is sedentary/reclining).

## Root cause

`get_row` (`thigh.py`) is a two-term rule with a **lower** inclination bound and
**no upper bound**:

```python
valid = (inclination_angle < df['inclination']) & (movement_threshold < df['sd_x'])
# CONFIG['thigh']['row'] = {bout: 15, movement_threshold: 0.075, inclination_angle: 87.5}
```

`inclination = arccos(x / |axis|)` ranges 0–180°: standing `x≈+1` → 0°, thigh
horizontal `x≈0` → 90°, **inverted long-axis `x≈−1` → 180°**. The rule accepts
*everything* from 87.5° to 180°. So a device worn upside-down (or a feet-up lying
posture) reads a large inclination on ordinary sitting/reclining, and any leg motion
clears `sd_x > 0.075` → the window is tagged `row`.

Two structural facts make it stick:

1. `row` is the **highest-priority** class in the `idxmax` order
   (`row > bicycle > stairs > run > walk > stand > sit`, `thigh.py` `_get_activity_column`),
   so it overrides everything.
2. The `lie` reassignment only rescues windows already labelled `sit`
   (`get_lie`), so a `row` window is never corrected.

`row` is the **only** posture class with a lower inclination bound but no upper
bound. Every sibling is bounded: walk/stairs/run/stand require `inclination < 47.5°`,
bicycle requires `inclination < 87.5°`. `row` inherits the asymmetry that lets an
inverted device masquerade as a horizontal rowing thigh.

## Evidence (veronika-phd cohort, 74 subjects)

Per-epoch `thigh_inclination` during `row` windows:

- **Inverted-device subjects** — `row` inclination median 130–165° (subj 75: p50 132°,
  p95 169°; residual subjects 2, 49, 54, 73 at 133–164°). Clearly past horizontal.
- **Residual cohort** (excluding the two worst): `row` inclination median ~115°,
  **58% above 110°** — dominated by the inverted mechanism.
- **Real rowing reference:** a rowing thigh is at/near horizontal (~90°), reaching
  perhaps ~100° at the lean-back finish. It is never inverted (`x` never strongly
  negative).

A distinct, rarer mechanism also exists (see Limitations): correctly-oriented deep
reclining with leg motion produces `row` at ~90–100° (subj 47: p50 94°). A threshold
cannot separate that from real rowing.

## Design

Add an **upper inclination bound** to `get_row`:

```python
def get_row(self, df, bout, movement_threshold, inclination_angle,
            inclination_upper=180.0, **kwargs):
    valid = (
        (inclination_angle < df['inclination'])
        & (df['inclination'] < inclination_upper)      # NEW: reject inverted regime
        & (movement_threshold < df['sd_x'])
    )
    valid = self._median_filter(valid, bout)
    valid.name = 'row'
    return valid
```

Config (`settings.py`, active `CONFIG['thigh']['row']`):

```python
'row': {'bout': 15, 'movement_threshold': 0.075,
        'inclination_angle': 87.5, 'inclination_upper': 110.0},   # NEW key
```

- **Threshold = 110°.** Real rowing tops out around 90–100°; 110° leaves ~10–20°
  margin while rejecting the inverted regime (≥110°). Tunable via config;
  we validate the value against subject 75 during testing.
- **Backward compatible.** `inclination_upper` defaults to `180.0` (a no-op = current
  behaviour), so any caller/legacy config that does not pass it is unaffected. Only
  the active `CONFIG` sets `110.0`.

## Safety analysis — where do excluded windows go?

Because `row` is top-priority, excluding a window exposes it to the lower-priority
classifiers. Verified from the mask conditions in `thigh.py`: **a window with
inclination > 110° cannot satisfy any class above `sit`.** Each has a hard,
AND-ed upper-inclination gate that no motion/cadence value can override:

| class | inclination gate | >110° window |
| --- | --- | --- |
| bicycle | `< 87.5°` | fails |
| stairs | `< 47.5°` | fails |
| run | `< 47.5°` | fails |
| walk | `< 47.5°` | fails |
| stand | `< 47.5°` | fails |
| **sit** | `> 47.5°` | **fires** |

So an excluded inverted window deterministically becomes `sit`, which the `lie`
step may further promote to `lie` (lateral-roll test). **Both are sedentary — the
correct destination.** There is **no path to walk/stairs/run/bicycle** (an MVPA
leak). `walk`/`stairs`/`run` are not "pure cadence": they each gate on
`inclination < 47.5°`, and the step feature is assigned only *after* classification,
to windows already labelled walk/stairs/run — it cannot pull a >110° window into MVPA.

**Edge case (not an MVPA risk):** a very short, isolated >110° blip can be dropped
from `sit` by the median filter and fall to the always-true `shuffle` fallback
(a light/standing class → maps to `stand`, not MVPA). Sustained inverted/feet-up
bouts — the realistic case — stay solidly `sit`/`lie`.

## Limitations (explicitly accepted)

This fix targets the **inverted-device** mechanism only. It does **not** address
**correctly-oriented deep reclining with leg motion** (thigh genuinely ~horizontal,
`inclination ≈ 90–100°`, `x≈0`), which is *posturally indistinguishable from real
rowing*. Separating those would require a stroke-cadence/periodicity feature and
rowing ground-truth we do not have. In the observed cohort this residue is ~0.36 h
across four subjects (noise). We accept it as a documented limitation rather than
introduce an unvalidated cadence heuristic that could harm genuine rowing detection.

## Testing

Unit tests (`tests/`), synthetic windows:

1. Inverted window (`inclination` 150°, `sd_x` 0.2) → **not** `row`; resolves to
   `sit`/`lie`.
2. Horizontal window (`inclination` 92°, `sd_x` 0.2) → **still** `row` (genuine
   rowing preserved).
3. Boundary: 109° → row; 111° → not row (bound at 110°).
4. Backward-compat: `get_row` without `inclination_upper` reproduces the old result
   on an inverted window (default 180° = no-op).
5. No-MVPA-leak regression: an inverted moving window never resolves to
   walk/stairs/run/bicycle.

Real-data validation (via the `veronika-phd` reprocess):

- **Subj 75** (inverted): `row` collapses; those epochs become `sit`/`lie`; total
  MVPA drops; real walking/standing unchanged. Confirms the threshold.
- **Subj 47** (reclining): `row` largely **survives** — expected, this is the
  accepted-limitation mechanism, and it confirms the fix does not over-reach.

## Consumer impact (veronika-phd)

The classifier change alters `activities.parquet`, so the cohort must be reprocessed
(Features + Activities), then exposures re-derived and the exposures-QC refreshed.
This is separate from the analysis-level cleaning policy (exclude subjects 47 & 75 as
bad placement; keep subject 25's bicycle as real MVPA; multiplicative replacement for
zero-MVPA days), which the consumer applies downstream.

## Rollout

- Bump acti-motus patch version; changelog entry under fixes.
- Editable dependency, so the consumer picks it up on reprocess.
