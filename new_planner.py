#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
new_planner.py — robust hire planning strategies

Exposes:
- auto_hire_plan(df_forecast, ramp, buffer, strategy="earliest", max_pm=3)
- plan_hires_earliest(demand, cap_start, ramp, buffer, max_pm=3)
- plan_hires_latest(demand, cap_start, ramp, buffer, max_pm=3)
- plan_hires_min_then_shift(demand, cap_start, ramp, buffer, max_pm=3)

Definitions
-----------
- demand: array-like of ints/floats, monthly client demand
- cap_start: array-like, capacity without new hires (same length as demand)
- ramp: array-like of nonnegative ints/floats, per-hire capacity contribution
        over successive months starting from the hire month. Example:
        [0,0,0,50,80,100,120,140,140]
- buffer: int >= 0 (extra demand cushion)
- max_pm: optional monthly cap on new hires (int >= 1)
- All months are indexed from 0.

Conventions
-----------
- lead_time = first index i with ramp[i] > 0, else +∞ (invalid if never > 0).
- full_cap = ramp[-1] (steady-state contribution per hire).
- A plan is a dict: {month_index: number_of_hires}.
- Feasibility means: for all m, cap_with_plan[m] >= demand[m] + buffer.

Notes
-----
- "latest" fills shortages as late as still feasible (just-in-time by lead_time).
- "earliest" places hires at the earliest feasible month (maximizes ramp overlap
   and typically minimizes total hires).
- "min+shift" uses the "earliest" plan (minimal hires), then pushes hires to
  later months when possible without creating shortages or violating max_pm.
"""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ───────────────────────── helpers ─────────────────────────

def _to_np(a) -> np.ndarray:
    return np.asarray(a, dtype=float)

def _validate_inputs(demand, cap_start, ramp, buffer: int, max_pm: int):
    demand = _to_np(demand)
    cap_start = _to_np(cap_start)
    ramp = _to_np(ramp)

    if demand.ndim != 1 or cap_start.ndim != 1:
        raise ValueError("demand and cap_start must be 1-D arrays")
    if len(demand) != len(cap_start):
        raise ValueError("demand and cap_start must have the same length")
    if ramp.ndim != 1 or len(ramp) == 0:
        raise ValueError("ramp must be a non-empty 1-D array")
    if np.any(ramp < 0):
        raise ValueError("ramp must be nonnegative")
    if buffer < 0:
        raise ValueError("buffer must be >= 0")
    if max_pm < 1:
        raise ValueError("max_pm must be >= 1")

    # lead time: first positive ramp index
    pos_idx = np.where(ramp > 0)[0]
    if len(pos_idx) == 0:
        raise ValueError("ramp never becomes > 0; lead-time infinite (invalid).")
    lead_time = int(pos_idx[0])
    full_cap = float(ramp[-1])

    return demand, cap_start, ramp, lead_time, full_cap


def _apply_ramp(cap: np.ndarray, month: int, n: int, ramp: np.ndarray):
    """In-place: add n * ramp shifted to start at `month`."""
    T = len(cap)
    L = len(ramp)
    start = max(month, 0)
    end = min(month + L, T)
    if start >= T or end <= start or n <= 0:
        return
    cap[start:end] += n * ramp[(start - month):(end - month)]


def _simulate(cap_start: np.ndarray, plan: Dict[int, int], ramp: np.ndarray) -> np.ndarray:
    """Return capacity timeline after applying plan to cap_start."""
    cap = cap_start.copy().astype(float)
    if plan:
        for m, n in sorted(plan.items()):
            _apply_ramp(cap, m, int(n), ramp)
    return cap


def _is_feasible(plan: Dict[int, int],
                 demand: np.ndarray,
                 cap_start: np.ndarray,
                 ramp: np.ndarray,
                 buffer: int) -> bool:
    cap = _simulate(cap_start, plan, ramp)
    need = demand + buffer
    return np.all(cap + 1e-9 >= need)  # small epsilon to avoid float issues


def _enforce_max_pm(plan: Dict[int, int], max_pm: int) -> bool:
    """Return True if plan obeys max_pm for all months."""
    return all(n <= max_pm for n in plan.values())


# ─────────────────────── strategies ───────────────────────

def plan_hires_latest(demand, cap_start, ramp, buffer: int, max_pm: int = 3) -> Dict[int, int]:
    """
    Place hires as late as possible: when a shortage is detected at month m,
    schedule enough hires at m - lead_time so their first positive ramp effect
    hits month m. If m - lead_time < 0 → impossible (shortage too early).
    """
    demand, cap_start, ramp, lead_time, full_cap = _validate_inputs(demand, cap_start, ramp, buffer, max_pm)
    T = len(demand)

    cap = cap_start.copy().astype(float)
    need_vec = demand + buffer
    plan: Dict[int, int] = {}

    m = 0
    while m < T:
        shortage = need_vec[m] - cap[m]
        if shortage <= 1e-9:
            m += 1
            continue

        if full_cap <= 0:
            raise RuntimeError("full_cap (ramp[-1]) <= 0, cannot cover shortages.")

        hire_m = m - lead_time
        if hire_m < 0:
            raise RuntimeError("Onoplosbaar tekort – lead-time te lang voor de eerste tekorten.")

        need = int(ceil(shortage / full_cap))
        # respect max_pm *in that hire month*
        addable = max_pm - plan.get(hire_m, 0)
        if addable <= 0:
            # can't add more in this month → try to spread across previous months within lead window
            # Greedy: distribute missing hires to previous months (hire_m-1, hire_m-2, ...)
            missing = need
            for hm in range(hire_m, -1, -1):
                add_here = min(max_pm - plan.get(hm, 0), missing)
                if add_here > 0:
                    plan[hm] = plan.get(hm, 0) + add_here
                    _apply_ramp(cap, hm, add_here, ramp)
                    missing -= add_here
                    if missing <= 0:
                        break
            if missing > 0:
                raise RuntimeError("Onoplosbaar tekort – max hires per maand verhindert tijdige dekking.")
        else:
            use = min(addable, need)
            plan[hire_m] = plan.get(hire_m, 0) + use
            _apply_ramp(cap, hire_m, use, ramp)

            remaining = need - use
            # If still missing, distribute further back in time.
            hm = hire_m - 1
            while remaining > 0 and hm >= 0:
                can = max_pm - plan.get(hm, 0)
                if can > 0:
                    take = min(can, remaining)
                    plan[hm] = plan.get(hm, 0) + take
                    _apply_ramp(cap, hm, take, ramp)
                    remaining -= take
                hm -= 1
            if remaining > 0:
                raise RuntimeError("Onoplosbaar tekort – max hires per maand verhindert tijdige dekking.")

        # re-check same month m (capacity updated)
        # do not increment m here
        if cap[m] + 1e-9 >= need_vec[m]:
            m += 1

    return plan


def plan_hires_earliest(demand, cap_start, ramp, buffer: int, max_pm: int = 3) -> Dict[int, int]:
    """
    Place hires as early as possible while fixing each detected shortage.
    For a shortage at month m, schedule at the *earliest* month that can still
    affect m (i.e., max(0, m - lead_time)), then continue.
    This tends to minimize total hires due to maximum ramp overlap.
    """
    demand, cap_start, ramp, lead_time, full_cap = _validate_inputs(demand, cap_start, ramp, buffer, max_pm)
    T = len(demand)
    cap = cap_start.copy().astype(float)
    need_vec = demand + buffer
    plan: Dict[int, int] = {}

    m = 0
    while m < T:
        shortage = need_vec[m] - cap[m]
        if shortage <= 1e-9:
            m += 1
            continue

        if full_cap <= 0:
            raise RuntimeError("full_cap (ramp[-1]) <= 0, cannot cover shortages.")

        hire_m_earliest = max(0, m - lead_time)

        # compute how many needed
        need = int(ceil(shortage / full_cap))

        # try to place as early as possible: fill hire_m_earliest first,
        # then move forward (hire_m_earliest+1, ...) while staying <= m - lead_time
        placed = 0
        hm = hire_m_earliest
        last_ok = m - lead_time
        while placed < need and hm <= last_ok:
            can = max_pm - plan.get(hm, 0)
            if can > 0:
                take = min(can, need - placed)
                plan[hm] = plan.get(hm, 0) + take
                _apply_ramp(cap, hm, take, ramp)
                placed += take
            hm += 1

        if placed < need:
            raise RuntimeError("Onoplosbaar tekort – max hires per maand verhindert tijdige dekking.")

        # re-evaluate same month after capacity update
        if cap[m] + 1e-9 >= need_vec[m]:
            m += 1

    return plan


def plan_hires_min_then_shift(demand, cap_start, ramp, buffer: int, max_pm: int = 3) -> Dict[int, int]:
    """
    Two-step:
    1) Build an earliest-feasible plan (typically minimal number of hires).
    2) Greedily shift hires to later months when possible without breaking feasibility
       or exceeding max_pm. Iterate until no further shifts are possible.
    """
    demand, cap_start, ramp, lead_time, full_cap = _validate_inputs(demand, cap_start, ramp, buffer, max_pm)

    # Step 1: minimal hires
    base_plan = plan_hires_earliest(demand, cap_start, ramp, buffer, max_pm)

    # Step 2: shift to latest
    plan = dict(base_plan)  # copy
    changed = True
    T = len(demand)
    while changed:
        changed = False
        # Iterate hires from latest to earliest month for stability
        for month in sorted(list(plan.keys()), reverse=True):
            count_here = plan.get(month, 0)
            if count_here <= 0:
                continue

            # Try to shift one hire at a time
            moves_made = 0
            while plan.get(month, 0) > 0:
                target = month + 1
                if target >= T:
                    break  # cannot move beyond horizon
                if plan.get(target, 0) >= max_pm:
                    break  # monthly cap prevents shifting

                # Propose move: move 1 hire from month -> target
                candidate = dict(plan)
                candidate[month] = candidate.get(month, 0) - 1
                if candidate[month] == 0:
                    del candidate[month]
                candidate[target] = candidate.get(target, 0) + 1

                if _is_feasible(candidate, demand, cap_start, ramp, buffer):
                    plan = candidate
                    changed = True
                    moves_made += 1
                else:
                    break  # further shifting this hire breaks feasibility

            # (optional) continue to next month
    return plan


# ─────────────────────── public API ───────────────────────

def auto_hire_plan(df_forecast: pd.DataFrame,
                   ramp: List[float] | np.ndarray,
                   buffer: int,
                   strategy: str = "earliest",
                   max_pm: int = 3) -> Dict[int, int]:
    """
    Convenience wrapper using the base forecast DataFrame (without hires).
    Expects columns: 'Vraag' and 'Cap'.
    """
    if "Vraag" not in df_forecast.columns or "Cap" not in df_forecast.columns:
        raise KeyError("df_forecast must contain columns 'Vraag' and 'Cap'.")

    demand = df_forecast["Vraag"].to_numpy(dtype=float)
    cap0 = df_forecast["Cap"].to_numpy(dtype=float)

    strategy = (strategy or "earliest").lower().strip()
    if strategy in ("earliest", "asap", "min"):
        plan = plan_hires_earliest(demand, cap0, ramp, buffer, max_pm)
    elif strategy in ("latest", "jit", "just-in-time"):
        plan = plan_hires_latest(demand, cap0, ramp, buffer, max_pm)
    elif strategy in ("min+shift", "minshift", "shift"):
        plan = plan_hires_min_then_shift(demand, cap0, ramp, buffer, max_pm)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Final safety check (should hold by construction)
    if not _is_feasible(plan, _to_np(demand), _to_np(cap0), _to_np(ramp), buffer):
        raise RuntimeError("Internal error: produced plan is not feasible.")

    if not _enforce_max_pm(plan, max_pm):
        raise RuntimeError("Internal error: produced plan violates max_pm.")

    return plan
