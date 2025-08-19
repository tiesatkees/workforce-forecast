#!/usr/bin/env python3
# hire_planner.py  –  twee algoritmen + “combi” (min → laatst-mogelijk)
# ---------------------------------------------------------------------
from __future__ import annotations
from math import ceil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────   hulpfuncties   ──────────────────────────
def _apply_ramp(cap: np.ndarray, month: int, n: int,
                ramp: List[int]) -> None:
    """Verhoog capaciteit‐vector in-place met `n*ramp` vanaf maand `month`."""
    for j, step in enumerate(ramp):
        t = month + j
        if t < len(cap):
            cap[t] += n * step


def _shortage_vector(demand: np.ndarray, cap: np.ndarray,
                     buffer: int) -> np.ndarray:
    """Positieve tekorten per maand (negatieve waarden = overschot)."""
    return np.maximum(0, demand + buffer - cap)


# ─────────────────────────   algoritme 1   ──────────────────────────
def plan_hires_earliest(demand: np.ndarray,
                        cap_start: np.ndarray,
                        ramp: List[int],
                        buffer: int = 0,
                        max_pm: int = 3) -> Dict[int, int]:
    """
    *‘Earliest fill’* – plant hires zo VROEG mogelijk → minimaal totaal.
    Keert dict {maand:index, hires:int}.
    """
    T, cap = len(demand), cap_start.copy()
    plan: Dict[int, int] = {}

    for m in range(T):                              # voorwaarts
        shortage = demand[m] + buffer - cap[m]
        if shortage <= 0:
            continue

        n = min(max_pm, ceil(shortage / ramp[-1]))  # vol vermogen telt
        plan[m] = plan.get(m, 0) + n
        _apply_ramp(cap, m, n, ramp)

    return plan


# ─────────────────────────   algoritme 2   ──────────────────────────
def plan_hires_latest(demand: np.ndarray,
                      cap_start: np.ndarray,
                      ramp: List[int],
                      buffer: int = 0,
                      max_pm: int = 3) -> Dict[int, int]:
    """
    *‘Latest fill’* – plant hires zo LAAT mogelijk binnen lead-time.
    """
    T          = len(demand)
    cap        = cap_start.copy()
    plan: Dict[int, int] = {}
    lead_time  = next(i for i, c in enumerate(ramp) if c > 0)
    full_cap   = ramp[-1]

    m = 0
    while m < T:
        shortage = demand[m] + buffer - cap[m]
        if shortage <= 0:
            m += 1
            continue

        need   = min(max_pm, ceil(shortage / full_cap))
        hire_m = m - lead_time
        if hire_m < 0:
            raise RuntimeError("Onoplosbaar tekort – lead-time te lang.")

        plan[hire_m] = plan.get(hire_m, 0) + need
        _apply_ramp(cap, hire_m, need, ramp)
        # her‐evaluate zelfde maand m zonder +=1

    return plan


# ─────── 3.  “combi” – eerst min, dán naar achter schuiven ───────────
def _shift_plan_to_latest(plan: Dict[int, int],
                          demand: np.ndarray,
                          cap_start: np.ndarray,
                          ramp: List[int],
                          buffer: int) -> Dict[int, int]:
    """
    Neem een bestaand plan (meestal earliest/min) en schuif elke hire
    zo ver mogelijk naar ACHTEREN zonder tekorten te veroorzaken.
    """
    T        = len(demand)
    new_plan = plan.copy()

    # probeer elke hire (oplopend) te verplaatsen
    for month in sorted(plan):
        n_hires = plan[month]
        # verwijder hires uit temp-plan & capaciteit
        tmp_plan = new_plan.copy()
        tmp_plan[month] -= n_hires
        if tmp_plan[month] == 0:
            tmp_plan.pop(month)
        cap = cap_start.copy()
        for m, n in tmp_plan.items():
            _apply_ramp(cap, m, n, ramp)

        # zoek laatste maand waarop je ze nog kunt zetten
        for target in range(month + 1, T):
            _apply_ramp(cap, target, n_hires, ramp)
            if (_shortage_vector(demand, cap, buffer) > 0).any():
                # tekort → stap terug & vastzetten
                _apply_ramp(cap, target, -n_hires, ramp)
                target -= 1
                break
            _apply_ramp(cap, target, -n_hires, ramp)   # undo en probeer verder
        else:
            target = T - 1  # helemaal aan eind mogelijk

        # definitief toevoegen
        new_plan[target] = new_plan.get(target, 0) + n_hires
        if month in new_plan and new_plan[month] == n_hires:
            new_plan.pop(month)

    return new_plan


def auto_hire_plan(df_forecast: pd.DataFrame,
                   ramp: List[int],
                   buffer: int,
                   strategy: str = "earliest",
                   max_pm: int = 3) -> Dict[int, int]:
    """
    strategy = {"earliest", "latest", "min+shift"}.
    """
    demand = df_forecast["Vraag"].to_numpy()
    cap0   = df_forecast["Cap"].to_numpy()

    if strategy == "earliest":
        return plan_hires_earliest(demand, cap0, ramp, buffer, max_pm)

    if strategy == "latest":
        return plan_hires_latest(demand, cap0, ramp, buffer, max_pm)

    if strategy == "min+shift":
        early = plan_hires_earliest(demand, cap0, ramp, buffer, max_pm)
        return _shift_plan_to_latest(early, demand, cap0, ramp, buffer)

    raise ValueError(f"Onbekende strategy '{strategy}'")
