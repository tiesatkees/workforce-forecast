# new_km.py
"""
Simple Kaplan–Meier estimator and seasonal index without external dependencies.
Assumes monthly granularity.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class KMResult:
    # Survival S(t) defined on integer months t = 0,1,2,... up to t_max
    # Stored as a numpy array of length t_max+1
    S: np.ndarray
    t_max: int

    def S_at(self, t: int) -> float:
        if t <= 0:
            return 1.0
        t = min(max(int(t), 0), self.t_max)
        return float(self.S[t])

    def hazard(self, t: int) -> float:
        # h(t) = 1 - S(t+1)/S(t)   clipped to [0,1]
        s_t = max(self.S_at(t), 1e-12)
        s_tp1 = self.S_at(t+1) if (t+1) <= self.t_max else self.S_at(self.t_max)
        h = 1.0 - (s_tp1 / s_t)
        return float(min(max(h, 0.0), 1.0))

def compute_km(durations_months: np.ndarray, events: np.ndarray, max_months: Optional[int]=None) -> KMResult:
    """
    durations_months: float or int months (time-to-event or censor time)
    events: 1 if event (exit), 0 if right-censored
    """
    durations = np.asarray(durations_months, dtype=float)
    events = np.asarray(events, dtype=int)
    assert durations.shape == events.shape, "durations/events length mismatch"

    # Group counts by integer month (floor) as event times
    # We will build standard KM: product over unique event times t of (1 - d_t / n_t)
    t_int = np.floor(durations + 1e-9).astype(int)
    # build risk set counts
    t_max = int(np.max(t_int)) if max_months is None else int(max_months)
    if np.isnan(t_max) or t_max < 0:
        t_max = 0

    # For each t, d_t = number of events (exits) at t, c_t = censored at t
    d = np.zeros(t_max+1, dtype=float)
    c = np.zeros(t_max+1, dtype=float)
    for ti, ev in zip(t_int, events):
        if ti > t_max:
            continue
        if ev == 1:
            d[ti] += 1
        else:
            c[ti] += 1

    # Number at risk n_t just before t: n_t = n_{t-1} - d_{t-1} - c_{t-1}, with n_0 = N
    N = len(durations)
    n_at_risk = np.zeros(t_max+1, dtype=float)
    n_at_risk[0] = N
    for t in range(1, t_max+1):
        n_at_risk[t] = n_at_risk[t-1] - d[t-1] - c[t-1]
        if n_at_risk[t] < 0:
            n_at_risk[t] = 0.0

    # KM survival
    S = np.ones(t_max+1, dtype=float)
    for t in range(0, t_max+1):
        if n_at_risk[t] > 0 and d[t] > 0:
            S[t:] *= (1.0 - d[t]/n_at_risk[t])
        # if no events at t, survival stays flat

    return KMResult(S=S, t_max=t_max)

def month_index_to_calendar(start_year: int, start_month: int, t: int) -> Tuple[int,int]:
    m0 = (start_year * 12 + (start_month-1)) + t
    y = m0 // 12
    m = (m0 % 12) + 1
    return (y, m)

def compute_seasonal_index(exits_by_month: Dict[int,int], active_by_month: Dict[int,int]) -> Dict[int,float]:
    """
    Build a simple seasonal index SEAS[1..12] based on exit rate per month
    normalized by the mean across all months.
    exits_by_month: dict {1..12 -> exit count}
    active_by_month: dict {1..12 -> avg active count}
    """
    months = range(1,13)
    rate = {}
    for m in months:
        e = exits_by_month.get(m, 0)
        a = max(active_by_month.get(m, 0), 1e-9)
        rate[m] = e / a
    mean_rate = np.mean([rate[m] for m in months])
    seas = {m: (rate[m] / mean_rate if mean_rate > 0 else 1.0) for m in months}
    # Smooth/clip extremes a bit
    for m in months:
        seas[m] = float(np.clip(seas[m], 0.5, 1.5))
    return seas
