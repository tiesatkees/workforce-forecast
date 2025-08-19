# new_forecast.py
"""
End-to-end forecast combining:
- KM-based hazard (monthly) + seasonal multiplier
- Simulation of active employees attrition
- Capacity = employees * cap_per_employee * availability + hires_ramp_contrib
- Monte Carlo uncertainty bands
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import date
import numpy as np
import pandas as pd

from new_settings import Config
from new_km import compute_km, month_index_to_calendar
from new_planner import auto_hire_plan

# ------------------------- Data Loading -------------------------

def _read_table(path: str) -> pd.DataFrame:
    if path is None:
        raise ValueError("Required file path is None.")
    path = str(path)
    if path.lower().endswith(('.xlsx','.xls')):
        return pd.read_excel(path)
    elif path.lower().endswith(('.csv','.txt')):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def _to_months(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, errors="coerce")

def _months_between(a: pd.Series, b: pd.Series) -> np.ndarray:
    # approximate months
    delta_days = (b - a).dt.days.to_numpy()
    return np.where(np.isfinite(delta_days), delta_days / 30.4375, np.nan)

@dataclass
class Inputs:
    durations: np.ndarray  # months
    events: np.ndarray     # 1=exit, 0=censored
    exits_by_month: Dict[int,int]
    active_by_month: Dict[int,int]
    start_active_employees: int
    demand: np.ndarray     # length T

def load_inputs(cfg: Config) -> Inputs:
    # Load historic KdB + Founders (can be either; concatenate if both given)
    frames = []
    if cfg.kdb_file: frames.append(_read_table(cfg.kdb_file))
    if cfg.founders_file: frames.append(_read_table(cfg.founders_file))
    if not frames:
        raise ValueError("Provide at least one historic roster file (kdb_file or founders_file).")
    hist = pd.concat(frames, ignore_index=True)

    # Parse dates
    start = _to_months(hist.get(cfg.col_start))
    end = _to_months(hist.get(cfg.col_end))
    if start is None or start.isna().all():
        raise ValueError(f"Historic files must contain column '{cfg.col_start}'.")
    if end is None:
        end = pd.Series([pd.NaT]*len(hist))

    as_of = pd.Timestamp(cfg.as_of)
    # Right-censor at as_of
    end_eff = end.fillna(as_of).clip(upper=as_of)

    durations = _months_between(start, end_eff)
    events = (end_eff < as_of).astype(int).to_numpy()  # 1 if truly ended before as_of

    # Monthly exits & active for seasonality
    hist["start"] = start
    hist["end"] = end
    hist["active_asof"] = (start <= as_of) & ((end.isna()) | (end > as_of))
    # exits per calendar month observed historically
    hist_exit = hist.loc[hist["end"].notna()].copy()
    hist_exit["end_month"] = hist_exit["end"].dt.month
    exits_by_month = hist_exit.groupby("end_month")["end_month"].size().to_dict()
    # approx active per month: count records active in that month (rough proxy)
    # to keep this light, use as_of snapshot for normalization
    active_asof_count = int(hist["active_asof"].sum())
    active_by_month = {m: max(active_asof_count, 1) for m in range(1,13)}  # simple baseline

    # Active roster for starting employees
    if cfg.active_file:
        active_df = _read_table(cfg.active_file)
        a_start = _to_months(active_df.get(cfg.col_start))
        a_end = _to_months(active_df.get(cfg.col_end)) if cfg.col_end in active_df.columns else pd.Series([pd.NaT]*len(active_df))
        act_asof = (a_start <= as_of) & ((a_end.isna()) | (a_end > as_of))
        start_active_employees = int(act_asof.sum())
    else:
        start_active_employees = active_asof_count  # fallback

    # Demand
    if cfg.demand_file:
        dem = _read_table(cfg.demand_file)
        col = cfg.col_demand.lower()
        # try several common variants
        cols_lower = {c.lower(): c for c in dem.columns}
        if col not in cols_lower:
            for alt in ["vraag","demand","klantvraag"]:
                if alt in cols_lower:
                    col = alt
                    break
        demand_series = pd.to_numeric(dem[cols_lower[col]], errors="coerce").dropna().to_numpy(dtype=float)
    else:
        raise ValueError("Provide demand_file with a column 'vraag' (or demand/klantvraag).")

    # Make sure the demand covers horizon
    T = cfg.horizon_months
    if len(demand_series) < T:
        # pad with last value
        last = demand_series[-1] if len(demand_series) > 0 else 0.0
        demand = np.pad(demand_series, (0, T - len(demand_series)), mode='constant', constant_values=last)
    else:
        demand = demand_series[:T]

    return Inputs(
        durations=np.nan_to_num(durations, nan=0.0),
        events=events,
        exits_by_month=exits_by_month,
        active_by_month=active_by_month,
        start_active_employees=start_active_employees,
        demand=demand
    )

# ------------------------- Forecast core -------------------------

def build_base_forecast(cfg: Config, inp: Inputs) -> pd.DataFrame:
    # KM
    km = compute_km(inp.durations, inp.events, max_months=120)
    # Seasonal
    # basic seasonal rate multiplier 1..12
    from new_km import compute_seasonal_index
    seas = compute_seasonal_index(inp.exits_by_month, inp.active_by_month)

    T = cfg.horizon_months
    emp = np.zeros(T, dtype=float)
    cap_no_hires = np.zeros(T, dtype=float)

    employees = float(inp.start_active_employees)
    for t in range(T):
        # hazard for month t (from KM) times seasonal index for calendar month
        year, month = month_index_to_calendar(cfg.start_year, cfg.start_month, t)
        h = km.hazard(t) * float(seas.get(month, 1.0))
        h = float(np.clip(h, 0.0, 1.0))
        # expected remaining after attrition
        leavers = employees * h
        employees = max(employees - leavers, 0.0)

        emp[t] = employees
        cap_no_hires[t] = employees * cfg.cap_per_employee * cfg.availability_factor

    # Construct DataFrame
    idx = pd.period_range(f"{cfg.start_year}-{cfg.start_month:02d}", periods=T, freq="M").to_timestamp()
    df = pd.DataFrame({
        "Maand": idx,
        "Vraag": inp.demand.astype(float),
        "Cap": cap_no_hires,
    })
    df["OK"] = (df["Cap"] + 1e-9 >= df["Vraag"] + cfg.buffer)
    df["Tekort"] = np.maximum(0.0, (df["Vraag"] + cfg.buffer) - df["Cap"])
    return df

def apply_plan(df_base: pd.DataFrame, plan: Dict[int,int], ramp: List[float]) -> pd.DataFrame:
    df = df_base.copy()
    T = len(df)
    ramp = np.asarray(ramp, dtype=float)
    cap = df["Cap"].to_numpy(dtype=float).copy()
    for m, n in sorted((plan or {}).items()):
        n = int(n)
        if n <= 0: 
            continue
        start = max(0, m)
        end = min(T, m + len(ramp))
        if start >= end:
            continue
        cap[start:end] += n * ramp[:(end - start)]
    df["Cap"] = cap
    return df

def recompute_metrics(df: pd.DataFrame, buffer: int) -> pd.DataFrame:
    df = df.copy()
    need = df["Vraag"] + buffer
    df["Tekort"] = np.maximum(0.0, need - df["Cap"])
    df["OK"] = (df["Tekort"] <= 1e-9)
    return df

def forecast_ci(df_base: pd.DataFrame,
                cfg: Config,
                inp: Inputs) -> pd.DataFrame:
    """
    Monte Carlo uncertainty on capacity WITHOUT new hires.
    Attrition is simulated as Binomial(employees_t, hazard_t).
    """
    km = compute_km(inp.durations, inp.events, max_months=120)
    from new_km import compute_seasonal_index
    seas = compute_seasonal_index(inp.exits_by_month, inp.active_by_month)

    T = cfg.horizon_months
    rng = np.random.default_rng(cfg.mc_seed)

    draws = cfg.mc_draws
    caps = np.zeros((draws, T), dtype=float)

    for d in range(draws):
        employees = float(inp.start_active_employees)
        for t in range(T):
            year, month = month_index_to_calendar(cfg.start_year, cfg.start_month, t)
            h = km.hazard(t) * float(seas.get(month, 1.0))
            h = float(np.clip(h, 0.0, 1.0))
            # binomial exits
            exits = rng.binomial(max(int(round(employees)), 0), h) if employees > 0 else 0
            employees = max(employees - exits, 0.0)
            caps[d, t] = employees * cfg.cap_per_employee * cfg.availability_factor

    mean = caps.mean(axis=0)
    low = np.quantile(caps, 0.025, axis=0)
    high = np.quantile(caps, 0.975, axis=0)

    out = pd.DataFrame({
        "Maand": df_base["Maand"],
        "Cap_mean": mean,
        "Cap_low": low,
        "Cap_high": high
    })
    return out

# ------------------------- Orchestration -------------------------

def run_pipeline(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int,int]]:
    inp = load_inputs(cfg)
    base = build_base_forecast(cfg, inp)
    # auto plan using base (no hires)
    plan = auto_hire_plan(base, cfg.ramp, cfg.buffer, strategy="earliest", max_pm=cfg.max_hires_per_month)
    final = apply_plan(base, plan, cfg.ramp)
    final = recompute_metrics(final, cfg.buffer)
    ci = forecast_ci(base, cfg, inp)
    return final, ci, plan
