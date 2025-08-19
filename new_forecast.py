# new_forecast.py
"""
Robuuste forecast-pipeline:
- Baseline capaciteit uit 'Huidige data' (per-medewerker, per-maand kolommen).
- Herkent NL-maandnamen ('september', ...) en datum-achtige kolommen (YYYY-MM(/-DD)).
- Actieve medewerkers: Status in {Actief, In dienst, Proeftijd} => 'Einde' -> NaT (doorlopend).
- Alle datumvergelijkingen veilig (geen None vs Timestamp).
- Handmatig hire-plan met absolute ramp-CSV.
- Buffer + vraag-vergelijking.
- Eenvoudige CI (±5%) voor fan-chart.

Afhankelijkheden: pandas, numpy, openpyxl (voor xlsx).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# Datum-helpers
# =========================

START_ALIASES = ["in dienst", "start", "startdatum", "start date", "datum in dienst"]
END_ALIASES   = ["uit dienst", "einde", "einddatum", "end", "end date", "datum uit dienst"]

NL_MONTH = {
    1: "januari", 2: "februari", 3: "maart", 4: "april", 5: "mei", 6: "juni",
    7: "juli", 8: "augustus", 9: "september", 10: "oktober", 11: "november", 12: "december"
}
ACTIVE_STATUSES = {"actief", "in dienst", "proeftijd"}  # case-insensitive


def _find_col(cols, aliases):
    cl = [str(c).lower().strip() for c in cols]
    for a in aliases:
        if a in cl:
            return cols[cl.index(a)]
    return None


def _to_dt(series: pd.Series) -> pd.Series:
    """Forceer datetime64[ns]; lege strings/None worden NaT."""
    if series is None:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0), dtype="datetime64[ns]")
    s = series.copy()
    s = s.replace("", np.nan)
    return pd.to_datetime(s, errors="coerce")


def _ensure_ts(x) -> pd.Timestamp:
    """Altijd een pd.Timestamp teruggeven (nooit None)."""
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.Timestamp(x)
    except Exception:
        return pd.Timestamp("today")


# =========================
# IO helpers (xlsx/csv)
# =========================

def _read_table(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = str(path).lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


# =========================
# Tijd-as helpers
# =========================

def _month_starts(start_year: int, start_month: int, horizon: int) -> List[pd.Timestamp]:
    start = pd.Timestamp(year=int(start_year), month=int(start_month), day=1)
    return [start + pd.offsets.MonthBegin(n) for n in range(horizon)]

def _month_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


# =========================
# Huidige data → baseline capaciteit
# =========================

def _detect_date_like_month_cols(df: pd.DataFrame) -> Dict[pd.Period, str]:
    """
    Datum-achtige kolomnamen (YYYY-MM of YYYY-MM-DD) -> map naar Period('YYYY-MM').
    """
    out: Dict[pd.Period, str] = {}
    for c in df.columns:
        cs = str(c).strip()
        try:
            ts = pd.to_datetime(cs, errors="raise")
            per = pd.Period(ts, freq="M")
            out[per] = c
        except Exception:
            pass
    return out


def _baseline_capacity_from_active(
    active_df: pd.DataFrame,
    months: List[pd.Timestamp],
    as_of_ts: pd.Timestamp
) -> np.ndarray:
    """
    Som van per-medewerker maandcapaciteiten uit 'Huidige data', per forecast-maand.
    Ondersteunt:
      - NL-maandkolommen ('september', 'oktober', …)
      - Datum-achtige kolommen ('2025-09', '2025-09-01', …)
    Actieve medewerker: (start <= maand-einde) & (end isna of end > maand-start),
    waarbij voor Status in {Actief, In dienst, Proeftijd} 'Einde' als NaT geldt.
    """
    H = len(months)
    if active_df.empty:
        return np.zeros(H, dtype=float)

    cols_lower = {str(c).lower().strip(): c for c in active_df.columns}

    # Start/Einde/Status veilig maken
    start_col = _find_col(active_df.columns, START_ALIASES)
    end_col   = _find_col(active_df.columns, END_ALIASES)
    status_col = None
    for cand in ["status", "state", "employment status"]:
        if cand in cols_lower:
            status_col = cols_lower[cand]
            break

    a_start = _to_dt(active_df[start_col]) if start_col else pd.Series(pd.Timestamp("1900-01-01"), index=active_df.index)
    a_end   = _to_dt(active_df[end_col]) if end_col else pd.Series(pd.NaT, index=active_df.index, dtype="datetime64[ns]")

    if status_col:
        is_active_status = active_df[status_col].astype(str).str.strip().str.lower().isin(ACTIVE_STATUSES)
        # Bij actieve status: treat Einde als NaT (doorlopend actief)
        a_end = a_end.mask(is_active_status, pd.NaT)

    # 1) Datum-achtige maandkolommen
    date_like_map = _detect_date_like_month_cols(active_df)
    # 2) NL-maandkolommen (case-insensitive)
    nl_cols = {name: cols_lower[name] for name in NL_MONTH.values() if name in cols_lower}

    cap = np.zeros(H, dtype=float)

    for i, m in enumerate(months):
        m_start = pd.Timestamp(year=m.year, month=m.month, day=1)
        m_end = m_start + pd.offsets.MonthEnd(0)
        active_mask = (a_start <= m_end) & (a_end.isna() | (a_end > m_start))

        # Eerst proberen datum-achtige kolom exact voor deze maand
        per = pd.Period(m, freq="M")
        col = date_like_map.get(per)

        if col is None:
            # Anders NL-maandnaam (los van jaar)
            nl_name = NL_MONTH[m.month]
            col = nl_cols.get(nl_name)

        if col is None:
            # Geen kolom voor deze maand -> 0
            cap[i] = 0.0
        else:
            vals = pd.to_numeric(active_df.loc[active_mask, col], errors="coerce").fillna(0.0)
            cap[i] = float(vals.sum())

    return cap


# =========================
# Hires & ramp
# =========================

def _parse_manual_plan(plan_text: str) -> Dict[int, int]:
    """
    Parseer "2:2, 3:1" → {2:2, 3:1}
    Betekenis: in maand-offset t (0=eerste forecast-maand) hire N mensen.
    """
    plan: Dict[int, int] = {}
    if not plan_text:
        return plan
    chunks = [s.strip() for s in plan_text.split(",") if s.strip()]
    for ch in chunks:
        if ":" not in ch:
            continue
        t_str, k_str = ch.split(":", 1)
        try:
            t = int(t_str.strip()); k = int(k_str.strip())
            if t < 0 or k <= 0:
                continue
            plan[t] = plan.get(t, 0) + k
        except Exception:
            continue
    return plan


def _hire_capacity_from_plan(plan: Dict[int, int],
                             ramp: List[float],
                             horizon: int) -> np.ndarray:
    """
    Extra capaciteit door hires met absolute ramp (per hire per maand).
    ramp: lijst absolute capaciteiten, bijv. [0,0,0,50,80,100,120,140,140]
    """
    if horizon <= 0:
        return np.zeros(0, dtype=float)
    ramp_arr = np.asarray(ramp, dtype=float) if (ramp and len(ramp) > 0) else np.zeros(0, dtype=float)
    out = np.zeros(horizon, dtype=float)
    if ramp_arr.size == 0:
        return out
    for t0, n_hires in plan.items():
        if t0 >= horizon or n_hires <= 0:
            continue
        for j, step in enumerate(ramp_arr):
            idx = t0 + j
            if idx >= horizon:
                break
            out[idx] += n_hires * float(step)
    return out


# =========================
# Vraag (demand)
# =========================

def _load_demand(cfg, months: List[pd.Timestamp], manual_demand_text: Optional[str]) -> np.ndarray:
    """
    Laad vraagvector van lengte horizon.
    Prioriteit:
      1) manual_demand_text: "1500,1520,..." (neemt eerste horizon waarden)
      2) demand_file: kolom 'vraag' (case-insensitive) of eerste numerieke kolom
      3) fallback: nullen
    """
    H = len(months)
    # 1) handmatig
    if manual_demand_text and manual_demand_text.strip():
        arr = []
        for tok in manual_demand_text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                arr.append(float(tok))
            except Exception:
                arr.append(np.nan)
        v = pd.Series(arr, dtype="float64").fillna(method="ffill").fillna(0.0).to_numpy()
        if len(v) >= H:
            return v[:H]
        out = np.zeros(H, dtype=float)
        out[:len(v)] = v
        if len(v) > 0:
            out[len(v):] = v[-1]
        return out

    # 2) uit bestand
    df = _read_table(getattr(cfg, "demand_file", None))
    if df.empty:
        return np.zeros(H, dtype=float)

    # Zoek 'vraag' kolom (case-insensitive)
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    vraag_col = cols_lower.get("vraag", None)
    vec = None
    if vraag_col is not None:
        vec = pd.to_numeric(df[vraag_col], errors="coerce")
    else:
        # neem eerste numerieke kolom
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 1:
                vec = s
                break
    if vec is None:
        return np.zeros(H, dtype=float)
    v = vec.fillna(method="ffill").fillna(0.0).to_numpy()

    if len(v) >= H:
        return v[:H]
    out = np.zeros(H, dtype=float)
    out[:len(v)] = v
    if len(v) > 0:
        out[len(v):] = v[-1]
    return out


# =========================
# Hoofdfuncties
# =========================

@dataclass
class Inputs:
    months: List[pd.Timestamp]
    active_df: pd.DataFrame
    baseline_cap: np.ndarray
    demand: np.ndarray


def load_inputs(cfg, manual_demand_text: Optional[str] = None) -> Inputs:
    """
    Leest alle benodigde inputs en zet ze om naar een consistente structuur.
    - Baseline capaciteit uit 'Huidige data' (NL-maandnamen of datum-achtige kolommen).
    - Actieve medewerkers: Status 'Actief/In dienst/Proeftijd' ⇒ 'Einde' wordt genegeerd.
    - Vraagvector met lengte horizon.
    """
    as_of_ts = _ensure_ts(getattr(cfg, "as_of", pd.Timestamp("today")))
    start_y  = int(getattr(cfg, "start_year", as_of_ts.year))
    start_m  = int(getattr(cfg, "start_month", as_of_ts.month))
    H        = int(getattr(cfg, "horizon_months", 12))

    months = _month_starts(start_y, start_m, H)

    # Huidige data (actief personeel)
    a = _read_table(getattr(cfg, "active_file", None))
    baseline = _baseline_capacity_from_active(a, months, as_of_ts)

    # Vraag
    demand = _load_demand(cfg, months, manual_demand_text)

    return Inputs(months=months, active_df=a, baseline_cap=baseline, demand=demand)


def run_pipeline(cfg,
                 manual_plan_text: str = "",
                 use_manual: bool = False,
                 strategy: str = "earliest",
                 manual_demand_text: str = ""
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], str]:
    """
    Output:
      - final_df  : 'Maand', 'Vraag', 'Cap', 'Tekort', 'Hires', 'OK'
      - ci_df     : 'Cap_low', 'Cap_mean', 'Cap_high' (±5%)
      - plan_dict : mapping YYYY-MM -> hires (alleen handmatig plan hier)
      - warn_msg  : waarschuwingstekst (string)
    """
    buffer_val = int(getattr(cfg, "buffer", 0))
    ramp_list  = list(getattr(cfg, "ramp", [0,0,0,50,80,100,120,140,140]))
    H          = int(getattr(cfg, "horizon_months", 12))

    # 1) Inputs
    inp = load_inputs(cfg, manual_demand_text=manual_demand_text)
    months   = inp.months
    baseline = inp.baseline_cap.copy()
    demand   = inp.demand.copy()

    # 2) Hires (handmatig)
    plan = {}
    cap = baseline.copy()
    if use_manual and manual_plan_text.strip():
        plan_idx = _parse_manual_plan(manual_plan_text)
        add = _hire_capacity_from_plan(plan_idx, ramp_list, H)
        cap = cap + add
        plan = { _month_label(months[t]): int(k) for t, k in plan_idx.items() if 0 <= t < H and k > 0 }

    # 3) Resultaten
    maand_labels = [_month_label(m) for m in months]
    vraag_buf = demand + float(buffer_val)
    tekort = np.maximum(0.0, vraag_buf - cap)
    hires_series = np.zeros(H, dtype=int)
    if plan:
        for i, m in enumerate(maand_labels):
            hires_series[i] = int(plan.get(m, 0))

    final = pd.DataFrame({
        "Maand": maand_labels,
        "Vraag": np.round(demand, 1),
        "Cap":   np.round(cap, 1),
        "Tekort": np.round(tekort, 1),
        "Hires": hires_series,
        "OK": (cap >= vraag_buf)
    })

    # 4) Simpele CI
    ci = pd.DataFrame({
        "Cap_low":  cap * 0.95,
        "Cap_mean": cap,
        "Cap_high": cap * 1.05
    })

    warn = ""
    return final, ci, plan, warn
