# forecast_full.py
"""
Zelfstandige forecast-module zonder 'data_loader' afhankelijkheid.

- Leest paden/instellingen uit 'paden.py'
- Bouwt bij import:
    * Kaplan–Meier survivorfunctie uit historiek (lifelines)
    * eenvoudige seizoensindex
- Baseline capaciteit uit 'Huidige data' (NL-maandnamen en datum-achtige kolommen)
- Past KM-survival toe per maand-offset (verwachte blijftijd/retentie)
- Neemt hires + RAMP (absolute cap per hire) mee
- BUFFER bij vraagvergelijking
- API:
    forecast(plan: dict[int,int]) -> pd.DataFrame
    forecast_ci(plan: dict[int,int]) -> pd.DataFrame
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# -------- instellingen uit paden.py (moeten gezet zijn vóór import/reload) ----------
import paden

# -------- externe dependency voor KM ----------
try:
    from lifelines import KaplanMeierFitter
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'lifelines'. Voeg 'lifelines' toe aan requirements.txt. "
        f"Original: {e}"
    )

# =========================
# Globale parameters (kunnen door UI worden overschreven voor reload)
# =========================

# BUFFER: extra klanten bovenop vraag waarmee je vergelijkt
BUFFER: int = getattr(paden, "BUFFER", 0)

# RAMP: absolute capaciteitsbijdrage per hire per maand (vanaf hire-maand)
# voorbeeld: [0,0,0,50,80,100,120,140,140]
RAMP: List[float] = getattr(paden, "RAMP", [0,0,0,50,80,100,120,140,140])

# Horizon in maanden (wordt uit start/eind afgeleid)
DEFAULT_HORIZON: int = getattr(paden, "DEFAULT_HORIZON", 12)

# Sheet-naam voor historisch tabblad (optioneel)
SHEET_HISTORISCH: Optional[str] = getattr(paden, "SHEET_HISTORISCH", None)

# =========================
# Helpers
# =========================

START_ALIASES = ["in dienst", "start", "startdatum", "start date", "datum in dienst"]
END_ALIASES   = ["uit dienst", "einde", "einddatum", "end", "end date", "datum uit dienst"]
STATUS_ALIASES = ["status", "state", "employment status"]
ACTIVE_STATUSES = {"actief", "in dienst", "proeftijd"}

NL_MONTH = {
    1: "januari", 2: "februari", 3: "maart", 4: "april", 5: "mei", 6: "juni",
    7: "juli", 8: "augustus", 9: "september", 10: "oktober", 11: "november", 12: "december"
}

def _find_col(cols, aliases):
    cl = [str(c).lower().strip() for c in cols]
    for a in aliases:
        if a in cl:
            return cols[cl.index(a)]
    return None

def _to_dt(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0), dtype="datetime64[ns]")
    s = series.copy()
    s = s.replace("", np.nan)
    return pd.to_datetime(s, errors="coerce")

def _ensure_ts(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.Timestamp(x)
    except Exception:
        return pd.Timestamp("today")

def _read_table(path: Optional[str|pd.PathLike], sheet: Optional[str]=None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = str(path).lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    # excel
    kw = {}
    if sheet:
        kw["sheet_name"] = sheet
    return pd.read_excel(path, **kw)

def _month_starts(start_dt: pd.Timestamp, horizon: int) -> List[pd.Timestamp]:
    start = pd.Timestamp(year=int(start_dt.year), month=int(start_dt.month), day=1)
    return [start + pd.offsets.MonthBegin(n) for n in range(horizon)]

def _month_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")

def _detect_date_like_month_cols(df: pd.DataFrame) -> Dict[pd.Period, str]:
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

def _active_mask_for_month(a_start: pd.Series, a_end: pd.Series,
                           m_start: pd.Timestamp) -> pd.Series:
    m_end = m_start + pd.offsets.MonthEnd(0)
    return (a_start <= m_end) & (a_end.isna() | (a_end > m_start))

# =========================
# 1) Kaplan–Meier uit historiek
# =========================

def _build_km() -> Dict[str, np.ndarray]:
    """
    Bouwt een kaplan-meier survivorfunctie op maandresolutie uit historische data.
    Verwacht in paden:
      - KDB_FILE, FOUNDERS_FILE (historische tabellen)
      - SHEET_HISTORISCH (optioneel)
    Zoekt start/eind kolommen robuust.
    Return dict met:
      - "surv": survival-probabilities per maandoffset (np.ndarray)
      - "horizon": int
    """
    frames = []
    for path in [getattr(paden, "KDB_FILE", None), getattr(paden, "FOUNDERS_FILE", None)]:
        if not path: 
            continue
        df = _read_table(path, sheet=SHEET_HISTORISCH)
        if df.empty: 
            continue
        frames.append(df)
    if not frames:
        # Geen historiek -> geen uitval; survival=1
        return {"surv": np.ones(DEFAULT_HORIZON, dtype=float), "horizon": DEFAULT_HORIZON}

    hist = pd.concat(frames, ignore_index=True)
    start_col = _find_col(hist.columns, START_ALIASES)
    end_col   = _find_col(hist.columns, END_ALIASES)
    if start_col is None:
        # Geen start -> neem ver verleden
        a_start = pd.Series(pd.Timestamp("1900-01-01"), index=hist.index)
    else:
        a_start = _to_dt(hist[start_col])
    a_end = _to_dt(hist[end_col]) if (end_col and end_col in hist.columns) else pd.Series(pd.NaT, index=hist.index)

    # durations in maanden (Als geen end -> censored)
    end_eff = a_end.fillna(pd.Timestamp("today"))
    durations_m = ((end_eff - a_start).dt.days.clip(lower=0) / 30.4375).astype(float)
    events = (~a_end.isna()).astype(int)

    # Fit KM
    km = KaplanMeierFitter()
    km.fit(durations_m, event_observed=events)

    # Maak een nette vector per maandoffset tot MAX_H
    max_months = int(np.nanmax(np.ceil(durations_m.fillna(0).to_numpy()))) if len(durations_m) else DEFAULT_HORIZON
    H = max(DEFAULT_HORIZON, min(max(12, max_months), 120))  # cap op 10 jaar
    t = np.arange(0, H + 1)
    # km.survival_function_ is op onregelmatige t; interp op maandgrid (forward-fill)
    sf = km.survival_function_.reindex(km.survival_function_.index.union(t)).interpolate(method="ffill").reindex(t).iloc[:,0].to_numpy()
    # gebruik t=0..H-1 (horizon maanden; cap voor de zekerheid)
    return {"surv": sf[:H], "horizon": H}

_KM_CACHE = _build_km()

# =========================
# 2) Seizoensindex (eenvoudig)
# =========================

def _build_seasonality() -> Dict[int, float]:
    """
    Eenvoudige seizoensindex per maand (1..12).
    Als geen data: allemaal 1.0
    (Je kunt dit vervangen door jouw eigen methode; interface blijft gelijk.)
    """
    try:
        # heuristiek: gebruik KDB/Founders aantallen per maand om index te maken
        frames = []
        for path in [getattr(paden, "KDB_FILE", None), getattr(paden, "FOUNDERS_FILE", None)]:
            if not path:
                continue
            df = _read_table(path, sheet=SHEET_HISTORISCH)
            if df.empty:
                continue
            frames.append(df)
        if not frames:
            return {m: 1.0 for m in range(1, 13)}

        hist = pd.concat(frames, ignore_index=True)
        # tel entries per maand van 'In dienst'
        start_col = _find_col(hist.columns, START_ALIASES)
        if start_col is None:
            return {m: 1.0 for m in range(1, 13)}
        s = _to_dt(hist[start_col]).dropna()
        bym = s.dt.month.value_counts().sort_index()
        idx = {int(m): float(bym.get(m, 0.0)) for m in range(1,13)}
        mean = np.mean(list(idx.values())) or 1.0
        return {m: (v/mean if mean else 1.0) for m, v in idx.items()}
    except Exception:
        return {m: 1.0 for m in range(1, 13)}

_SEAS = _build_seasonality()

# =========================
# 3) Actieve baseline capaciteit uit Huidige data
# =========================

def _baseline_from_active(active_df: pd.DataFrame, months: List[pd.Timestamp]) -> np.ndarray:
    """
    Som per maand over actieve medewerkers van hun maandcapaciteitskolommen.
    Herkent:
      - datum-achtige kolommen (YYYY-MM, YYYY-MM-DD)
      - NL-maandnamen ('september', ...)
    Behandelt 'Status' in {Actief, In dienst, Proeftijd} alsof 'Einde' NaT is.
    """
    H = len(months)
    if active_df.empty:
        return np.zeros(H, dtype=float)

    cols_lower = {str(c).lower().strip(): c for c in active_df.columns}

    start_col = _find_col(active_df.columns, START_ALIASES)
    end_col   = _find_col(active_df.columns, END_ALIASES)
    status_col = _find_col(active_df.columns, STATUS_ALIASES)

    a_start = _to_dt(active_df[start_col]) if start_col else pd.Series(pd.Timestamp("1900-01-01"), index=active_df.index)
    a_end   = _to_dt(active_df[end_col]) if end_col else pd.Series(pd.NaT, index=active_df.index, dtype="datetime64[ns]")

    if status_col:
        is_active = active_df[status_col].astype(str).str.strip().str.lower().isin(ACTIVE_STATUSES)
        a_end = a_end.mask(is_active, pd.NaT)

    date_like = _detect_date_like_month_cols(active_df)
    nl_cols = {name: cols_lower[name] for name in NL_MONTH.values() if name in cols_lower}

    out = np.zeros(H, dtype=float)
    for i, m in enumerate(months):
        mask = _active_mask_for_month(a_start, a_end, m)
        per = pd.Period(m, freq="M")
        col = date_like.get(per)
        if col is None:
            col = nl_cols.get(NL_MONTH[m.month])
        if col is None:
            continue
        vals = pd.to_numeric(active_df.loc[mask, col], errors="coerce").fillna(0.0)
        out[i] = float(vals.sum())
    return out

# =========================
# 4) Vraag laden
# =========================

def _load_vraag(months: List[pd.Timestamp]) -> np.ndarray:
    """
    Laad vraagvector:
    - Als paden.KLANTVRAAG_PROGNOSE is gezet: neem kolom 'vraag' (case-insensitive) of eerste numerieke kolom.
    - Als paden.MANUAL_VRAAG bestaat en een csv-string bevat: gebruik die.
    - Anders: nullen.
    """
    H = len(months)

    # manual string?
    manual = getattr(paden, "MANUAL_VRAAG", "")
    if isinstance(manual, str) and manual.strip():
        arr = []
        for tok in manual.split(","):
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

    # file?
    path = getattr(paden, "KLANTVRAAG_PROGNOSE", None)
    df = _read_table(path)
    if df.empty:
        return np.zeros(H, dtype=float)
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    vraag_col = cols_lower.get("vraag", None)
    vec = None
    if vraag_col:
        vec = pd.to_numeric(df[vraag_col], errors="coerce")
    else:
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
# 5) Hires → extra capaciteit via RAMP
# =========================

def _hire_cap_from_plan(plan: Dict[int,int], ramp: List[float], H: int) -> np.ndarray:
    out = np.zeros(H, dtype=float)
    ramp_arr = np.asarray(ramp, dtype=float) if (ramp and len(ramp)>0) else np.zeros(0)
    if ramp_arr.size == 0:
        return out
    for t0, n in plan.items():
        if t0 < 0 or t0 >= H or n <= 0:
            continue
        for j, step in enumerate(ramp_arr):
            idx = t0 + j
            if idx >= H:
                break
            out[idx] += n * float(step)
    return out

# =========================
# 6) Seizoen + KM toepassen
# =========================

def _apply_survival_and_season(baseline_cap: np.ndarray, months: List[pd.Timestamp]) -> np.ndarray:
    """
    Past KM-survival toe op baseline per maand-offset, en daarna seizoensindex.
    Baseline_cap: capaciteit zonder uitval.
    Return: gecorrigeerde capaciteit.
    """
    H = len(months)
    surv = _KM_CACHE.get("surv", np.ones(max(H,1)))
    if len(surv) < H:
        # verleng laatste waarde
        tail = np.full(H - len(surv), fill_value=float(surv[-1]) if len(surv)>0 else 1.0)
        surv = np.concatenate([surv, tail])
    # Corrigeer baseline met KM retentie per offset
    cap = baseline_cap * surv[:H]
    # Seizoensindex
    seas = np.array([_SEAS.get(m.month, 1.0) for m in months], dtype=float)
    cap = cap * seas
    return cap

# =========================
# 7) Publieke API
# =========================

def forecast(plan: Dict[int,int]) -> pd.DataFrame:
    """
    Bouw de forecast tabel met plan (maand-index -> hires).
    Kolommen: Maand, Vraag, Cap, Tekort, OK
    """
    # datums
    start_dt = _ensure_ts(getattr(paden, "FORECAST_START_DT", pd.Timestamp("today") + pd.offsets.MonthBegin()))
    horizon  = int(getattr(paden, "HORIZON", DEFAULT_HORIZON))
    months = _month_starts(start_dt, horizon)
    maand_labels = [_month_label(m) for m in months]

    # inlezen active
    active_df = _read_table(getattr(paden, "ACTIVE_FILE", None))
    baseline = _baseline_from_active(active_df, months)

    # survival + seizoen
    cap0 = _apply_survival_and_season(baseline, months)

    # hires
    add = _hire_cap_from_plan(plan, RAMP, horizon)
    cap = cap0 + add

    # vraag + buffer
    vraag = _load_vraag(months)
    vraag_buf = vraag + float(BUFFER or 0)

    tekort = np.maximum(0.0, vraag_buf - cap)
    ok = cap >= vraag_buf

    df = pd.DataFrame({
        "Maand": maand_labels,
        "Vraag": np.round(vraag, 1),
        "Cap":   np.round(cap, 1),
        "Tekort": np.round(tekort, 1),
        "OK": ok
    })
    return df

def forecast_ci(plan: Dict[int,int]) -> pd.DataFrame:
    """
    Eenvoudige CI rondom Cap (95%). Hier +/-10% band (kan je verfijnen naar KM-variant).
    Kolommen: Maand, Cap_low, Cap_mean, Cap_high
    """
    start_dt = _ensure_ts(getattr(paden, "FORECAST_START_DT", pd.Timestamp("today") + pd.offsets.MonthBegin()))
    horizon  = int(getattr(paden, "HORIZON", DEFAULT_HORIZON))
    months = _month_starts(start_dt, horizon)
    maand_labels = [_month_label(m) for m in months]

    # Recompute cap (zoals in forecast)
    active_df = _read_table(getattr(paden, "ACTIVE_FILE", None))
    baseline = _baseline_from_active(active_df, months)
    cap0 = _apply_survival_and_season(baseline, months)
    add = _hire_cap_from_plan(plan, RAMP, horizon)
    cap = cap0 + add

    # CI band
    low  = cap * 0.90
    high = cap * 1.10
    out = pd.DataFrame({
        "Maand": maand_labels,
        "Cap_low": low,
        "Cap_mean": cap,
        "Cap_high": high
    })
    return out
