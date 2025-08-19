#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:15:06 2025

@author: tiesvandenheuvel
"""

#!/usr/bin/env python3
# forecast_full.py  – geen recency-weging, dynamische seizoensfactor
from __future__ import annotations
import logging, importlib
from pathlib import Path
import pandas as pd, numpy as np
from lifelines import KaplanMeierFitter
from dateutil.relativedelta import relativedelta

import paden                                   # ← config (wordt steeds gepatcht)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
def _seasonal_index(kdb_path: Path, founders_path: Path) -> dict[int,float]:
    """Tel exits per kalender-maand in beide bestanden → normaliseer naar index."""
    START, END, COST = 'In dienst', 'Uit dienst', 'Kostenplaatsen'
    sel = {kdb_path: 'Operations | Bookies KdB',
           founders_path: 'Operations | Bookies Founders'}

    exits = pd.Series(0, index=range(1,13), dtype=int)
    for path, flt in sel.items():
        df = pd.read_excel(path, sheet_name=paden.SHEET_HISTORISCH)
        df = df[df[COST] == flt]
        end = pd.to_datetime(df[END], errors='coerce')
        exits += end.dropna().dt.month.value_counts().reindex(range(1,13), fill_value=0)

    mean = exits.mean() or 1          # fallback 1 als geen exits
    return {m: round(c/mean,3) for m,c in exits.items()}

# ──────────────────────────────────────────────────────────────────────────────
def _build_km() -> KaplanMeierFitter:
    """KM-model ZONDER gewichten; berekent gelijk nieuwe seizoensfactor."""
    START, END, COST = 'In dienst', 'Uit dienst', 'Kostenplaatsen'
    FLT = {paden.KDB_FILE: 'Operations | Bookies KdB',
           paden.FOUNDERS_FILE: 'Operations | Bookies Founders'}

    parts=[]
    for path, flt in FLT.items():
        d = pd.read_excel(path, sheet_name=paden.SHEET_HISTORISCH)
        parts.append(d[d[COST]==flt][[START, END]])
    df = pd.concat(parts, ignore_index=True)

    as_of = paden.FORECAST_GENERATION_DATE
    df['start_dt'] = pd.to_datetime(df[START], errors='coerce')
    df['end_dt']   = pd.to_datetime(df[END],   errors='coerce').fillna(as_of)
    df = df[df['start_dt'].notna()].copy()
    df['dur_m']=(df['end_dt']-df['start_dt']).dt.days/30.4375
    df['event']=(df['end_dt']!=as_of).astype(int)
    df = df[df['dur_m']>=0]

    km = KaplanMeierFitter()
    km.fit(df['dur_m'], df['event'])
    log.info("KM: %s rec, %s events", len(df), df['event'].sum())

    # ── dynamische seizoensfactor
    global SEAS
    SEAS = _seasonal_index(paden.KDB_FILE, paden.FOUNDERS_FILE)
    log.info("Seizoensindex: %s", SEAS)

    return km

# ── Cache KM + Greenwood-var (ongewijzigd verder) ────────────────────────────
_KM_CACHE = _build_km()
_S_VAR    = _KM_CACHE.variance if hasattr(_KM_CACHE,"variance") else (
    ((_KM_CACHE.confidence_interval_survival_function_["KM_estimate_upper_0.95"]
     -_KM_CACHE.confidence_interval_survival_function_["KM_estimate_lower_0.95"])
     /(2*1.96))**2)

def S(t_m:float)->float: return float(_KM_CACHE.predict(t_m).clip(0,1))
def month_hazard(t_m:float)->float:
    s0,s1=S(t_m),S(t_m+1);  return (s0-s1)/s0 if s0>1e-12 else 0.0

# ── BUFFER, RAMP komen via paden/gradio patch; functies forecast()/forecast_ci
#     blijven ongewijzigd en gebruiken nu de nieuwe globale SEAS.
#     (code ingekort voor leesbaarheid)
from forecast_full_core import *   # ← hier staat jouw bestaande forecast() logic
