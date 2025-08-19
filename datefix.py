#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 15:11:14 2025

@author: tiesvandenheuvel
"""

# datefix.py
"""
Robuuste hulpfuncties voor datums (voorkomt None vs Timestamp vergelijkingen).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# veelvoorkomende kolomnamen (case-insensitive)
START_ALIASES = ["in dienst", "start", "startdatum", "start date", "datum in dienst"]
END_ALIASES   = ["uit dienst", "einde", "einddatum", "end", "end date", "datum uit dienst"]

def find_col(cols, aliases):
    """Zoek de eerste kolomnaam die matcht met een alias (case-insensitive)."""
    cl = [str(c).lower().strip() for c in cols]
    for a in aliases:
        if a in cl:
            return cols[cl.index(a)]
    return None

def to_dt(series: pd.Series) -> pd.Series:
    """Forceer datetime64[ns]; lege waarden (None/“”) worden NaT."""
    if series is None:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0), dtype="datetime64[ns]")
    s = series.copy()
    s = s.replace("", np.nan)
    return pd.to_datetime(s, errors="coerce")

def ensure_timestamp(x) -> pd.Timestamp:
    """Zet elke date/datetime/str veilig om naar pd.Timestamp (nooit None)."""
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.Timestamp(x)
    except Exception:
        return pd.Timestamp("today")

def get_active_mask(df: pd.DataFrame, as_of,
                    start_aliases=None, end_aliases=None):
    """
    Retourneert (a_start, a_end, as_of_ts, mask) waarbij:
    mask = (a_start <= as_of_ts) & (a_end.isna() | (a_end > as_of_ts))
    """
    start_aliases = start_aliases or START_ALIASES
    end_aliases   = end_aliases or END_ALIASES

    start_col = find_col(df.columns, start_aliases)
    if start_col is None:
        raise ValueError("Kon geen startkolom vinden (probeer: In dienst/Start/Startdatum).")

    end_col = find_col(df.columns, end_aliases)

    a_start = to_dt(df[start_col])
    a_end   = to_dt(df[end_col]) if (end_col is not None and end_col in df.columns) \
             else pd.Series(pd.NaT, index=df.index)

    as_of_ts = ensure_timestamp(as_of)
    mask = (a_start <= as_of_ts) & (a_end.isna() | (a_end > as_of_ts))
    return a_start, a_end, as_of_ts, mask
