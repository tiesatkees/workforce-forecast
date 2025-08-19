#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:17:43 2025

@author: tiesvandenheuvel
"""

from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_full.py
----------------
• KM-model zonder recency-weging
• Nieuwe hires krijgen dezelfde hazard vanaf maand 0
• Buffer +100
• forecast(hire_plan)      → DataFrame per maand
• forecast_ci(hire_plan)   → DataFrame met mean, P2.5, P97.5
• run_demo()               → voorbeeld + fan-chart
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from dateutil.relativedelta import relativedelta

import paden
from data_loader import load_active_staff_and_capacity


# logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# forecast_full.py  (bovenaan, vlak na imports)
def _seasonal_index(kdb_path: Path, founders_path: Path) -> dict[int, float]:
    """Bereken seizoensfactor op basis van exits in beide bestanden."""
    START, END, COST = 'In dienst', 'Uit dienst', 'Kostenplaatsen'
    sel = {
        kdb_path:      'Operations | Bookies KdB',
        founders_path: 'Operations | Bookies Founders'
    }

    exits = pd.Series([0]*12, index=range(1,13), dtype=int)
    for path, flt in sel.items():
        df = pd.read_excel(path, sheet_name=paden.SHEET_HISTORISCH)
        df = df[df[COST] == flt]
        end = pd.to_datetime(df[END], errors='coerce')
        exits += end.dropna().dt.month.value_counts().reindex(range(1,13), fill_value=0)

    mean = exits.mean() if exits.sum() else 1
    return {m: round(c/mean, 3) for m,c in exits.items()}


# ───────────────────────────────────────────────────────────────
# 1  KM-MODEL (geen weging)  – cache eenmaal
def _build_km() -> KaplanMeierFitter:
    START, END, COST = 'In dienst', 'Uit dienst', 'Kostenplaatsen'
    FLT = {
        paden.KDB_FILE:      'Operations | Bookies KdB',
        paden.FOUNDERS_FILE: 'Operations | Bookies Founders'
    }
    parts=[]
    for path, val in FLT.items():
        d = pd.read_excel(path, sheet_name=paden.SHEET_HISTORISCH)
        parts.append(d[d[COST]==val][[START, END]])
    df = pd.concat(parts, ignore_index=True)

    as_of = paden.FORECAST_GENERATION_DATE
    df['start_dt']=pd.to_datetime(df[START], errors='coerce')
    df['end_dt']  =pd.to_datetime(df[END],  errors='coerce').fillna(as_of)
    df=df[df['start_dt'].notna()].copy()
    df['dur_m']=(df['end_dt']-df['start_dt']).dt.days/30.4375
    df['event']=(df['end_dt']!=as_of).astype(int)
    df=df[df['dur_m']>=0]
    km=KaplanMeierFitter()
    km.fit(df['dur_m'], df['event'])
    log.info("KM (no weight): %s rec, %s events", len(df), df['event'].sum())
    return km

_KM_CACHE=_build_km()

def S(t_m: float)->float:
    """Survival op t maanden (geclipt [0,1])."""
    return float(_KM_CACHE.predict(t_m).clip(0,1))


# ── BOVENAAN na _build_km() ──────────────────────────
def _greenwood_var(kmf: KaplanMeierFitter) -> pd.Series:
    """
    Retourneert Greenwood-variantie S(t)².
    Compatibel met oude en nieuwe lifelines-versies.
    """
    # 1) lifelines ≤0.26 had .variance
    if hasattr(kmf, "variance"):
        return kmf.variance
    # 2) lifelines ≥0.27: reconstrueer via CI
    ci = kmf.confidence_interval_survival_function_
    lo = ci["KM_estimate_lower_0.95"]
    hi = ci["KM_estimate_upper_0.95"]
    sigma = (hi - lo) / (2 * 1.96)
    return sigma.pow(2)


# Greenwood variantie (lifelines geeft t-index in jaren; we converteren)
_S_VAR = _greenwood_var(_KM_CACHE)
_S_VAR.index = _KM_CACHE.timeline # timeline staat al in maanden

def month_hazard(t_m: float)->float:
    """h_m = (S(t)-S(t+1))/S(t)  – deterministisch."""
    s0,s1=S(t_m), S(t_m+1)
    return (s0-s1)/s0 if s0>1e-12 else 0.

# ───────────────────────────────────────────────────────────────
# 2  PARAMETERS
SEAS = {m:c/(50/12) for m,c in {1:4,2:8,3:5,4:2,5:4,6:4,7:0,8:9,9:2,10:2,11:4,12:6}.items()}
RAMP=[0,0,0,50,80,100,120,140,140]
BUFFER=100

# ───────────────────────────────────────────────────────────────
def _inject_hires(staff: pd.DataFrame,
                  months: List[pd.Timestamp],
                  plan: Dict[int,int])->pd.DataFrame:
    rows=[]
    L=len(RAMP)
    labels=[m.strftime("%Y-%m") for m in months]
    for idx,n in plan.items():
        for k in range(n):
            start=months[idx]
            r={paden.COL_ID_ACTIEF:f"NEW{idx}_{k}",
               paden.COL_START_ACTIEF:start,
               "P_active":1.0}
            for j in range(len(months)):
                ramp=j-idx
                r[f"Cap_{labels[j]}"] = RAMP[ramp] if 0 <= ramp < L else 0
            rows.append(r)
    return pd.concat([staff, pd.DataFrame(rows)], ignore_index=True) if rows else staff

# ───────────────────────────────────────────────────────────────
def forecast(hire_plan:Dict[int,int]|None=None)->pd.DataFrame:
    """Deterministische forecast (mean)."""
    if hire_plan is None:
        hire_plan={}
    staff,_=load_active_staff_and_capacity()
    months=pd.date_range(paden.FORECAST_START_DT, periods=len(paden.KLANTVRAAG_PROGNOSE), freq="MS")
    labels=[m.strftime("%Y-%m") for m in months]
    staff=_inject_hires(staff,list(months),hire_plan)
    staff["P_active"]=1.0
    rows=[]
    for i,m in enumerate(months):
        season=SEAS[m.month]
        cap_tot,next_p=0.0,[]
        for _,r in staff.iterrows():
            p_start=r["P_active"]
            tenure=(m-r[paden.COL_START_ACTIEF]).days/30.4375
            if tenure>=0:
                h=month_hazard(tenure)*season
                p_end=p_start*(1-h)
            else:
                p_end=p_start
            cap=p_start * r[f"Cap_{labels[i]}"] 
            cap_tot+=cap
            next_p.append(p_end)
        staff["P_active"]=next_p
        demand=paden.KLANTVRAAG_PROGNOSE[i]+BUFFER
        rows.append(dict(Maand=labels[i],
                         Vraag=demand,
                         Cap=round(cap_tot,1),
                         Tekort=round(demand-cap_tot,1),
                         OK=(cap_tot>=demand)))
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────
# 3  Monte-Carlo 95 % CI
def _draw_S(t_m:float)->float:
    """Trek S~N(mu,var) met clipping op [0,1]."""
    mu=S(t_m)
    var=_S_VAR.get(t_m,0.0)
    s=np.random.normal(mu, np.sqrt(max(var,1e-12)))
    return float(np.clip(s,0,1))

def forecast_ci(hire_plan:Dict[int,int]|None=None, draws:int=800)->pd.DataFrame:
    """Mean, 2.5- and 97.5-percentiel van capaciteit."""
    months=pd.date_range(paden.FORECAST_START_DT, periods=len(paden.KLANTVRAAG_PROGNOSE), freq="MS")
    labels=[m.strftime("%Y-%m") for m in months]
    cap_matrix=np.zeros((draws, len(months)))
    staff0,_=load_active_staff_and_capacity()
    staff0=_inject_hires(staff0,list(months), hire_plan or {})
    for d in range(draws):
        staff=staff0.copy()
        staff["P_active"]=1.0
        for i,m in enumerate(months):
            season=SEAS[m.month]
            cap_tot,next_p=0.0,[]
            for _,r in staff.iterrows():
                p_start=r["P_active"]
                tenure=(m-r[paden.COL_START_ACTIEF]).days/30.4375
                if tenure>=0:
                    s0=_draw_S(tenure)
                    s1=_draw_S(tenure+1)
                    h=max(0,(s0-s1)/s0) * season if s0>0 else 0
                    p_end=p_start*(1-h)
                else:
                    p_end=p_start
                cap_tot += p_start * r[f"Cap_{labels[i]}"]
                next_p.append(p_end)
            staff["P_active"]=next_p
            cap_matrix[d,i]=cap_tot
    mean=cap_matrix.mean(axis=0)
    lo=np.percentile(cap_matrix,2.5,axis=0)
    hi=np.percentile(cap_matrix,97.5,axis=0)
    demand=np.array(paden.KLANTVRAAG_PROGNOSE)+BUFFER
    return pd.DataFrame(dict(
        Maand=labels, Vraag=demand,
        Cap_mean=mean.round(1),
        Cap_low=lo.round(1),
        Cap_high=hi.round(1)
    ))

# ───────────────────────────────────────────────────────────────
def run_demo():
    """Voorbeeld: 2 hires in december 2025."""
    plan={3:2}
    df=forecast(plan)
    print(df.to_string(index=False))

    ci=forecast_ci(plan, draws=800)
    print("\nFan-chart data (mean ±95%):")
    print(ci.to_string(index=False))

    # Fan-chart plot
    plt.figure(figsize=(9,4))
    plt.plot(ci['Maand'], ci['Cap_mean'], lw=2,label='verwacht')
    plt.fill_between(ci['Maand'], ci['Cap_low'], ci['Cap_high'],
                     color='tab:blue', alpha=.2,label='95% CI')
    plt.plot(ci['Maand'], ci['Vraag'], ls='--', color='red', label='vraag+buffer')
    plt.xticks(rotation=45); plt.ylabel('klanten')
    plt.tight_layout(); plt.legend(); plt.grid(alpha=.3)
    plt.show()

# ───────────────────────────────────────────────────────────────
if __name__=="__main__":
    run_demo()

