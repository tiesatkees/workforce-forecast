# app_streamlit.py — Streamlit UI die je Gradio-app 1-op-1 nabouwt
from __future__ import annotations
import io, os, importlib, tempfile
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---- JOUW PIPELINE-MODULES (zelfde als Gradio) ----
import paden
import forecast_full as fc          # bevat forecast() & forecast_ci()
try:
    from hire_planner import auto_hire_plan
    HAS_AUTO = True
except Exception:
    HAS_AUTO = False

# ---------- helpers ------------------------------------------------------
def _tmp_copy(up_file) -> Path:
    """Sla upload op als tijdelijk bestand en geef pad terug (zoals in Gradio)."""
    if up_file is None:
        return None
    suffix = Path(up_file.name).suffix
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(up_file.read()); f.flush(); f.close()
    return Path(f.name)

def _parse_csv_list(txt: str) -> list[float]:
    if not txt or not txt.strip():
        return []
    out = []
    for s in txt.split(","):
        s = s.strip()
        if not s: continue
        try:
            out.append(float(s))
        except Exception:
            pass
    return out

def _parse_manual_plan(txt: str) -> dict[int, int]:
    """
    '2:2, 3:1' -> {2:2, 3:1}
    maand-index 0 = eerste forecastmaand
    """
    plan = {}
    if not txt or not txt.strip():
        return plan
    for seg in txt.split(","):
        seg = seg.strip()
        if ":" not in seg: 
            continue
        t, k = seg.split(":", 1)
        try:
            t_i = int(t.strip()); k_i = int(k.strip())
            if t_i >= 0 and k_i > 0:
                plan[t_i] = plan.get(t_i, 0) + k_i
        except Exception:
            continue
    return plan

def _set_dates(mode: str, today_str: str, start_str: str):
    """Exact zelfde logica als in hire_gradio_app.py."""
    today = pd.to_datetime(today_str)
    if mode.startswith("Gebruik"):
        paden.FORECAST_GENERATION_DATE = today
        paden.FORECAST_START_DT = (today + pd.offsets.MonthBegin()).normalize()
        paden.CONTRACT_BUFFER_MAANDEN = 1
    else:
        start = pd.to_datetime(start_str)
        paden.FORECAST_GENERATION_DATE = start - pd.offsets.MonthBegin(1)
        paden.FORECAST_START_DT = start
        paden.CONTRACT_BUFFER_MAANDEN = 0

def _patch_paths(kdb_p: Path, fd_p: Path, act_p: Path,
                 buffer_val: int, ramp_list: list[float], vraag_path: Path|None):
    """Paden + parameters in modules zetten en forecast_full herladen (bouwt KM + SEAS)."""
    paden.KDB_FILE      = kdb_p
    paden.FOUNDERS_FILE = fd_p
    paden.ACTIVE_FILE   = act_p
    paden.KLANTVRAAG_PROGNOSE = vraag_path  # je forecast_full leest dit net als in Gradio
    fc.BUFFER = int(buffer_val)
    fc.RAMP   = ramp_list
    importlib.reload(fc)  # triggert _build_km() → verse Kaplan–Meier + SEAS

# ---------- UI -----------------------------------------------------------
st.set_page_config(page_title="Workforce Forecast (Streamlit)", layout="wide")
st.title("Upload data & kies opties")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Historisch KdB.xlsx**")
    kdb_up = st.file_uploader("CompanyEmployeeHoursSalary_KeesdeBoekhouder_Office_BV.xlsx",
                              type=["xlsx","xls"], key="kdb")
    st.markdown("**Actief personeel.xlsx**")
    act_up = st.file_uploader("Huidige data.xlsx", type=["xlsx","xls"], key="act")
with c2:
    st.markdown("**Historisch Founders.xlsx**")
    fd_up = st.file_uploader("CompanyEmployeeHoursSalary_Founders_Finance_BV.xlsx",
                             type=["xlsx","xls"], key="fd")
    st.markdown("**Klantforecast (csv/xlsx)**")
    vr_up = st.file_uploader("klantenvoorspelling.xlsx / .csv",
                             type=["xlsx","xls","csv"], key="vr")

st.markdown("---")
st.subheader("Forecast-modus")
mode = st.radio(
    "",
    ["Gebruik 'huidige datum' + buffer", "Gebruik expliciete start-datum (geen buffer)"],
    index=0, horizontal=True
)

cols = st.columns(2)
with cols[0]:
    today_in = st.text_input("Huidige datum", value=str(date.today()))
with cols[1]:
    start_in = st.text_input("Start-datum forecast", value=str(date.today()))

st.subheader("Hire-strategie")
strat = st.radio("", ["earliest", "latest", "min+shift"], index=2, horizontal=True)

auto_chk = st.checkbox("Automatisch hire-plan", value=False, disabled=not HAS_AUTO)
hire_txt = st.text_input("Handmatig plan (bijv. 5:2,8:1)", value="2:2, 3:1, 4:1, 5:1")

vraag_txt = st.text_input("Handmatige vraag (1500,1520,…)", value="")  # zoals in Gradio
buffer_nb = st.number_input("Buffer (+klanten)", min_value=0, value=100, step=10)

ramp_txt = st.text_input("Ramp-up CSV", value="0,0,0,50,80,100,120,140,140")
run_btn = st.button("Run forecast", type="primary")

out_df_slot = st.empty()
out_plan_slot = st.empty()
out_plot_slot = st.empty()

# ---------- actie ---------------------------------------------------------
if run_btn:
    try:
        if not (kdb_up and fd_up and act_up):
            st.error("Upload minimaal de drie Excel-bestanden.")
            st.stop()

        # save uploads
        kdb = _tmp_copy(kdb_up)
        fd  = _tmp_copy(fd_up)
        act = _tmp_copy(act_up)
        vr_p = _tmp_copy(vr_up) if vr_up else None

        # parse ramp & vraag
        ramp = _parse_csv_list(ramp_txt)
        if not ramp:
            ramp = [0,0,0,50,80,100,120,140,140]

        # set dates + paths (zoals Gradio)
        _set_dates(mode, today_in, start_in)
        _patch_paths(kdb, fd, act, int(buffer_nb), ramp, vr_p)

        # basis-forecast zonder hires (voor auto-planner input)
        base_df = fc.forecast({})   # plan leeg
        # auto plan?
        if auto_chk:
            if not HAS_AUTO:
                st.error("Automatisch plan niet beschikbaar (hire_planner.py ontbreekt).")
                st.stop()
            try:
                plan = auto_hire_plan(base_df, ramp, buffer=int(buffer_nb), strategy=strat)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
        else:
            plan = _parse_manual_plan(hire_txt)

        # finale forecast
        df = fc.forecast(plan)
        ci = fc.forecast_ci(plan)

        # Voeg hires-kolom toe (zoals Gradio)
        if "Hires" not in df.columns:
            df["Hires"] = 0
        for idx, n in plan.items():      # plan: {maand-index: aantal}
            if idx in df.index:
                df.at[idx, "Hires"] = n

        # Orden kolommen
        front = [c for c in ["Maand", "Vraag", "Cap", "Tekort", "Hires", "OK"] if c in df]
        df = df[front + [c for c in df.columns if c not in front]]

        # toon tabel
        out_df_slot.subheader("Forecast")
        out_df_slot.dataframe(df, use_container_width=True)

        # toon plan (JSON)
        out_plan_slot.subheader("Hire plan (maand-index → hires)")
        out_plan_slot.json(plan)

        # Fan-chart plot (zoals Gradio)
        x = pd.to_datetime(ci["Maand"])
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, ci["Cap_mean"], lw=2, label="verwacht")
        ax.fill_between(x, ci["Cap_low"], ci["Cap_high"], alpha=.25, label="95 % CI")
        if "Vraag" in df.columns and "OK" in df.columns:
            # Vraag + buffer-lijn zoals in Gradio
            # Vraag zit al in df; fc.BUFFER is toegepast in 'Tekort/OK' logic
            # Voor visuele consistentie tonen we Vraag + BUFFER
            try:
                vraag_plus = df["Vraag"].to_numpy(dtype=float) + float(fc.BUFFER or 0)
                ax.plot(pd.to_datetime(df["Maand"]), vraag_plus, ls="--", label="vraag+buffer")
            except Exception:
                pass
        ax.set_title("Fan-chart"); ax.legend()
        out_plot_slot.pyplot(fig)

        st.success("✅ Forecast afgerond")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
