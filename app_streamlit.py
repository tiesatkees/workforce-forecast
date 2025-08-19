# app_streamlit.py — Streamlit UI die je Gradio-pipeline 1-op-1 gebruikt
from __future__ import annotations
import importlib, tempfile
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Laad alléén paden-module meteen; de rest pas ná het zetten van paden.
import paden

# ---------- helpers ----------
def _tmp_copy(up_file) -> Path | None:
    if up_file is None:
        return None
    p = Path(up_file.name)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=p.suffix)
    f.write(up_file.read()); f.flush(); f.close()
    return Path(f.name)

def _parse_csv_list(txt: str) -> list[float]:
    if not txt or not txt.strip():
        return []
    out = []
    for s in txt.split(","):
        s = s.strip()
        if not s: 
            continue
        try:
            out.append(float(s))
        except:
            pass
    return out

def _parse_manual_plan(txt: str) -> dict[int, int]:
    # "2:2, 3:1" -> {2:2, 3:1}
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
        except:
            continue
    return plan

def _set_dates(mode: str, today_str: str, start_str: str):
    """Exacte moduslogica zoals in je Gradio-app."""
    today = pd.to_datetime(today_str)
    if mode.startswith("Gebruik 'huidige datum'"):
        paden.FORECAST_GENERATION_DATE = today
        paden.FORECAST_START_DT = (today + pd.offsets.MonthBegin()).normalize()
        paden.CONTRACT_BUFFER_MAANDEN = 1
    else:
        start = pd.to_datetime(start_str)
        paden.FORECAST_GENERATION_DATE = start - pd.offsets.MonthBegin(1)
        paden.FORECAST_START_DT = start
        paden.CONTRACT_BUFFER_MAANDEN = 0

def _patch_paths(kdb: Path, fd: Path, act: Path,
                 vraag_path: Path | None,
                 manual_vraag_text: str | None):
    """Stel alles in wat forecast_full en data_loader bij import nodig hebben."""
    paden.KDB_FILE      = kdb
    paden.FOUNDERS_FILE = fd
    paden.ACTIVE_FILE   = act
    paden.KLANTVRAAG_PROGNOSE = vraag_path
    # optioneel: handmatige vraagstring zoals in Gradio
    paden.MANUAL_VRAAG = (manual_vraag_text or "").strip()

# ---------- UI ----------
st.set_page_config(page_title="Workforce Forecast", layout="wide")
st.title("Upload data & kies opties")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Historisch KdB.xlsx**")
    kdb_up = st.file_uploader("CompanyEmployeeHoursSalary_KeesdeBoekhouder_Office_BV.xlsx",
                              type=["xlsx","xls"], key="kdb", label_visibility="visible")
    st.markdown("**Actief personeel.xlsx**")
    act_up = st.file_uploader("Huidige data.xlsx", type=["xlsx","xls"], key="act", label_visibility="visible")
with c2:
    st.markdown("**Historisch Founders.xlsx**")
    fd_up = st.file_uploader("CompanyEmployeeHoursSalary_Founders_Finance_BV.xlsx",
                             type=["xlsx","xls"], key="fd", label_visibility="visible")
    st.markdown("**Klantforecast (csv/xlsx)**")
    vr_up = st.file_uploader("klantenvoorspelling.xlsx / .csv",
                             type=["xlsx","xls","csv"], key="vr", label_visibility="visible")

st.markdown("---")
st.subheader("Forecast-modus")
mode = st.radio("Forecast-modus",
                ["Gebruik 'huidige datum' + buffer",
                 "Gebruik expliciete start-datum (geen buffer)"],
                index=0, horizontal=True, label_visibility="collapsed")

cols = st.columns(2)
with cols[0]:
    today_in = st.text_input("Huidige datum (YYYY-MM-DD)", value=str(date.today()))
with cols[1]:
    start_in = st.text_input("Start-datum forecast (YYYY-MM-DD)", value=str(date.today()))

st.subheader("Hire-strategie")
strategy = st.radio("Hire-strategie",
                    ["earliest","latest","min+shift"],
                    index=2, horizontal=True, label_visibility="collapsed")

auto_chk = st.checkbox("Automatisch hire-plan gebruiken", value=False)
hire_txt = st.text_input("Handmatig plan (bijv. 2:2, 3:1, ...)", value="2:2, 3:1")

vraag_txt = st.text_input("Handmatige vraag (1500,1520,… — laat leeg als je een bestand uploadt)", value="")
buffer_nb = st.number_input("Buffer (+klanten)", min_value=0, value=100, step=10)

ramp_txt = st.text_input("Ramp-up CSV", value="0,0,0,50,80,100,120,140,140")
run_btn = st.button("Run forecast", type="primary")

out_df_slot = st.empty()
out_plan_slot = st.empty()
out_plot_slot = st.empty()

if run_btn:
    try:
        # 1) Controleer uploads
        if not (kdb_up and fd_up and act_up):
            st.error("Upload minimaal KdB, Founders en Actief personeel.")
            st.stop()

        # 2) Sla files op
        kdb = _tmp_copy(kdb_up)
        fd  = _tmp_copy(fd_up)
        act = _tmp_copy(act_up)
        vr_p = _tmp_copy(vr_up) if vr_up else None

        # 3) Stel datums & paden (zoals Gradio)
        _set_dates(mode, today_in, start_in)
        _patch_paths(kdb, fd, act, vr_p, vraag_txt)

        # 4) NU pas pipeline-modules importeren/refreshen (exacte volgorde als Gradio)
        dl  = importlib.import_module("data_loader")          # 1
        fcc = importlib.import_module("forecast_full_core")   # 2
        hp = None
        if auto_chk:
            try:
                hp = importlib.import_module("hire_planner")  # 3 (optioneel)
            except Exception:
                hp = None
        fc  = importlib.import_module("forecast_full")        # 4

        # 5) BUFFER en RAMP op module zetten en reloaden zodat KM/seasonality op paden is gebouwd
        fc.BUFFER = int(buffer_nb)
        fc.RAMP = _parse_csv_list(ramp_txt) or [0,0,0,50,80,100,120,140,140]
        importlib.reload(fc)  # bouw KM/seasonality (lifelines) opnieuw op basis van huidige paden + params

        # 6) Basisforecast zonder hires (zoals in Gradio)
        base_df = fc.forecast({})

        # 7) Plan bepalen
        if auto_chk and hp is not None and hasattr(hp, "auto_hire_plan"):
            plan = hp.auto_hire_plan(base_df, fc.RAMP, buffer=int(buffer_nb), strategy=strategy)
        else:
            plan = _parse_manual_plan(hire_txt)

        # 8) Finale forecast
        df = fc.forecast(plan)
        ci = fc.forecast_ci(plan)

        # 9) Hires-kolom toevoegen (zichtbaar in tabel)
        if "Hires" not in df.columns:
            df["Hires"] = 0
        for idx, n in plan.items():
            if idx in df.index:
                df.at[idx, "Hires"] = n

        # 10) Kolomvolgorde
        front = [c for c in ["Maand","Vraag","Cap","Tekort","Hires","OK"] if c in df]
        df = df[front + [c for c in df.columns if c not in front]]

        # 11) Output UI
        out_df_slot.subheader("Forecast")
        out_df_slot.dataframe(df, use_container_width=True)

        out_plan_slot.subheader("Hire plan (maand-index → hires)")
        out_plan_slot.json(plan)

        # Fan-chart
        x = pd.to_datetime(ci["Maand"])
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, ci["Cap_mean"], lw=2, label="verwacht")
        ax.fill_between(x, ci["Cap_low"], ci["Cap_high"], alpha=.25, label="95 % CI")
        try:
            vraag_plus = df["Vraag"].to_numpy(dtype=float) + float(fc.BUFFER or 0)
            ax.plot(pd.to_datetime(df["Maand"]), vraag_plus, ls="--", label="vraag+buffer")
        except Exception:
            pass
        ax.set_title("Fan-chart"); ax.legend()
        out_plot_slot.pyplot(fig)

        st.success("✅ Forecast afgerond")

    except FileNotFoundError as e:
        st.error(f"Bestand niet gevonden: {e}. Controleer dat je alle uploads hebt gedaan.")
    except ModuleNotFoundError as e:
        st.error(f"Dependency mist: {e}. Zorg dat benodigde modules in requirements.txt staan (bijv. 'lifelines', 'scipy').")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
