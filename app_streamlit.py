# app_streamlit.py — Streamlit UI die je Gradio-scherm nabouwt (NL)
# Vereist: streamlit, pandas, numpy, matplotlib, openpyxl
# Pipeline: forecast_full.run_pipeline(...) of new_forecast.run_pipeline(...)

from __future__ import annotations
import io
import tempfile
from datetime import date, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----- Pipeline import (pas aan als jouw namen anders zijn) -----
# Probeer eerst forecast_full (zoals je uploadde); val terug op new_forecast.
try:
    from new_settings import default_config, Config  # jouw config
except Exception:
    # Fallback: maak een minimale dummy zodat de app niet crasht tijdens import
    from dataclasses import dataclass, field
    @dataclass
    class Config:
        as_of: date = date.today()
        start_year: int = date.today().year
        start_month: int = date.today().month
        horizon_months: int = 12
        buffer: int = 0
        ramp: list[float] = field(default_factory=lambda: [0,0,0,50,80,100,120,140,140])
        max_hires_per_month: int = 3
        kdb_file: str | None = None
        founders_file: str | None = None
        active_file: str | None = None
        demand_file: str | None = None
    def default_config(_): return Config()

# kies jouw pipeline
run_pipeline = None
_err = None
try:
    from forecast_full import run_pipeline as _rp
    run_pipeline = _rp
except Exception as e:
    _err = e
    try:
        from new_forecast import run_pipeline as _rp2
        run_pipeline = _rp2
        _err = None
    except Exception as e2:
        _err = e2

# ---------------- Helpers ----------------
def _save_upload(f) -> str | None:
    """Bewaar een upload tijdelijk en return het pad (voor je pipeline)."""
    if not f:
        return None
    suffix = ""
    if "." in f.name:
        suffix = "." + f.name.split(".")[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.read()); tmp.flush(); tmp.close()
    return tmp.name

def _parse_date_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _parse_ym(s: str) -> tuple[int,int] | None:
    try:
        y, m = s.split("-")
        return int(y), int(m)
    except Exception:
        return None

def _try_run(cfg: Config,
             manual_plan_text: str | None,
             use_auto: bool,
             strategy: str,
             manual_demand_text: str | None):
    """
    Roept je run_pipeline aan met verschillende veelvoorkomende signaturen.
    Past zich aan als je forecast_full/new_forecast net even anders heet.
    Verwacht return: (final_df, ci_df, plan) of (final_df, ci_df, plan, waarschuwing)
    """
    if run_pipeline is None:
        raise RuntimeError(f"Kon run_pipeline niet importeren: {_err}")

    # Probeer varianten (zonder/ met handmatig plan)
    # 1) forecast_full style (handmatig plan + strategie + handmatige vraag)
    try:
        return run_pipeline(cfg,
                            manual_plan_text=manual_plan_text or "",
                            use_manual=(not use_auto and bool(manual_plan_text)),
                            strategy=strategy,
                            manual_demand_text=(manual_demand_text or "").strip())
    except TypeError:
        pass
    # 2) new_forecast style (geen handmatige vraag)
    try:
        return run_pipeline(cfg,
                            manual_plan_text=manual_plan_text or "",
                            use_manual=(not use_auto and bool(manual_plan_text)),
                            strategy=strategy)
    except TypeError:
        pass
    # 3) kale variant (alleen cfg)
    return run_pipeline(cfg)

# ---------------- UI ----------------
st.set_page_config(page_title="Workforce Forecast", layout="wide")
st.title("📊 Workforce Forecast (Streamlit)")

left, right = st.columns(2)

with left:
    st.markdown("**Actief personeel.xlsx**")
    active_up = st.file_uploader("Huidige data (.xlsx/.csv)", type=["xlsx","xls","csv"], key="active")

with right:
    st.markdown("**Klantforecast (csv/xlsx)**")
    demand_up = st.file_uploader("klantenvoorspelling (.xlsx/.csv)", type=["xlsx","xls","csv"], key="demand")

st.markdown("---")
st.subheader("Forecast-modus")

mode = st.radio(
    "Forecast-modus",
    ["Gebruik 'huidige datum' + buffer", "Gebruik expliciete start-datum (geen buffer)"],
    index=0,
    horizontal=True
)

col_dt1, col_dt2 = st.columns(2)
with col_dt1:
    as_of_str = st.text_input("Huidige datum", value=str(date.today()))
with col_dt2:
    start_str = st.text_input("Start-datum forecast", value=str(date.today()))

st.subheader("Hire-strategie")
strategy = st.radio("Hire-strategie", ["earliest","latest","min+shift"], index=2, horizontal=True)

auto = st.checkbox("Automatisch hire-plan", value=False)
manual_plan = st.text_input("Handmatig plan (bijv. 5:2,8:1)", value="2:2, 3:1, 4:1, 5:1")

manual_demand = st.text_input("Handmatige vraag (1500,1520,…)", value="")

buffer_val = st.number_input("Buffer (+klanten)", min_value=0, value=100, step=10)

ramp_csv = st.text_input("Ramp-up CSV", value="0,0,0,50,80,100,120,140,140")

run = st.button("Run forecast", type="primary")

# ---------------- RUN ----------------
if run:
    try:
        # Config vullen
        cfg = default_config(date.today())
        cfg.kdb_file = None           # (optioneel upload; niet in screenshot)
        cfg.founders_file = None      # (optioneel upload; niet in screenshot)
        cfg.active_file = _save_upload(active_up)
        cfg.demand_file = _save_upload(demand_up)

        # Datums
        cfg.as_of = _parse_date_ymd(as_of_str.strip()) if as_of_str else date.today()
        ym = _parse_ym(start_str.strip())
        if ym:
            cfg.start_year, cfg.start_month = ym
        # Horizon uit je default; pas aan als je een input wil
        # Buffer logica per modus
        if mode.startswith("Gebruik 'huidige datum'"):
            cfg.buffer = int(buffer_val or 0)
        else:
            cfg.buffer = 0  # expliciete start → geen buffer

        # Ramp & max hires
        cfg.ramp = [float(x.strip()) for x in ramp_csv.split(",") if x.strip()]
        # max hires gebruik je alleen bij auto-plan; laat default staan
        # cfg.max_hires_per_month = 3

        # Run de pipeline (met/zonder manual)
        final, ci, plan, *maybe_warn = _try_run(
            cfg=cfg,
            manual_plan_text=(manual_plan if not auto else ""),
            use_auto=auto,
            strategy=strategy,
            manual_demand_text=(manual_demand if manual_demand.strip() else None)
        )

        warn = maybe_warn[0] if maybe_warn else ""
        if warn:
            st.info(warn)

        st.success("✅ Forecast afgerond")
        # ------ Output tabel(len) ------
        st.subheader("Forecast")
        st.dataframe(final, use_container_width=True)

        st.subheader("Hire plan")
        st.json(plan)

        # ------ Fan-chart / plot ------
        if isinstance(ci, pd.DataFrame) and {"Cap_low","Cap_high","Cap_mean"}.issubset(ci.columns):
            fig, ax = plt.subplots(figsize=(8,4))
            x = np.arange(len(ci))
            ax.fill_between(x, ci["Cap_low"], ci["Cap_high"], alpha=0.25, label="Cap 95% CI")
            ax.plot(x, ci["Cap_mean"], label="Cap (mean)")
            # Zet vraag + buffer en gerealiseerde Cap indien aanwezig
            if "Cap" in final.columns:
                ax.plot(x, final["Cap"].values, label="Cap (with hires)")
            if "Vraag" in final.columns:
                ax.plot(x, final["Vraag"].values + cfg.buffer, label="Vraag + buffer")
            ax.set_title("Fan-chart")
            ax.set_xlabel("Maand-index")
            ax.set_ylabel("Klanten / maand")
            ax.legend()
            st.pyplot(fig)
        else:
            # Simpele plot als er geen CI is
            if "Cap" in final.columns or "Vraag" in final.columns:
                fig, ax = plt.subplots(figsize=(8,4))
                x = np.arange(len(final))
                if "Cap" in final.columns:
                    ax.plot(x, final["Cap"].values, label="Cap")
                if "Vraag" in final.columns:
                    ax.plot(x, final["Vraag"].values + cfg.buffer, label="Vraag + buffer")
                ax.legend(); ax.set_title("Capacity vs Demand"); ax.set_xlabel("Maand-index"); ax.set_ylabel("Klanten / maand")
                st.pyplot(fig)

        # ------ Download knop ------
        csv_bytes = final.to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", data=csv_bytes, file_name="forecast.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
