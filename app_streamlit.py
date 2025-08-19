import io
from datetime import date, datetime
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from new_settings import Config, default_config
from new_forecast import run_pipeline  # of: new_forecast_manual

st.set_page_config(page_title="Workforce Forecast", layout="wide")
st.title("Workforce Forecast (Streamlit)")

def _parse_date(s): return datetime.strptime(s, "%Y-%m-%d").date()

col_files = st.columns(4)
with col_files[0]: kdb = st.file_uploader("KdB.xlsx/.csv", type=["xlsx","xls","csv"])
with col_files[1]: founders = st.file_uploader("Founders.xlsx/.csv", type=["xlsx","xls","csv"])
with col_files[2]: active = st.file_uploader("Actief.xlsx/.csv", type=["xlsx","xls","csv"])
with col_files[3]: demand = st.file_uploader("Demand (kolom 'vraag')", type=["xlsx","xls","csv"])

col_opts = st.columns(6)
as_of = col_opts[0].text_input("As-of (YYYY-MM-DD)", value=str(date.today()))
start  = col_opts[1].text_input("Start (YYYY-MM)", value="")
horizon = col_opts[2].number_input("Horizon (m)", 1, 60, 12)
buffer_val = col_opts[3].number_input("Buffer (+klanten)", 0, 9999, 0)
ramp_csv = col_opts[4].text_input("Ramp CSV", "0,0,0,50,80,100,120,140,140")
max_pm = col_opts[5].number_input("Max hires/maand", 0, 50, 3)

if st.button("Run"):
    try:
        cfg = default_config(date.today())
        # Schrijf uploads naar tijdelijke files zodat pipeline ze kan lezen
        def save(f): 
            if not f: return None
            import tempfile, os
            suffix = "." + f.name.split(".")[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(f.read()); tmp.flush(); tmp.close()
            return tmp.name

        cfg.kdb_file      = save(kdb)
        cfg.founders_file = save(founders)
        cfg.active_file   = save(active)
        cfg.demand_file   = save(demand)

        if as_of: cfg.as_of = _parse_date(as_of)
        if start:
            y,m = start.split("-"); cfg.start_year, cfg.start_month = int(y), int(m)
        cfg.horizon_months = int(horizon)
        cfg.buffer = int(buffer_val)
        cfg.ramp = [float(x.strip()) for x in ramp_csv.split(",") if x.strip()]
        cfg.max_hires_per_month = int(max_pm)

        final, ci, plan = run_pipeline(cfg)

        st.subheader("Forecast (eerste 12 rijen)")
        st.dataframe(final.head(12))
        st.subheader("Hire plan")
        st.json(plan)

        fig = plt.figure()
        x = np.arange(len(final))
        if {"Cap_low","Cap_high","Cap_mean"}.issubset(ci.columns):
            plt.fill_between(x, ci["Cap_low"], ci["Cap_high"], alpha=0.25, label="Cap 95% CI")
            plt.plot(x, ci["Cap_mean"], label="Cap (mean)")
        plt.plot(x, final["Cap"].values, label="Cap (with hires)")
        plt.plot(x, final["Vraag"].values + cfg.buffer, label="Vraag+buffer")
        plt.legend(); plt.title("Capacity vs Demand"); plt.xlabel("Month"); plt.ylabel("Customers / month")
        st.pyplot(fig)

        csv = final.to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
