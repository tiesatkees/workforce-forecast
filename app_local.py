# app_local.py — Run locally with Gradio, upload your own files
# Usage:
#   pip install -r requirements.txt
#   python app_local.py
#
# Then open the local URL Gradio prints (usually http://127.0.0.1:7860)

from __future__ import annotations
import io
from datetime import date, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe in most environments
import matplotlib.pyplot as plt
import gradio as gr
import traceback

from new_settings import Config, default_config
from new_forecast import run_pipeline

def _parse_date(s: str, fallback: date) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return fallback

def run_with_uploads(kdb_file, founders_file, active_file, demand_file,
                     as_of, start, horizon, buffer_val,
                     cap_per_emp, availability, ramp_csv, max_pm, strategy):
    """
    Runs the pipeline using uploaded files. Handles both CSV and Excel.
    """
    try:
        cfg = default_config(date.today())

        # assign uploaded paths (Gradio supplies a tempfile with .name)
        cfg.kdb_file = kdb_file.name if kdb_file else None
        cfg.founders_file = founders_file.name if founders_file else None
        cfg.active_file = active_file.name if active_file else None
        cfg.demand_file = demand_file.name if demand_file else None

        if not cfg.kdb_file and not cfg.founders_file:
            return None, None, "Upload minstens één historisch roster (KdB of Founders)."

        if not cfg.demand_file:
            return None, None, "Upload een demand-bestand met kolom 'vraag'."

        cfg.as_of = _parse_date(as_of, date.today())
        if start:
            y, m = start.split("-")
            cfg.start_year, cfg.start_month = int(y), int(m)

        if horizon: cfg.horizon_months = int(horizon)
        if buffer_val is not None: cfg.buffer = int(buffer_val)
        if cap_per_emp: cfg.cap_per_employee = float(cap_per_emp)
        if availability: cfg.availability_factor = float(availability)
        if ramp_csv:
            cfg.ramp = [float(x.strip()) for x in str(ramp_csv).split(",") if str(x).strip()]
        if max_pm: cfg.max_hires_per_month = int(max_pm)

        # Run pipeline
        final_df, ci_df, plan = run_pipeline(cfg)

        # Apply chosen strategy by re-running only the plan if needed
        # (run_pipeline uses 'earliest' by default; strategy switch done by recomputing plan in new_forecast if desired)
        # For now we keep default; if user wants another strategy, they can change in new_forecast.py

        # Plot
        fig = plt.figure()
        x = np.arange(len(final_df))
        if ci_df is not None and {"Cap_low","Cap_high","Cap_mean"}.issubset(ci_df.columns):
            plt.fill_between(x, ci_df["Cap_low"], ci_df["Cap_high"], alpha=0.25, label="Cap 95% CI")
            plt.plot(x, ci_df["Cap_mean"], label="Cap (mean)")
        plt.plot(x, final_df["Cap"].values, label="Cap (with hires)")
        plt.plot(x, (final_df["Vraag"].values + cfg.buffer), label="Vraag + buffer")
        plt.legend()
        plt.title("Capacity vs Demand")
        plt.xlabel("Month")
        plt.ylabel("Customers / month")
        bio = io.BytesIO()
        plt.tight_layout()
        fig.savefig(bio, format="png", dpi=160)
        bio.seek(0)

        # Tidy plan for display
        plan_str = "{\n" + "\n".join(f"  {k}: {v}" for k,v in sorted(plan.items())) + "\n}"

        return final_df, bio, f"✅ Plan\n{plan_str}"

    except Exception as e:
        return None, None, f"❌ Error:\n\n{traceback.format_exc()}"

with gr.Blocks(title="Workforce Forecast (Local)") as demo:
    gr.Markdown("# Workforce Forecast (Local)")
    gr.Markdown("Upload je bestanden (CSV of Excel). Demand moet een kolom **'vraag'** hebben.")

    with gr.Row():
        kdb = gr.File(label="KdB (CSV/XLSX)", file_types=[".xlsx",".xls",".csv"])
        founders = gr.File(label="Founders (CSV/XLSX)", file_types=[".xlsx",".xls",".csv"])
        active = gr.File(label="Actief (CSV/XLSX)", file_types=[".xlsx",".xls",".csv"])
        demand = gr.File(label="Demand (kolom 'vraag', CSV/XLSX)", file_types=[".xlsx",".xls",".csv"])

    with gr.Row():
        as_of = gr.Textbox(label="As-of (YYYY-MM-DD)", value=str(date.today()))
        start = gr.Textbox(label="Forecast start (YYYY-MM)", value="")
        horizon = gr.Number(label="Horizon (maanden)", value=12, precision=0)
        buffer_val = gr.Number(label="Buffer (customers)", value=0, precision=0)

    with gr.Row():
        cap_per_emp = gr.Number(label="Cap per medewerker (customers/mnd)", value=60.0)
        availability = gr.Number(label="Beschikbaarheidsfactor (0-1)", value=0.95)
        ramp_csv = gr.Textbox(label="Ramp CSV", value="0,0,0,50,80,100,120,140,140")
        max_pm = gr.Number(label="Max hires/maand", value=3, precision=0)
        strategy = gr.Dropdown(label="Hire-strategie (default in code)", choices=["earliest","latest","min+shift"], value="earliest")

    run_btn = gr.Button("Run")
    out_df = gr.Dataframe(label="Forecast", interactive=False)
    out_plot = gr.Image(label="Plot")
    out_log = gr.Textbox(label="Plan / Logs", lines=12)

    run_btn.click(
        run_with_uploads,
        [kdb, founders, active, demand, as_of, start, horizon, buffer_val, cap_per_emp, availability, ramp_csv, max_pm, strategy],
        [out_df, out_plot, out_log],
    )

if __name__ == "__main__":
    demo.launch()
