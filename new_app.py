# new_app.py
"""
Optional Gradio UI for the NEW pipeline, with CLI fallback.
You can run:
  python new_app.py --help
  python new_app.py --cli --config my_config.json
  python new_app.py   (to launch Gradio if installed)
"""

from __future__ import annotations
import argparse, json, sys, os
import traceback
from datetime import date, datetime
import numpy as np
import pandas as pd

from new_settings import Config, default_config
from new_forecast import run_pipeline

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def run_cli(args):
    cfg = default_config(date.today())
    # override from args / json
    if args.config:
        with open(args.config, "r") as f:
            data = json.load(f)
        cfg = Config(**{
            **default_config(date.today()).__dict__,
            **data
        })

    # simple overrides
    for k in ["kdb_file","founders_file","active_file","demand_file"]:
        v = getattr(args, k)
        if v: setattr(cfg, k, v)
    if args.as_of: cfg.as_of = _parse_date(args.as_of)
    if args.start:
        y, m = args.start.split("-")
        cfg.start_year, cfg.start_month = int(y), int(m)
    if args.horizon: cfg.horizon_months = int(args.horizon)
    if args.buffer is not None: cfg.buffer = int(args.buffer)

    final, ci, plan = run_pipeline(cfg)
    print("=== PLAN ===")
    print(plan)
    print("\n=== FORECAST (head) ===")
    print(final.head(12).to_string(index=False))
    out_csv = "forecast_new_pipeline.csv"
    final.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

def run_gradio():
    try:
        import gradio as gr
    except Exception as e:
        print("Gradio is not installed. Use --cli mode or install gradio.", file=sys.stderr)
        sys.exit(1)

    def do_run(kdb, founders, active, demand, as_of, start, horizon, buffer, cap_per_emp, availability, ramp_csv, max_pm):
        try:
            cfg = default_config(date.today())
            cfg.kdb_file = kdb.name if kdb else None
            cfg.founders_file = founders.name if founders else None
            cfg.active_file = active.name if active else None
            cfg.demand_file = demand.name if demand else None

            cfg.as_of = _parse_date(as_of) if as_of else date.today()
            if start:
                y, m = start.split("-")
                cfg.start_year, cfg.start_month = int(y), int(m)
            if horizon: cfg.horizon_months = int(horizon)
            if buffer is not None: cfg.buffer = int(buffer)
            if cap_per_emp: cfg.cap_per_employee = float(cap_per_emp)
            if availability: cfg.availability_factor = float(availability)
            if ramp_csv:
                cfg.ramp = [float(x.strip()) for x in ramp_csv.split(",") if x.strip()]
            if max_pm: cfg.max_hires_per_month = int(max_pm)

            final, ci, plan = run_pipeline(cfg)

            # Build plot
            import matplotlib.pyplot as plt
            import io, base64

            fig = plt.figure()
            x = np.arange(len(final))
            plt.fill_between(x, ci["Cap_low"], ci["Cap_high"], alpha=0.25, label="Cap 95% CI")
            plt.plot(x, ci["Cap_mean"], label="Cap (mean)")
            plt.plot(x, final["Cap"].values, label="Cap (with hires)")
            plt.plot(x, final["Vraag"].values + cfg.buffer, label="Vraag+buffer")
            plt.legend()
            plt.title("Capacity vs Demand")
            plt.xlabel("Month")
            plt.ylabel("Customers / month")
            bio = io.BytesIO()
            plt.tight_layout()
            fig.savefig(bio, format="png", dpi=160)
            bio.seek(0)

            return final, plan, gr.Image.update(value=bio)

        except Exception as e:
            tb = traceback.format_exc()
            return pd.DataFrame(), {}, gr.update(value=None), f"Error: {e}\n\n{tb}"

    with gr.Blocks(title="NEW Workforce Forecast") as demo:
        gr.Markdown("# NEW Workforce Forecast (KM + Planner)")

        with gr.Row():
            kdb = gr.File(label="KdB.xlsx / .csv", file_types=[".xlsx",".xls",".csv"])
            founders = gr.File(label="Founders.xlsx / .csv", file_types=[".xlsx",".xls",".csv"])
            active = gr.File(label="Actief personeel.xlsx / .csv", file_types=[".xlsx",".xls",".csv"])
            demand = gr.File(label="Klantvraag (kolom 'vraag')", file_types=[".xlsx",".xls",".csv"])

        with gr.Row():
            as_of = gr.Textbox(label="As-of (YYYY-MM-DD)", value=str(date.today()))
            start = gr.Textbox(label="Forecast start (YYYY-MM)", value="")
            horizon = gr.Number(label="Horizon (maanden)", value=12, precision=0)
            buffer = gr.Number(label="Buffer (customers)", value=0, precision=0)

        with gr.Row():
            cap_per_emp = gr.Number(label="Cap per medewerker (customers/mnd)", value=60.0)
            availability = gr.Number(label="Beschikbaarheidsfactor (0-1)", value=0.95)
            ramp_csv = gr.Textbox(label="Ramp CSV", value="0,0,0,50,80,100,120,140,140")
            max_pm = gr.Number(label="Max hires per maand", value=3, precision=0)

        run_btn = gr.Button("Run")
        out_df = gr.Dataframe(label="Forecast", interactive=False)
        out_plan = gr.JSON(label="Hire plan")
        out_plot = gr.Image(label="Plot")
        out_err = gr.Textbox(label="Errors", value="", lines=4)

        def wrapped_run(*args):
            df, plan, plot, err = do_run(*args)
            return df, json.dumps(plan, indent=2), plot, (err or "")

        run_btn.click(
            wrapped_run,
            [kdb, founders, active, demand, as_of, start, horizon, buffer, cap_per_emp, availability, ramp_csv, max_pm],
            [out_df, out_plan, out_plot, out_err]
        )

    demo.launch()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true", help="Run in CLI mode")
    ap.add_argument("--config", type=str, help="Path to JSON config")
    ap.add_argument("--kdb_file", type=str)
    ap.add_argument("--founders_file", type=str)
    ap.add_argument("--active_file", type=str)
    ap.add_argument("--demand_file", type=str)
    ap.add_argument("--as_of", type=str, help="YYYY-MM-DD")
    ap.add_argument("--start", type=str, help="YYYY-MM")
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--buffer", type=int)

    args = ap.parse_args()

    if args.cli:
        run_cli(args)
    else:
        run_gradio()
