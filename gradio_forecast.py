#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:45:05 2025

@author: tiesvandenheuvel
"""

#!/usr/bin/env python3
# gradio_forecast.py  – 4 uploads + live forecast
import gradio as gr, pandas as pd, numpy as np, matplotlib.pyplot as plt
import tempfile, shutil, importlib, os
from pathlib import Path

import paden                          # jouw config
import forecast_full as fc            # bevat forecast() & forecast_ci()

# ───────────────── helpers ────────────────────────────────────
def _tmp_copy(upload) -> Path:
    """kopieer geüploade file naar tmp en return pad"""
    f = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix)
    shutil.copyfile(upload.name, f.name)
    return Path(f.name)

# ... bovenaan ongewijzigd ...
def _patch_paths(kdb, founders, active, buffer, ramp, vraag):
    paden.KDB_FILE      = kdb
    paden.FOUNDERS_FILE = founders
    paden.ACTIVE_FILE   = active
    paden.KLANTVRAAG_PROGNOSE = vraag
    fc.BUFFER = buffer
    fc.RAMP   = ramp
    importlib.reload(fc)      # _build_km() berekent SEAS opnieuw


# ───────────────── main compute fn ────────────────────────────
def run_forecast(kdb_file, founders_file, active_file, vraag_file,
                 hire_plan_txt, buffer_val, ramp_csv):

    # 0) guards
    if not all([kdb_file, founders_file, active_file, vraag_file]):
        return "Upload alle vier de bestanden", None

    # 1) tmp-kopieën
    kdb_p      = _tmp_copy(kdb_file)
    founders_p = _tmp_copy(founders_file)
    active_p   = _tmp_copy(active_file)
    vraag_p    = _tmp_copy(vraag_file)

    # 2) klantforecast inlezen
    if vraag_p.suffix.lower() == ".csv":
        vr = pd.read_csv(vraag_p)
    else:
        vr = pd.read_excel(vraag_p)
    if 'vraag' not in vr.columns:
        return "Klantforecast: kolom 'vraag' ontbreekt", None
    vraag_lijst = vr['vraag'].tolist()

    # 3) hire-plan
    plan={}
    if hire_plan_txt.strip():
        for seg in hire_plan_txt.split(","):
            idx,n = seg.split(":")
            plan[int(idx.strip())] = int(n.strip())

    # 4) ramp-profiel
    ramp=[int(x) for x in ramp_csv.split(",") if x.strip().isdigit()]

    # 5) patch & reload
    _patch_paths(kdb_p, founders_p, active_p,
                 int(buffer_val), ramp, vraag_lijst)

    # 6) forecast + CI
    df = fc.forecast(plan)
    ci = fc.forecast_ci(plan)

    # 7) fan-chart
    x = pd.to_datetime(ci['Maand'])
    plt.figure(figsize=(8,4))
    plt.plot(x, ci['Cap_mean'], lw=2, label="verwacht")
    plt.fill_between(x, ci['Cap_low'], ci['Cap_high'],
                     alpha=.25, label="95% CI")
    plt.plot(x, ci['Vraag'], ls='--', c='red', label='vraag+buffer')
    plt.xticks(rotation=45); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout()
    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.savefig(img_path, dpi=120); plt.close()

    return df, img_path


# ───────────────────────── Gradio UI ───────────────────────────
with gr.Blocks(title="Boekhouder-forecast (4 uploads)") as demo:
    gr.Markdown("## Upload 4 Excel/CSV-bestanden en stel hire-plan in")

    with gr.Row():
        kdb_up  = gr.File(label="Historisch KdB (.xlsx)",     file_types=[".xlsx"])
        fd_up   = gr.File(label="Historisch Founders (.xlsx)",file_types=[".xlsx"])
    with gr.Row():
        act_up  = gr.File(label="Actief personeel (.xlsx)",   file_types=[".xlsx"])
        vraag_up= gr.File(label="Klantforecast (.csv/.xlsx, kolom 'vraag')",
                          file_types=[".csv",".xlsx"])

    hire_txt = gr.Textbox(label="Hire-plan (bijv. 5:2,7:1)(juli=0,augustus=1,september=2, etc.", value="")
    buffer   = gr.Number(label="Buffer (+klanten)", value=100, precision=0)
    ramp_txt = gr.Textbox(label="Ramp-up CSV", value="0,0,0,50,80,100,120,140,140")

    btn = gr.Button("Run forecast")

    out_df  = gr.Dataframe(label="Forecast-tabel")
    out_img = gr.Image(type="filepath", label="Fan-chart")

    btn.click(run_forecast,
              inputs=[kdb_up, fd_up, act_up, vraag_up,
                      hire_txt, buffer, ramp_txt],
              outputs=[out_df, out_img])

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"]="False"
    demo.launch(share=True)
