#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 09:21:14 2025

@author: tiesvandenheuvel
"""

#!/usr/bin/env python3
# gradio_forecast.py  – 4 uploads + opt. hand­matige vraaglijst
import gradio as gr, pandas as pd, numpy as np, matplotlib.pyplot as plt
import tempfile, shutil, importlib, os
from pathlib import Path

import paden
import forecast_full as fc        # bevat forecast() & forecast_ci()

# ───────── helpers ────────────────────────────────────────────
def _tmp_copy(upload) -> Path:
    f = tempfile.NamedTemporaryFile(delete=False,
                                    suffix=Path(upload.name).suffix)
    shutil.copyfile(upload.name, f.name)
    return Path(f.name)

def _patch_paths(kdb, founders, active, buffer, ramp, vraag):
    paden.KDB_FILE      = kdb
    paden.FOUNDERS_FILE = founders
    paden.ACTIVE_FILE   = active
    paden.KLANTVRAAG_PROGNOSE = vraag
    fc.BUFFER = buffer
    fc.RAMP   = ramp
    importlib.reload(fc)          # triggert _build_km → verse SEAS

# ───────── hoofd-functie ─────────────────────────────────────
def run_forecast(kdb_file, founders_file, active_file,
                 vraag_file, vraag_txt,
                 hire_plan_txt, buffer_val, ramp_csv):

    # 0) tmp-kopieën
    kdb_p      = _tmp_copy(kdb_file)
    founders_p = _tmp_copy(founders_file)
    active_p   = _tmp_copy(active_file)

    # 1) VRAAG:  upload > tekst > default
    if vraag_file is not None:
        vr_path = _tmp_copy(vraag_file)
        df_vr = (pd.read_csv(vr_path) if vr_path.suffix==".csv"
                 else pd.read_excel(vr_path))
        if 'vraag' not in df_vr.columns:
            return "Klantforecast-file mist kolom 'vraag'.", None
        vraag = df_vr['vraag'].tolist()
    elif vraag_txt.strip():
        try:
            vraag = [int(x) for x in vraag_txt.split(",")]
        except ValueError:
            return "Vraag-lijst: gebruik comma-gescheiden integers.", None
    else:
        vraag = paden.KLANTVRAAG_PROGNOSE

    # 2) hire-plan
    plan={}
    if hire_plan_txt.strip():
        for seg in hire_plan_txt.split(","):
            idx,n = seg.split(":")
            plan[int(idx.strip())]=int(n.strip())

    # 3) ramp
    ramp=[int(x) for x in ramp_csv.split(",") if x.strip().isdigit()]

    # 4) paths + reload
    _patch_paths(kdb_p, founders_p, active_p, int(buffer_val), ramp, vraag)

    # 5) forecast
    df = fc.forecast(plan);  ci = fc.forecast_ci(plan)

    # 6) plot
    x = pd.to_datetime(ci['Maand'])
    plt.figure(figsize=(8,4))
    plt.plot(x, ci['Cap_mean'], lw=2, label="verwacht")
    plt.fill_between(x, ci['Cap_low'], ci['Cap_high'], alpha=.25,label="95% CI")
    plt.plot(x, ci['Vraag'], ls='--', c='red', label='vraag+buffer')
    plt.xticks(rotation=45); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    img = tempfile.NamedTemporaryFile(delete=False,suffix=".png").name
    plt.savefig(img, dpi=120); plt.close()

    return df, img

# ──────────────── Gradio UI ───────────────────────────────────
with gr.Blocks(title="Boekhouder-forecast (dynamisch)") as demo:
    gr.Markdown("### Upload bestanden en/of vul handmatig de forecast in")

    with gr.Row():
        kdb_up  = gr.File(label="Historisch KdB (.xlsx)",     file_types=[".xlsx"])
        fd_up   = gr.File(label="Historisch Founders (.xlsx)",file_types=[".xlsx"])
    with gr.Row():
        act_up  = gr.File(label="Actief personeel (.xlsx)",   file_types=[".xlsx"])
        vraag_up= gr.File(label="Klantforecast file (.csv/.xlsx met kolom 'vraag')",
                          file_types=[".csv",".xlsx"])

    vraag_txt = gr.Textbox(label="(Optie) Handmatige vraag-lijst (comma-sep)", value="")
    hire_txt  = gr.Textbox(label="Hire-plan (bijv. 5:2,7:1)", value="")
    buffer    = gr.Number(label="Buffer", value=100, precision=0)
    ramp_txt  = gr.Textbox(label="Ramp-up CSV", value="0,0,0,50,80,100,120,140,140")

    btn = gr.Button("Run forecast")

    out_df  = gr.Dataframe(label="Forecast-tabel")
    out_img = gr.Image(type="filepath", label="Fan-chart")

    btn.click(run_forecast,
              inputs=[kdb_up, fd_up, act_up, vraag_up,
                      vraag_txt, hire_txt, buffer, ramp_txt],
              outputs=[out_df, out_img])

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"]="False"
    demo.launch()
