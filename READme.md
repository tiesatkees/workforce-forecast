# Workforce Forecast (Streamlit)

## Run lokaal
python -m venv .venv
source .venv/bin/activate    # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_streamlit.py

## Gebruik
1. Upload 3 bestanden (KdB, Founders, Actief) en optioneel Klantforecast.
2. Kies forecast-modus, datums, buffer, ramp.
3. Optioneel: handmatige vraag (CSV) of upload een vraagbestand.
4. Optioneel: automatisch hire-plan (als hire_planner.py aanwezig is).
5. Klik "Run forecast".
