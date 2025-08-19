#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:31:58 2025

@author: tiesvandenheuvel
"""

# data_loader.py
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import paden

def load_active_staff_and_capacity():
    """
    Laadt de actieve medewerkers data, filtert deze, en koppelt capaciteitskolommen
    aan de forecast maanden.
    """
    print(f"INFO: Laden van actieve medewerkers uit {paden.ACTIVE_FILE}...")
    try:
        df = pd.read_excel(paden.ACTIVE_FILE, sheet_name=paden.SHEET_ACTIEF)
    except FileNotFoundError:
        print(f"FOUT: Bestand niet gevonden: {paden.ACTIVE_FILE}")
        return None
    except Exception as e:
        print(f"FOUT: Kon {paden.ACTIVE_FILE} niet lezen. Error: {e}")
        return None

    # --- Kolom Check & Data Cleaning ---
    required_cols = [
        paden.COL_ID_ACTIEF,
        paden.COL_START_ACTIEF,
        paden.COL_STATUS_ACTIEF
    ]
    if not all(col in df.columns for col in required_cols):
        print(f"FOUT: Niet alle vereiste kolommen ({required_cols}) gevonden in {paden.SHEET_ACTIEF}.")
        return None

    # Filter op actieve status
    df_active = df[df[paden.COL_STATUS_ACTIEF].isin(paden.ACTIEVE_STATUS_WAARDEN)].copy()

    # Converteer startdatum
    df_active[paden.COL_START_ACTIEF] = pd.to_datetime(df_active[paden.COL_START_ACTIEF], errors='coerce')
    df_active = df_active.dropna(subset=[paden.COL_ID_ACTIEF, paden.COL_START_ACTIEF])

    # Maak ID uniek (voor het geval dat) en zet als index
    if df_active[paden.COL_ID_ACTIEF].duplicated().any():
        print(f"WAARSCHUWING: Dubbele ID's gevonden in {paden.COL_ID_ACTIEF}. Dit kan problemen veroorzaken.")
    df_active = df_active.set_index(paden.COL_ID_ACTIEF, drop=False) # Houd ID ook als kolom

    print(f"INFO: {len(df_active)} actieve medewerkers geladen.")

    # --- Capaciteitskolommen Verwerken ---
    print("INFO: Verwerken capaciteitskolommen...")
    forecast_months_count = len(paden.KLANTVRAAG_PROGNOSE)
    forecast_dates = [paden.FORECAST_START_DT + relativedelta(months=i) for i in range(forecast_months_count)]

    capacity_data = {} # Dict om {ID: [cap_m1, cap_m2, ...]} op te slaan

    for medewerker_id, row in df_active.iterrows():
        capacities = []
        for dt in forecast_dates:
            month_name = paden.MAAND_NAMEN_NL.get(dt.month) # Haal 'juli', 'augustus', etc.
            if not month_name:
                print(f"FOUT: Kan maandnaam niet vinden voor maand {dt.month}.")
                capacities.append(0) # Fallback
                continue

            # --- Belangrijk: Kolom Matching ---
            # Probeer kolom te vinden. Eerst met kleine letters, dan met hoofdletter.
            # PAS DIT AAN als je Excel-kolommen ANDERS heten!
            col_to_try = month_name.lower()
            col_to_try_cap = month_name.capitalize()

            if col_to_try in row:
                capacity = row[col_to_try]
            elif col_to_try_cap in row:
                capacity = row[col_to_try_cap]
            else:
                print(f"WAARSCHUWING: Capaciteitskolom '{col_to_try}' of '{col_to_try_cap}' niet gevonden. Gebruik 0 capaciteit voor {medewerker_id} in {dt.strftime('%Y-%m')}.")
                capacity = 0

            # Zorg dat capaciteit numeriek is
            capacities.append(pd.to_numeric(capacity, errors='coerce'))

        capacity_data[medewerker_id] = capacities

    # Voeg de capaciteiten als een lijst toe aan het DataFrame, of maak een nieuw DataFrame
    # Een manier is om een nieuw DataFrame te maken met ID, Startdatum en de maandelijkse capaciteiten
    df_result = df_active[[paden.COL_ID_ACTIEF, paden.COL_START_ACTIEF]].copy()

    # Maak kolomnamen voor de capaciteiten, bijv. Cap_2025-07
    cap_cols = [f"Cap_{dt.strftime('%Y-%m')}" for dt in forecast_dates]

    # Maak een DataFrame van de capacity_data en voeg het samen
    cap_df = pd.DataFrame.from_dict(capacity_data, orient='index', columns=cap_cols)
    df_result = df_result.join(cap_df)

    # Vul eventuele NaNs in capaciteit met 0
    df_result[cap_cols] = df_result[cap_cols].fillna(0)

    print(f"INFO: Capaciteitsdata succesvol verwerkt voor {len(df_result)} medewerkers.")
    return df_result, forecast_dates

 #Voorbeeld hoe je dit zou kunnen gebruiken:
if __name__ == "__main__":
     active_staff_df, forecast_dates_list = load_active_staff_and_capacity()
     if active_staff_df is not None:
         print("\n--- Voorbeeld Geladen Data ---")
         print(active_staff_df.head())
         print("\n--- Forecast Datums ---")
         print(forecast_dates_list)