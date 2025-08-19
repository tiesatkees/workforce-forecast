# paden.py
from pathlib import Path
from datetime import datetime

# ---------- MAPPEN ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" # Optioneel: map voor resultaten

# Zorg ervoor dat de output map bestaat
# OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- INPUT BESTANDEN ----------
KDB_FILE      = DATA_DIR / "CompanyEmployeeHoursSalary_KeesdeBoekhouder_Office_BV.xlsx"
FOUNDERS_FILE = DATA_DIR / "CompanyEmployeeHoursSalary_Founders_Finance_BV.xlsx"
ACTIVE_FILE   = DATA_DIR / "Kopie van Operations Founders 2.0 (2) (2).xlsx"

# ---------- HISTORISCHE DATA (KM Training) ----------
SHEET_HISTORISCH = "Page_1"         # In KDB_FILE & FOUNDERS_FILE
COL_IN_DIENST_HIST = "In dienst"
COL_UIT_DIENST_HIST = "Uit dienst"
COL_KOSTENPLAATSEN_HIST = "Kostenplaatsen" # Kolomnaam voor kostenplaatsen
KOSTENPLAATSEN_WAARDEN = [
    "Operations | Bookies KdB",
    "Operations | Bookies Founders"
]

# ---------- ACTIEVE DATA & CAPACITEIT ----------
SHEET_ACTIEF = "Kopie van overzicht2" # In ACTIVE_FILE
COL_ID_ACTIEF = "ID"                 # ID kolom (belangrijk!)
COL_START_ACTIEF = "Start"           # STARTKOLOM ACTIEF: Check of dit 'Start' of 'Startdatum' moet zijn!
COL_STATUS_ACTIEF = "Status"
ACTIEVE_STATUS_WAARDEN = ["In dienst", "Actief", "Proeftijd"]
# Lijst van kolomnamen in Excel voor capaciteit - PAS DIT AAN AAN JE EXCEL!
# Voorbeeld: ['mei', 'juni', 'juli', 'augustus', 'september', 'oktober', 'november', 'december', 'januari', 'februari', 'maart']
# Het is vaak beter om dit in de code zelf dynamisch te bepalen o.b.v. de forecast maanden.

# ---------- FORECAST PARAMETERS ----------
FORECAST_START_DT = datetime(2025, 7, 1)    # Startmaand forecast (1 Juli 2025)
FORECAST_GENERATION_DATE = datetime(2025, 6, 30) # Datum waarop forecast gemaakt wordt
CONTRACT_BUFFER_MAANDEN = 0                 # Aantal maanden gegarandeerd (1 = tot einde volgende maand)

# Vraag-prognose per maand (klantvraag)
KLANTVRAAG_PROGNOSE = [ 
    1488,  # jul-25
    1537,  # aug-25
    1595,  # spet-25
    1640,  # okt-26
    1688,  # nov-26
    1737,  # dec-26
    1792,
    1856,
    1915,
]
# Het aantal maanden wordt nu impliciet bepaald door len(KLANTVRAAG_PROGNOSE)

# ---------- MAAND MAPPING (Voor referentie/rapportage, of als kolomnamen exact matchen) ----------
# Gebruik deze OF definieer een mapping van forecast maand naar *exacte* Excel kolomnaam.
# Wees voorzichtig met hoofdletters!
MAAND_NAMEN_NL = {
    1: 'januari', 2: 'februari', 3: 'maart', 4: 'april', 5: 'mei', 6: 'juni',
    7: 'juli', 8: 'augustus', 9: 'september', 10: 'oktober', 11: 'november', 12: 'december'
}
# Mogelijk handiger voor kolom lookup (als je Excel afkortingen gebruikt):
MAAND_AFKORTINGEN_NL = {
    1: 'jan', 2: 'feb', 3: 'mrt', 4: 'apr', 5: 'mei', 6: 'jun',
    7: 'jul', 8: 'aug', 9: 'sep', 10: 'okt', 11: 'nov', 12: 'dec'
}

