# new_settings.py
"""
Central settings for the NEW workforce forecast pipeline.
All units are *customers per month* unless stated otherwise.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

@dataclass
class Config:
    # Dates
    as_of: date                      # Date on which the forecast is generated (censoring date)
    start_year: int                  # Forecast start year
    start_month: int                 # Forecast start month (1..12)
    horizon_months: int = 12         # Forecast length in months

    # Capacity assumptions
    cap_per_employee: float = 60.0   # customers/month delivered by one fully productive employee
    availability_factor: float = 0.95  # generic reduction for absence/holidays

    # Ramp profile (customers/month contribution per hire)
    ramp: List[float] = field(default_factory=lambda: [0,0,0,50,80,100,120,140,140])

    # Buffer (extra customers above demand that must also be covered)
    buffer: int = 0

    # Planner constraints
    max_hires_per_month: int = 3

    # Monte Carlo settings for uncertainty bands
    mc_draws: int = 300
    mc_seed: int = 12345

    # Input file paths (can be CSV or Excel; autodetected by extension)
    kdb_file: Optional[str] = None       # historic roster (KdB) with 'In dienst','Uit dienst'
    founders_file: Optional[str] = None  # historic roster (Founders) same columns
    active_file: Optional[str] = None    # active roster (people active at as_of) with 'In dienst','Uit dienst' optional
    demand_file: Optional[str] = None    # monthly demand with a column named 'vraag' (lower case)

    # Column names (Dutch defaults)
    col_start: str = "In dienst"
    col_end: str = "Uit dienst"
    col_demand: str = "vraag"

    # Optional cohort label column (ignored if not present)
    col_cohort: str = "cohort"

def default_config(as_of: date) -> Config:
    # Forecast from next month for 12 months by default
    start_year = as_of.year + (1 if as_of.month == 12 else 0)
    start_month = 1 if as_of.month == 12 else as_of.month + 1
    return Config(as_of=as_of, start_year=start_year, start_month=start_month)
