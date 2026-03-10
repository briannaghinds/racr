"""
@author: Brianna Hinds
Description: helper funcitons for the F1 application
"""

from constants import INPUT_COLS, DEFAULT_VALS, TIRE_COLORS, TIRE_DEGRAD
from telemetry_profiles import lookup_profile

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go


# define data paths (no path errors)
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

def _data(filename: str) -> str:
    return os.path.join(_DATA, filename)

def data_cleaning(user_choices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures all model inputs columns are included and in the correct order.

    ARGS
        user_choices: DataFrame object of what choices the user picked.
    """
    cols_missing = [i for i in INPUT_COLS if i not in user_choices]

    for cols in cols_missing:
        user_choices[cols] = DEFAULT_VALS.get(cols, 0)

    # make sure input is in order the model expects
    user_choices = user_choices[INPUT_COLS]

    return user_choices


## BASELINE LAP TIMES ##
def baseline_value(track_choice: str, total_laps: int) -> float:
    """
    Obtains the median singular lap time and the overlap median lap time. (FLAT AVERAGE BASELINE)

    ARGS
        track_choice: user's track choice from the UI
        total_laps: total amount of laps a chosen Grand Prix is
    """
    data = _data("baseline_references.csv")

    if os.path.exists(data):
        baseline_df = pd.read_csv(data)
        row = baseline_df.loc[baseline_df["track"] == track_choice, "median_lap_time"]

        # if the row is filled with a value mutiply the time with number of laps
        if not row.empty:
            return float(row.values[0]) * total_laps
        
    # else return 1:30 minutes * the number of laps
    return 90.0 * total_laps

def build_baseline_lap_times(track_choice: str, total_laps: int, compound: str = "MEDIUM") -> list[float]:
    """
    Realistic baseline with a linear tire degradation, so the cumulative gap chart is meaningful.
    Baseline is no longer = median lap time * # of laps.
    Returns a list of different lap times based on tire degradation.

    ARGS
        track_choice: user's track choice from the UI
        total_laps: total amount of laps a chosen Grand Prix is
        compound: user's tire compount per stint
    """
    path = _data("baseline_references.csv")
    deg = TIRE_DEGRAD.get(compound.upper(), 0.07)

    if os.path.exists(path):
        baseline_df = pd.read_csv(path)
        row = baseline_df.loc[baseline_df["track"] == track_choice, "median_lap_time"]
        base = float(row.values[0]) if not row.empty else 90.0
    else:
        base = 90.0

    return [base + deg * lap for lap in range(total_laps)]

def load_track_encoding() -> dict[str, float]:
    """
    Load the per-track target encoding lookup built in the training notebook.
    Falls back to the global mean (92.73) for any track not in the CSV.
    """
    path = _data("track_encoding.csv")
    if not os.path.exists(path):
        return {}   # simulator will use DEFAULT_VALS["track_te"] as fallback

    df = pd.read_csv(path)
    return dict(zip(df["track"], df["track_te_value"]))


## TELEMETRY PROFILES ##
def load_telemetry_profiles() -> pd.DataFrame | None:
    """
    Load the telemetry_profiles.csv if it exists.
    Returns None if the file does not exist.
    """
    path = _data("telemetry_profiles.csv")

    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None

def _lookup_profile(profiles: pd.DataFrame, track_choice: str, compound: str, tire_age: int | float) -> dict:
    """
    Creates telemetry profile dictionaries for the tuple (track, compound, tire age).
    Returns the either a filled or blank dictionary.

    ARGS
        profiles: DataFrame from telemetry_profiles.csv
        track_choice: circuit chosen by user
        compound: tire compound chosen by user
        tire_age: current tire age in laps
    """
    return lookup_profile(profiles, track_choice, compound, tire_age)

def build_telemetry_series(strategy: list[dict], lap_compounds: list[str], total_laps: int, track_choice: str, profiles: pd.DataFrame | None) -> dict[str, list[float]]:
    """
    Build a per-lap telemetry series for the pit wall graphics via referencing the profile of the (compound, tire age) combination.
    Returns a dictionary with the following keys: avg_speed_kmh, throttle_percent, braking_percent, avg_gear, gear_changes.

    ARGS
        strategy: list of dictionaries containing the user's race strategy
        lap_compounds: list of tire compounds user chose for their strategy
        total_laps: total number of laps of the chosen circuit
        track_choice: circuit chosen by user
        profiles: telemetry profiles defined from _lookup_profiles()
    """
    if profiles is None:
        return {}
    
    # build the per-lap tire age from the passed strategy
    tire_ages = []
    stints = list(strategy)
    for i, stint in enumerate(stints):
        start = stint["start_lap"]
        end = stints[i+1]["start_lap"] - 1 if i + 1 < len(stints) else total_laps

        for lap_offset in range(end-start+1):
            tire_ages.append(lap_offset + 1)

    series = {
        "avg_speed_kmh": [],
        "throttle_percent": [],
        "braking_percent": [],
        "avg_gear": [],
        "gear_changes": []
    }

    for lap_idx in range(total_laps):
        compound = lap_compounds[lap_idx] if lap_idx < len(lap_compounds) else "MEDIUM"
        tire_age = tire_ages[lap_idx] if lap_idx < len(tire_ages) else lap_idx + 1
        profs = _lookup_profile(profiles, track_choice, compound, tire_age)

        for key in series:
            val = profs.get(key, np.nan)

            # add a small lap to lap jitter (so the line is not perfectly flat since all laps are exact)
            if not np.isnan(val):
                jitter = np.random.default_rng(lap_idx + hash(key) % 100).normal(0, val * 0.005)
                val += jitter

            series[key].append(val)

    return series