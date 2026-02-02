"""
@author: Brianna Hinds
Description: helper funcitons for the F1 application
"""
from constants import INPUT_COLS, DEFAULT_VALS
import pandas as pd
import matplotlib.pyplot as plt


def data_cleaning(user_choices, pull_default=True):
    cols_missing = [i for i in INPUT_COLS if i not in user_choices]

    for cols in cols_missing:
        user_choices[cols] = DEFAULT_VALS.get(cols, 0)

    # make sure input is in order the model expects
    user_choices = user_choices[INPUT_COLS]

    return user_choices

def baseline_value(track_choice, total_laps):
    baseline_df = pd.read_csv("../data/baseline_references.csv")
    baseline_val = baseline_df.loc[baseline_df["track"] == track_choice, "median_lap_time"]
    baseline_val = baseline_val.values[0] * total_laps  # access the first thing in the Series

    return baseline_val

def race_explain_visualizer(lap_times: list, pits: list):
    """
    Takes the lap times list and creates a line plot that plots lap vs lap time (vertical dashes will be placed on the pit stop laps).
    Returns a matplotlib object.

    Args
        lap_times: list of all lap times predicted by the XGBoost model
        pits: list of laps that were pit stops
    """
    # make a line plot (lap vs lap_times)

    pass
