import streamlit as st
from helpers import data_cleaning, baseline_value, race_explain_visualizer
from constants import TRACKS, PIT_STOP_LOSS
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# """GOALS
# - figure out what metrics I want to be changed by the user
# """

## MODEL LOADING ##
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("../models/UPDATED_lap_time_predictor.json")
    return model

model = load_model()

## STREAMLIT UI __INIT__ ##
def intialize_window():
    """
    Setup the basic metadata of the page.
    """
    st.set_page_config(
        page_title= "racr",
        page_icon=":racing_car:",
        layout="wide"
    )
    st.title("🏎️ Race Strategy Simulator")

## VISUALIZATION ##
def plot_circuit(track_choice, show_corners=True):
    """
    Pull the circuit information and create a plotly object.
    """
    df = pd.read_csv("../data/circuit_info.csv")
    track_df = df[(df["track"] == track_choice) & (df["type"]=="track")].sort_values("seq")
    corner_df = df[(df["track"] == track_choice) & (df["type"]=="corner")]

    fig = go.Figure()

    # Track line
    fig.add_trace(go.Scatter(
        x=track_df["X"],
        y=track_df["Y"],
        mode="lines",
        line=dict(color="white", width=6),
        hoverinfo="skip"
    ))

    # Corners
    if show_corners and not corner_df.empty:
        fig.add_trace(go.Scatter(
            x=corner_df["X"],
            y=corner_df["Y"],
            mode="markers+text",
            text=corner_df["corner"],
            textposition="top center",
            marker=dict(color="red", size=9),
            hovertemplate="Corner %{text}<br>Angle: %{customdata}°<extra></extra>",
            customdata=corner_df["angle"]
        ))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="black"
    )

    fig.update_traces(showlegend=False)

    return fig

## SIMULATION LOGIC ##
def simulate_race(strategy, track, track_length, total_laps, model):
    """
    Simulates a sequence of lap predictions based on different decisions and resets.

    Args:
        strategy: a list of (lap to pit, tire compound) tuples
        track: circuit
        total_laps: total laps of the specified Grand Prix
        model: trained lap-time predictor
    """
    # intialize simulator variables
    print(strategy)
    current_compound = strategy[0]["compound"]
    next_pit_index = 1  # pointer in the strategy list
    tire_age = 0
    race_time = 0.0
    lap_times = []
    laps_pit = []

    for lap in range(1, total_laps+1):
        tire_age += 1  # age of tire goes up with lap 1:1

        # check if current lap needs a pit
        if next_pit_index < len(strategy) and lap == strategy[next_pit_index]["start_lap"]:
            current_compound = strategy[next_pit_index]["compound"]  # next pit index = 1 (at this point)
            race_time += PIT_STOP_LOSS  # NOTE: add a pit loss (either generated or static)
            next_pit_index += 1  # move the pointer
            tire_age = 0  # reset tire age

            laps_pit.append(lap)

        # build feature inputs
        new_feature_vals = pd.DataFrame(data={
            "track": [track],
            f"compound_{current_compound}": [1],
            "tire_age": [tire_age],
            "tire_age_squared": [tire_age**2],
            "circuit_length(km)": [track_length]
        })

        features = data_cleaning(new_feature_vals)
        print(features)

        # predict lap time
        lap_time = float(model.predict(features)[0])
        lap_times.append(lap_time)
        race_time += lap_time

    return race_time, lap_times, laps_pit

## MAIN UI BUILD ##
def build_ui_structure():
    """
    Building the UI of the application.
    """
    # # create page foundation
    # st.title("Lap Time Simulator")
    # st.write("Test your lap strategy!")
    # # st.write("Pick a track, pick a tire, and see ...")

    # pull specific data used later in workflow
    track_choice = st.selectbox("Select a track:", TRACKS, index=0, placeholder="Select track...")
    track_df = pd.read_csv("../data/track_df.csv")
    track_length = track_df.loc[(track_df["track"] == track_choice), "circuit_length(km)"].values[0]
    total_laps = int(track_df.loc[(track_df["track"] == track_choice), "laps"].values[0])
    show_corners = st.checkbox("Show corner labels", value=True)

    # split a column (col1 = track visual, col2 = parameters user will change)
    col1, col2 = st.columns([3, 2]) 

    # track chosen
    with col1:
        fig = plot_circuit(track_choice, show_corners)
        st.plotly_chart(fig, width="stretch", config={"staticPlot": True})

    # data
    with col2:
        st.subheader("Strategy Builder")

        # intialize user strategy
        strategy = []
        current_lap = 1

        # number of stints
        # stint: continous period of time a car stays on track between pit stops
        num_stints = st.number_input("Number of Stints", 1, 4, 2)

        # # weather (rain/no rain)
        # is_rain = st.checkbox("Rain", value=False)
        # is_rain = int(is_rain)  

        for i in range(num_stints):
            # tire choice
            compound = st.selectbox(f"Stint {i+1} compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], key=f"compound_{i}", index=1)  # default is MEDIUM
            print(compound)
            start_lap = current_lap
            strategy.append({
                "start_lap": start_lap,
                "compound": compound
            })

            if i < num_stints - 1: 
                current_lap = st.slider(
                    f"Pit after lap (stint {i+1})",
                    start_lap+1,
                    total_laps,
                    start_lap+15,
                    key=f"pit_{i}"
                )

    race_time, lap_times_lst, laps_pit_lst = simulate_race(strategy, track_choice, track_length, total_laps, model)
    base = baseline_value(track_choice, total_laps)
    delta_val = base - race_time
    delta_val_sec = f"+{delta_val:.2f} sec" if delta_val < 0 else f"-{delta_val:.2f} sec"

    # convert to a time format
    base = time.strftime("%H:%M:%S", time.gmtime(base))
    pred = time.strftime("%H:%M:%S", time.gmtime(race_time))
    st.markdown(f"**Baseline value calculated ({base}), assumes flat average lap time, no pit stops.")

    st.write("after predictions", base, race_time)  # SANITY PRINT

    # write metric object to show change
    col1, col2 = st.columns(2)
    col1.metric("Lap Time Change", value=base, delta=delta_val_sec, border=True, delta_color="inverse")
    col2.metric("Predicted Lap Time", value=pred, delta=delta_val_sec, delta_color="inverse", border=True)

    # strategy explaination
    fig = race_explain_visualizer(lap_times_lst, laps_pit_lst)

## MAIN ##
# initialize page 
intialize_window()
build_ui_structure()