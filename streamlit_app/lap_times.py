import streamlit as st
from constants import TRACKS, INPUT_COLS, DEFAULT_VALS
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# """GOALS
# - build all the UI first (NO "backend" yet)
# - make a simple UI with the track (start static) then implement a track drop down
# - figure out what metrics I want to be changed by the user
# """

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("../models/lap_time_predictor.json")
    return model

model = load_model()

def intialize_window():
    """
    Setup the basic metadata of the page.
    """
    st.set_page_config(
        page_title= "racr",
        page_icon=":racing_car:",
        layout="wide"
    )

def data_cleaning(user_choices):
    cols_missing = [i for i in INPUT_COLS if i not in user_choices]

    for cols in cols_missing:
        user_choices[cols] = DEFAULT_VALS.get(cols, 0)

    # make sure input is in order the model expects
    user_choices = user_choices[INPUT_COLS]

    # SANITY PRINTS
    print(user_choices)
    st.dataframe(user_choices)

    return user_choices

def lap_time_prediction(track_choice, user_choices):
    baseline_df = pd.read_csv("../data/baseline_references.csv")
    baseline_val = baseline_df.loc[baseline_df["track"] == track_choice, "avg_lap_time"]
    baseline_val = baseline_val.values[0]  # access the first thing in the Series

    # get input columns for model
    X = data_cleaning(user_choices)

    # predict new time based on user values
    time_prediction = float(model.predict(X))
    print(time_prediction)  # SANITY PRINT

    # # i want to return: baseline, predicted
    return baseline_val, time_prediction

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
        line=dict(color="white", width=7),
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
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig.update_traces(showlegend=False)

    return fig

def build_ui_structure():
    """
    Building the UI of the application.
    """
    # create page foundation
    st.title("Lap Time Simulator")
    st.write("Test your lap strategy!")
    # st.write("Pick a track, pick a tire, and see ...")

    # create a dropdown for each track
    track_choice = st.selectbox("Select a track:", TRACKS, index=0, placeholder="Select track...")
    show_corners = st.checkbox("Show corner labels", value=True)

    # split a column (col1 = track visual, col2 = parameters user will change)
    col1, col2 = st.columns([3, 2]) 

    # track chosen
    with col1:
        fig = plot_circuit(track_choice, show_corners)
        st.plotly_chart(fig, width="stretch", config={"staticPlot": True})

    # data
    with col2:
        # tire choice
        tire_choice = st.radio("Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], index=1)  # default is MEDIUM

        # tire age
        tire_age = st.slider("Tire Age (laps)", 1, 80, value=1)
        tire_age_sqrt = tire_age ** 2

        # weather (rain/no rain)
        is_rain = st.checkbox("Rain", value=False)
        is_rain = int(is_rain)  

    # display delta changes based on user input
    track_df = pd.read_csv("../data/track_df.csv")
    user_choice_df = pd.DataFrame(data={
        f"compound_{tire_choice}": [1],
        "tire_age": [tire_age],
        "tire_age_squared": [tire_age_sqrt],
        "is_rain": [is_rain],
        "circuit_length(km)": [track_df.loc[(track_df["track"] == track_choice), "circuit_length(km)"].values[0]]
    })

    base, pred = lap_time_prediction(track_choice, user_choice_df)
    delta_val = base - pred
    # delta_val_sec = abs(delta_val)
    delta_val_sec = f"+{delta_val:.2f} sec" if delta_val < 0 else f"-{delta_val:.2f} sec"

    # convert to a time format
    base = time.strftime("%M:%S", time.gmtime(base))
    pred = time.strftime("%M:%S", time.gmtime(pred))

    st.write("after predictions", base, pred)  # SANITY PRINT

    # write metric object to show change
    col1, col2 = st.columns(2)
    col1.metric("Lap Time Change", value=base, delta=delta_val_sec, border=True, delta_color="inverse")
    col2.metric("Predicted Lap Time", value=pred, delta=delta_val_sec, delta_color="inverse", border=True)







## MAIN ##
# initialize page 
intialize_window()
build_ui_structure()