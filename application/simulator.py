"""
@author: Brianna Hinds
Description: Race Strategy Simulator (main page)
"""

import streamlit as st
from helpers import data_cleaning, baseline_value, plot_circuit_animated, race_explain_visualizer
from constants import APP_UI, TRACKS, PIT_STOP_LOSS
import time
import xgboost as xgb
import plotly.graph_objects as go
import pandas as pd

## MODEL LOADING ##
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("../models/UPDATED_lap_time_predictor.json")
    return model

model = load_model()

## STREAMLIT UI __INIT__ ##
def initialize_window():
    """
    Setup the basic metadata of the page with F1 theming.
    """
    st.set_page_config(
        page_title="F1 Race Strategy Simulator",
        # page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for F1 dashboard aesthetic
    st.markdown(APP_UI, unsafe_allow_html=True)

## SIMULATION LOGIC ##
def simulate_race(strategy, track, track_length, total_laps, model):
    """
    Simulates a sequence of lap predictions based on strategy decisions.
    Returns total race time, a list of all predicted lap times, list of laps that were pitted.
    
    Args:
        strategy: list of dicts with 'start_lap' and 'compound' keys
        track: circuit name
        track_length: circuit length in km
        total_laps: total laps in the race
        model: trained XGBoost lap-time predictor
    """
    # initialize simulation variables
    current_compound = strategy[0]["compound"]
    next_pit_index = 1
    tire_age = 0
    race_time = 0.0
    lap_times = []
    laps_pit = []
    
    for lap in range(1, total_laps + 1):
        # check if we need to pit at this lap
        if next_pit_index < len(strategy) and lap == strategy[next_pit_index]["start_lap"]:
            # car has pitted
            current_compound = strategy[next_pit_index]["compound"]
            race_time += PIT_STOP_LOSS
            tire_age = 0  # reset tire age
            next_pit_index += 1
            laps_pit.append(lap)
        else:
            # if no put stop, tire age + 1
            tire_age += 1
        
        # build user features for prediction
        new_feature_vals = pd.DataFrame(data={
            "track": [track],
            f"compound_{current_compound}": [1],
            "tire_age": [tire_age],
            "tire_age_squared": [tire_age ** 2],
            "circuit_length(km)": [track_length]
        })
        
        features = data_cleaning(new_feature_vals)
        
        # predict the lap time
        lap_time = float(model.predict(features)[0])
        lap_times.append(lap_time)
        race_time += lap_time
    
    return race_time, lap_times, laps_pit

## MAIN UI BUILD ##
def build_ui_structure():
    """
    Build F1 race engineer dashboard UI.
    """
    # Header
    st.markdown("<h1>🏎️ RACE STRATEGY SIMULATOR 💨</h1>", unsafe_allow_html=True)
    st.markdown('<div class="subtitle">POWERED BY MACHINE LEARNING TELEMETRY</div>', unsafe_allow_html=True)
    # st.markdown('<div class="subtitle">CREATED BY BRIANNA HINDS</div>', unsafe_allow_html=True)
    
    # track selection
    col_track, col_laps, col_circuit_length = st.columns([3, 1, 1])
    
    with col_track:
        track_choice = st.selectbox(
            "SELECT CIRCUIT",
            TRACKS,
            index=0,
            help="Choose the Grand Prix circuit",
            accept_new_options=False
        )
    
    # pull track data after user choice
    track_df = pd.read_csv("../data/track_df.csv")
    track_length = track_df.loc[(track_df["track"] == track_choice), "circuit_length(km)"].values[0]
    total_laps = int(track_df.loc[(track_df["track"] == track_choice), "laps"].values[0])
    
    with col_laps:
        st.markdown(f"""
            <div class="telemetry-item">
                <div class="telemetry-label">Total Laps</div>
                <div class="telemetry-value">{total_laps}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_circuit_length:
        st.markdown(f"""
            <div class="telemetry-item">
                <div class="telemetry-label">Circuit Length (km)</div>
                <div class="telemetry-value">{track_length}</div>
            </div>
        """, unsafe_allow_html=True)

    # main dashboard layout
    col1, col2 = st.columns([3, 2])
    
    # LEFT PANEL: track visualization
    with col1:
        st.markdown("### TRACK TELEMETRY")
        
        # build strategy first to pass to animation
        compounds_used = []
        with col2:
            st.markdown("### STRATEGY CONFIGURATION")
            
            # number of stints
            num_stints = st.number_input(
                "NUMBER OF STINTS",
                min_value=1,
                max_value=4,
                value=2,
                help="A stint is a continuous period between pit stops"
            )
            
            # build strategy based on user 
            strategy = []
            current_lap = 1
            
            for i in range(num_stints):
                st.markdown(f'<span class="stint-badge">STINT {i+1}</span>', unsafe_allow_html=True)
                
                compound = st.selectbox(
                    f"Tire Compound",
                    ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
                    key=f"compound_{i}",
                    index=1,
                    help="Tire compound choice affects grip and degradation"
                )
                compounds_used.append(compound)
                
                start_lap = current_lap
                strategy.append({
                    "start_lap": start_lap,
                    "compound": compound
                })
                
                if i < num_stints - 1:
                    current_lap = st.slider(
                        f"Pit Stop After Lap",
                        min_value=start_lap + 1,
                        max_value=total_laps - (num_stints - i - 1),
                        value=min(start_lap + 15, total_laps - (num_stints - i - 1)),
                        key=f"pit_{i}",
                        help="Choose when to pit for fresh tires"
                    )
                    
                st.markdown("---")
        
        # Run simulation
        race_time, lap_times_strategy, laps_pit = simulate_race(
            strategy, track_choice, track_length, total_laps, model
        )
        
        # Calculate baseline (no pit stops, average lap time)
        baseline_time = baseline_value(track_choice, total_laps)
        
        # Create baseline lap times (uniform)
        avg_lap_time = baseline_time / total_laps
        lap_times_baseline = [avg_lap_time] * total_laps
        
        # Display animated track with both cars
        fig = plot_circuit_animated(
            track_choice,
            lap_times_strategy=lap_times_strategy,
            lap_times_baseline=lap_times_baseline,
            total_laps=total_laps,
            laps_pit=laps_pit,
            # tire_strategy=compound
        )
        
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False}
        )
        
        pit_summary = "READY" if not laps_pit else f"PIT PLANNED: LAP {', '.join(map(str, laps_pit))}"        # Track status
        st.markdown(f"""
            <div class="telemetry-bar">
                <div class="telemetry-item">
                    <div class="telemetry-label">Pit Stops</div>
                    <div class="telemetry-value">{len(laps_pit)}</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">Tires Used</div>
                    <div class="telemetry-value">{", ".join(map(str, set(compounds_used)))}</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">Status</div>
                    <div class="telemetry-value"><span class="status-indicator"></span> {pit_summary}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # model results
    st.markdown("### PERFORMANCE ANALYSIS")
    
    # calculate delta times
    delta_val = baseline_time - race_time
    is_faster = delta_val > 0
    delta_display = f"+{abs(delta_val):.2f}s" if not is_faster else f"-{abs(delta_val):.2f}s"
    
    # format times
    baseline_formatted = time.strftime("%H:%M:%S", time.gmtime(baseline_time))
    predicted_formatted = time.strftime("%H:%M:%S", time.gmtime(race_time))
    
    # display metrics
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.metric(
            "BASELINE TIME",
            baseline_formatted,
            help="Flat median lap time, no pit loss, no tire degradation."
        )
    
    with metric_col2:
        st.metric(
            "PREDICTED TIME",
            predicted_formatted,
            delta=delta_display,
            delta_color="inverse",
            help="Your strategy's predicted race time"
        )
    
    # performance verdict
    if is_faster:
        st.success(f"**FASTER BY {abs(delta_val):.2f} SECONDS** - Excellent strategy!")
    else:
        st.error(f"**SLOWER BY {abs(delta_val):.2f} SECONDS** - Consider adjusting your strategy")

    ##### BREAK HERE HAVE THE STRATEGY SUMMARY BE A SEPARATE PAGE
        
        # Strategy summary
        st.markdown("### 📋 STRATEGY SUMMARY")
        for i, stint in enumerate(strategy):
            start = stint["start_lap"]
            end = strategy[i+1]["start_lap"] - 1 if i+1 < len(strategy) else total_laps
            st.markdown(f"""
                <div style="background: #1a1a1a; padding: 0.8rem; margin: 0.5rem 0; border-left: 3px solid #e10600; border-radius: 4px;">
                    <strong>STINT {i+1}:</strong> Laps {start}-{end}<br>
                    <span style="color: #ffaa00;">Compound: {stint["compound"]}</span><br>
                    <span style="color: #888;">Duration: {end - start + 1} laps</span>
                </div>
            """, unsafe_allow_html=True)
        
        # Lap time chart
        st.markdown("### 📈 LAP TIME PROGRESSION")
        fig_progression = race_explain_visualizer(lap_times_strategy, laps_pit)
        st.pyplot(fig_progression)

## MAIN ##
initialize_window()
build_ui_structure()