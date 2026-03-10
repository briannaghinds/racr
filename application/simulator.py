"""
@author: Brianna Hinds
Description: Race Strategy Simulator (main page)
"""
from helpers import data_cleaning, build_baseline_lap_times, load_telemetry_profiles, load_track_encoding, build_telemetry_series, plot_circuit_animated, race_explain_visualizer, _lookup_profile
from constants import APP_UI, PIT_WALL_UI, TRACKS, PIT_STOP_LOSS, TIRE_COLORS
from pit_wall import build_pit_wall_figure

import os
import streamlit as st
import time
import xgboost as xgb
import plotly.graph_objects as go
import pandas as pd

## DATA PATHS ##
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
_MODEL = os.path.join(_HERE, "..", "models", "0.997_lap_time_predictor.json")

def _data(filename: str) -> str:
    return os.path.join(_DATA, filename)


## MODEL AND PROFILES LOADING ##
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(_MODEL)
    return model

model = load_model()

@st.cache_resource
def load_profiles():
    """
    Load the telemetry profiles once and cache it.
    Returns None if not built yet.
    """
    return load_telemetry_profiles()


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
    
    # Custom CSS for F1 dashboard aesthetic + pit wall
    st.markdown(APP_UI, unsafe_allow_html=True)
    st.markdown(PIT_WALL_UI, unsafe_allow_html=True)


## SIMULATION LOGIC ##
def simulate_race(strategy, track, track_length, total_laps, model, profiles):
    """
    Simulates a sequence of lap predictions based on strategy decisions.
    Returns total race time, a list of all predicted lap times, list of laps that were pitted.
    
    Args:
        strategy: list of dicts with 'start_lap' and 'compound' keys
        track: circuit name
        track_length: circuit length in km
        total_laps: total laps in the race
        model: trained XGBoost lap-time predictor
        profiles: telemetry profiles created in the earlier pipeline stages
    """
    # initialize simulation variables
    current_compound = strategy[0]["compound"]
    next_pit_index = 1
    tire_age = 0
    race_time = 0.0
    lap_times = []
    laps_pit = []
    lap_compounds = []
    track_encoding = load_track_encoding()
    
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
        new_feature_vals = {
            "track": track,
            f"compound_{current_compound}": 1,
            "tire_age": tire_age,
            "tire_age_squared": tire_age**2,
            "circuit_length(km)": track_length,
            "fuel_effect": lap / total_laps,
            "track_te": track_encoding.get(track, 92.727),
        }

        # add telemetry data
        if profiles is not None:
            prof = _lookup_profile(profiles, track, current_compound, tire_age)

            for feat in ("avg_speed_kmh", "throttle_percent", "braking_percent", "avg_gear", "gear_changes"):
                val = prof.get(feat)

                if val is not None and not (isinstance(val, float)):
                    new_feature_vals[feat] = val
        
        features = data_cleaning(pd.DataFrame([new_feature_vals]))
        
        # predict the lap time
        lap_time = float(model.predict(features)[0])
        lap_times.append(lap_time)
        lap_compounds.append(current_compound)
        race_time += lap_time
    
    return race_time, lap_times, laps_pit, lap_compounds


## MAIN UI BUILD ##
def build_ui_structure():
    """
    Build F1 race engineer dashboard UI.
    """
    # Header
    st.markdown("<h1>🏎️ RACE STRATEGY SIMULATOR 💨</h1>", unsafe_allow_html=True)
    st.markdown('<div class="subtitle">PIT WALL - POWERED BY MACHINE LEARNING TELEMETRY</div>', unsafe_allow_html=True)
    # st.markdown('<div class="subtitle">CREATED BY BRIANNA HINDS</div>', unsafe_allow_html=True)

    # load shared resources
    model = load_model()
    profiles = load_profiles()

    # show what telemetry is used (real/simulated)
    if profiles is not None:
        badge = '<span class="data-badge real">REAL TELEMETRY</span>'
    else:
        badge = '<span class="data-badge sim">SIMULATED TELEMETRY</span>'
        
    st.markdown(
        f'<p style="font-family:monospace;font-size:0.75rem;color:#888;">'
        f'Telemetry source: {badge}</p>',
        unsafe_allow_html=True,
    )
    
    # track selection
    col_track, col_laps, col_circuit_length = st.columns([4, 1, 1])
    
    with col_track:
        track_choice = st.selectbox(
            "SELECT CIRCUIT",
            TRACKS,
            index=0,
            help="Choose the Grand Prix circuit",
            accept_new_options=False
        )
    
    # pull track data after user choice
    track_df = pd.read_csv(_data("track_df.csv"))
    track_row = track_df[track_df["track"] == track_choice]
    track_length = float(track_row["circuit_length(km)"].values[0])
    total_laps = int(track_row["laps"].values[0])
    
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

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # main dashboard layout (3 column pit wall)
    col_strategy, col_telem = st.columns([1, 3])

    # LEFT PANEL: strategy builder
    with col_strategy:
        st.markdown('<div class="panel"><div class="panel-header"> STRATEGY CONFIGURATION </div>', unsafe_allow_html=True)

        # number of stints
        num_stints = st.number_input(
            "NUMBER OF STINTS",
            min_value=1,
            max_value=4,
            value=2,
            help="A stint is a continuous period between pit stops"
        )
        
        # build strategy based on user 
        strategy, compounds_used = [], []
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
            dot = TIRE_COLORS.get(compound, "#888")
            st.markdown(
                f'<span class="tire-dot" style="background:{dot}"></span>'
                f'<span style="font-size:0.8rem;color:#aaa">{compound}</span>',
                unsafe_allow_html=True,
            )
            compounds_used.append(compound)
            
            strategy.append({
                "start_lap": current_lap,
                "compound": compound
            })
            
            if i < num_stints - 1:
                current_lap = st.slider(
                    f"Pit Stop After Lap",
                    min_value=current_lap + 1,
                    max_value=total_laps - (num_stints - i - 1),
                    value=min(current_lap + 15, total_laps - (num_stints - i - 1)),
                    key=f"pit_{i}",
                    help="Choose when to pit for fresh tires"
                )
                
            st.markdown("---")

        st.markdown("</div>", unsafe_allow_html=True)

        # stint summary
        st.markdown('<div class="panel"><div class="panel-header">📋 Stint Summary</div>', unsafe_allow_html=True)
        for i, stint in enumerate(strategy):
            start = stint["start_lap"]
            end   = strategy[i + 1]["start_lap"] - 1 if i + 1 < len(strategy) else total_laps
            dot   = TIRE_COLORS.get(stint["compound"], "#888")
            st.markdown(f"""
            <div style="padding:6px 0; border-bottom:1px solid #1a1a1a;">
                <span class="tire-dot" style="background:{dot}"></span>
                <b>Stint {i+1}</b>
                <span style="color:#888;font-size:0.8rem;">
                    · Laps {start}–{end} ({end - start + 1} laps) · {stint["compound"]}
                </span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) 


    # run simulation
    race_time, lap_times_strategy, laps_pit, lap_compounds = simulate_race(
        strategy, track_choice, track_length, total_laps, model, profiles
    )    

    # NOTE: fix to have a baseline for the different compounds passed
    lap_times_baseline = build_baseline_lap_times(track_choice, total_laps)
    baseline_time = sum(lap_times_baseline)

    # ── RIGHT: metrics ────────────────────────────────────────────────────
    with col_telem:
        delta_val = baseline_time - race_time
        is_faster = delta_val > 0

        def fmt_time(secs):
            h = int(secs // 3600)
            m = int((secs % 3600) // 60)
            s = secs % 60
            return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m}:{s:05.2f}"

        delta_cls  = "good" if is_faster else "bad"
        delta_sign = f"-{abs(delta_val):.2f}s" if is_faster else f"+{abs(delta_val):.2f}s"

        st.markdown('<div class="panel"><div class="panel-header">⚡ Race Metrics</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="telem-row">
            <div class="telem-tile">
                <div class="label">Predicted</div>
                <div class="value">{fmt_time(race_time)}</div>
            </div>
            <div class="telem-tile">
                <div class="label">Baseline</div>
                <div class="value">{fmt_time(baseline_time)}</div>
            </div>
            <div class="telem-tile">
                <div class="label">Delta</div>
                <div class="value {delta_cls}">{delta_sign}</div>
            </div>
        </div>
        <div class="telem-row">
            <div class="telem-tile">
                <div class="label">Pit Stops</div>
                <div class="value warn">{len(laps_pit)}</div>
            </div>
            <div class="telem-tile">
                <div class="label">Pit Laps</div>
                <div class="value" style="font-size:0.95rem">
                    {", ".join(map(str, laps_pit)) if laps_pit else "—"}
                </div>
            </div>
            <div class="telem-tile">
                <div class="label">Compounds</div>
                <div class="value" style="font-size:0.95rem">
                    {" · ".join(sorted(set(compounds_used)))}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        verdict_cls = "faster" if is_faster else "slower"
        verdict_txt = (f"FASTER BY {abs(delta_val):.2f}s — Excellent strategy!"
                       if is_faster else
                       f"SLOWER BY {abs(delta_val):.2f}s — Adjust your strategy")
        st.markdown(f'<div class="verdict {verdict_cls}">{verdict_txt}</div>',
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── PIT WALL — unified animated figure ───────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # build per-lap telemetry series (real profiles or synthetic fallback)
    if profiles is not None:
        telem_series = build_telemetry_series(
            strategy, lap_compounds, total_laps, track_choice, profiles
        )

    src_label  = "REAL DATA" if profiles is not None else "SIMULATED"
    badge_cls  = "real"      if profiles is not None else "sim"
    st.markdown(
        f'<div class="panel"><div class="panel-header">'
        f'🏎 PIT WALL  '
        f'<span class="data-badge {badge_cls}">{src_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    fig_wall = build_pit_wall_figure(
        track_choice=track_choice,
        lap_times_strategy=lap_times_strategy,
        lap_times_baseline=lap_times_baseline,
        total_laps=total_laps,
        laps_pit=laps_pit,
        lap_compounds=lap_compounds,
        telem_series=telem_series,
        strategy=strategy,
    )
    st.plotly_chart(
        fig_wall,
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown("</div>", unsafe_allow_html=True)
    # # CENTER PANEL: animated track
    # with col_track_vis:
    #     st.markdown('<div class="panel"><div class="panel-header">🗺 Track Visualisation</div>', unsafe_allow_html=True)

    #     # Display animated track with both cars
    #     fig = plot_circuit_animated(
    #         track_choice,
    #         lap_times_strategy=lap_times_strategy,
    #         lap_times_baseline=lap_times_baseline,
    #         total_laps=total_laps,
    #         laps_pit=laps_pit,
    #         tire_strategy=lap_compounds,
    #         tel_profiles=profiles,
    #         lap_compounds=lap_compounds,
    #         strategy=strategy
    #     )

    #     st.plotly_chart(
    #         fig,
    #         width="stretch",
    #         config={"displayModeBar": False}
    #     )
    #     st.markdown("</div>", unsafe_allow_html=True)

    # # RIGHT PANEL: telemetry metrics
    # with col_telem:
    #     delta_val = baseline_time - race_time
    #     is_faster = delta_val > 0

    #     def fmt_time(secs):
    #         h = int(secs // 3600)
    #         m = int((secs % 3600) // 60)
    #         s = secs % 60
    #         return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m}:{s:05.2f}"

    #     delta_cls  = "good" if is_faster else "bad"
    #     delta_sign = f"-{abs(delta_val):.2f}s" if is_faster else f"+{abs(delta_val):.2f}s"

    #     st.markdown('<div class="panel"><div class="panel-header">⚡ Race Metrics</div>',
    #                 unsafe_allow_html=True)
    #     st.markdown(f"""
    #     <div class="telem-row">
    #         <div class="telem-tile">
    #             <div class="label">Predicted</div>
    #             <div class="value">{fmt_time(race_time)}</div>
    #         </div>
    #         <div class="telem-tile">
    #             <div class="label">Baseline</div>
    #             <div class="value">{fmt_time(baseline_time)}</div>
    #         </div>
    #         <div class="telem-tile">
    #             <div class="label">Delta</div>
    #             <div class="value {delta_cls}">{delta_sign}</div>
    #         </div>
    #     </div>
    #     <div class="telem-row">
    #         <div class="telem-tile">
    #             <div class="label">Pit Stops</div>
    #             <div class="value warn">{len(laps_pit)}</div>
    #         </div>
    #         <div class="telem-tile">
    #             <div class="label">Pit Laps</div>
    #             <div class="value" style="font-size:0.95rem">
    #                 {", ".join(map(str, laps_pit)) if laps_pit else "—"}
    #             </div>
    #         </div>
    #         <div class="telem-tile">
    #             <div class="label">Compounds</div>
    #             <div class="value" style="font-size:0.95rem">
    #                 {" · ".join(sorted(set(compounds_used)))}
    #             </div>
    #         </div>
    #     </div>""", unsafe_allow_html=True)

    #     verdict_cls = "faster" if is_faster else "slower"
    #     verdict_txt = (f"FASTER BY {abs(delta_val):.2f}s — Excellent strategy!"
    #                    if is_faster else
    #                    f"SLOWER BY {abs(delta_val):.2f}s — Adjust your strategy")
    #     st.markdown(f'<div class="verdict {verdict_cls}">{verdict_txt}</div>', unsafe_allow_html=True)
    #     st.markdown("</div>", unsafe_allow_html=True)

    # # telemetry dashboard
    # st.markdown('<div class="divider"></div', unsafe_allow_html=True)
    # src_label = "REAL DATA" if profiles is not None else "SIMULATED"
    # st.markdown(
    #     f'<div class="panel"><div class="panel-header">'
    #     f' Pit Wall Telemetry Dashboard  '
    #     f'<span class="data-badge {"real" if profiles is not None else "sim"}">'
    #     f'{src_label}</span></div>',
    #     unsafe_allow_html=True,
    # )

    # fig_telem = race_explain_visualizer(
    #     lap_times=lap_times_strategy,
    #     pits=laps_pit,
    #     lap_times_baseline=lap_times_baseline,
    #     strategy=strategy,
    #     lap_compounds=lap_compounds,
    #     track=track_choice,
    #     tel_profiles=profiles
    # )
    # st.pyplot(fig_telem, use_container_width=True)
    # st.markdown("</div>", unsafe_allow_html=True)




## MAIN ##
initialize_window()
build_ui_structure()