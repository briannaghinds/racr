import streamlit as st
from helpers import data_cleaning, baseline_value, race_explain_visualizer
from constants import TRACKS, PIT_STOP_LOSS
import matplotlib.pyplot as plt
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
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for F1 dashboard aesthetic
    st.markdown("""
        <style>
        /* Import F1-style font */
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;900&display=swap');
        
        /* Global styles */
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            font-family: 'Rajdhani', sans-serif;
        }
        
        /* Main title styling */
        h1 {
            font-family: 'Orbitron', monospace;
            font-weight: 900;
            font-size: 3.5rem !important;
            background: linear-gradient(90deg, #e10600 0%, #ff1e00 50%, #e10600 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            letter-spacing: 4px;
            text-align: center;
            margin-bottom: 0.5rem !important;
            text-shadow: 0 0 30px rgba(225, 6, 0, 0.3);
        }
        
        /* Subtitle */
        .subtitle {
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.2rem;
            color: #888;
            text-align: center;
            letter-spacing: 2px;
            margin-bottom: 2rem;
            text-transform: uppercase;
        }
        
        /* Section headers */
        h2, h3 {
            font-family: 'Orbitron', monospace;
            color: #e10600;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 700;
        }
        
        /* Dashboard panels */
        .dashboard-panel {
            background: rgba(20, 20, 20, 0.8);
            border: 2px solid #333;
            border-left: 4px solid #e10600;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }
        
        /* Metric cards */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
            border: 2px solid #333;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        div[data-testid="metric-container"] > label {
            font-family: 'Orbitron', monospace;
            font-size: 0.9rem !important;
            color: #888 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div[data-testid="metric-container"] > div {
            font-family: 'Orbitron', monospace;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #e10600 !important;
        }
        
        /* Selectbox and inputs */
        .stSelectbox, .stNumberInput, .stSlider {
            font-family: 'Rajdhani', sans-serif;
        }
        
        .stSelectbox label, .stNumberInput label, .stSlider label {
            font-family: 'Orbitron', monospace;
            color: #e10600 !important;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 1px;
        }
        
        /* Buttons */
        .stButton button {
            font-family: 'Orbitron', monospace;
            background: linear-gradient(135deg, #e10600 0%, #ff1e00 100%);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.75rem 2rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(225, 6, 0, 0.3);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(225, 6, 0, 0.5);
        }
        
        /* Checkbox */
        .stCheckbox label {
            font-family: 'Rajdhani', sans-serif;
            color: #ccc;
            font-size: 1rem;
        }
        
        /* Toggle */
        .stToggle label {
            font-family: 'Orbitron', monospace;
            color: #e10600 !important;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        /* Telemetry bar */
        .telemetry-bar {
            background: #0a0a0a;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            display: flex;
            justify-content: space-around;
            align-items: center;
        }
        
        .telemetry-item {
            text-align: center;
        }
        
        .telemetry-label {
            font-family: 'Rajdhani', sans-serif;
            color: #888;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .telemetry-value {
            font-family: 'Orbitron', monospace;
            color: #e10600;
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        /* Track status indicator */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff00;
            box-shadow: 0 0 10px #00ff00;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Warning text */
        .warning-text {
            color: #ffaa00;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 600;
        }
        
        /* Stint badge */
        .stint-badge {
            display: inline-block;
            background: #e10600;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-family: 'Orbitron', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 1px;
            margin-right: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

## VISUALIZATION ##
def plot_circuit_animated(
    track_choice,
    show_corners,
    lap_times_strategy=None,
    lap_times_baseline=None,
    total_laps=None,
    laps_pit=None,
    auto_play=True
):
    """
    Create animated track visualization with strategy car and ghost baseline car.
    Animation is continuous (loops).
    
    Args:
        track_choice: Name of the track
        show_corners: Boolean to show corner markers
        lap_times_strategy: List of lap times for the strategy car
        lap_times_baseline: List of lap times for the baseline (ghost) car
        total_laps: Total number of laps in the race
        laps_pit: List of laps where pit stops occur
        auto_play: Boolean to auto-play animation
    """
    df = pd.read_csv("../data/circuit_info.csv")
    track_df = df[(df["track"] == track_choice) & (df["type"]=="track")].sort_values("seq")
    corner_df = df[(df["track"] == track_choice) & (df["type"]=="corner")]
    
    # Calculate bounds with padding to prevent cutoff
    x_min, x_max = track_df["X"].min(), track_df["X"].max()
    y_min, y_max = track_df["Y"].min(), track_df["Y"].max()
    x_padding = (x_max - x_min) * 0.15
    y_padding = (y_max - y_min) * 0.15
    
    fig = go.Figure()
    
    # Add the track line
    fig.add_trace(go.Scatter(
        x=track_df["X"],
        y=track_df["Y"],
        mode="lines",
        line=dict(color="#ffffff", width=8),
        hoverinfo="skip",
        name="Track"
    ))
    
    # Add start/finish line
    start_x, start_y = track_df["X"].iloc[0], track_df["Y"].iloc[0]
    fig.add_trace(go.Scatter(
        x=[start_x],
        y=[start_y],
        mode="markers",
        marker=dict(
            symbol="star",
            size=20,
            color="#00ff00",
            line=dict(color="#ffffff", width=2)
        ),
        hoverinfo="text",
        hovertext="START/FINISH",
        name="Start/Finish"
    ))
    
    # Define corners if requested
    if show_corners and not corner_df.empty:
        fig.add_trace(go.Scatter(
            x=corner_df["X"],
            y=corner_df["Y"],
            mode="markers+text",
            text=corner_df["corner"],
            textposition="top center",
            textfont=dict(color="#ffaa00", size=10, family="Orbitron"),
            marker=dict(color="#ffaa00", size=8, symbol="circle"),
            hoverinfo="skip",
            name="Corners"
        ))
    
    # Ghost car (baseline) - semi-transparent
    fig.add_trace(go.Scatter(
        x=[track_df["X"].iloc[0]],
        y=[track_df["Y"].iloc[0]],
        mode="markers",
        marker=dict(
            size=16,
            color="rgba(100, 100, 255, 0.5)",
            symbol="diamond",
            line=dict(color="#6666ff", width=2)
        ),
        name="Baseline Ghost",
        hoverinfo="text",
        hovertext="Ghost Car (Baseline)"
    ))
    
    # Strategy car (main) - solid red
    fig.add_trace(go.Scatter(
        x=[track_df["X"].iloc[0]],
        y=[track_df["Y"].iloc[0]],
        mode="markers",
        marker=dict(
            size=18,
            color="#e10600",
            symbol="circle",
            line=dict(color="#ffffff", width=2)
        ),
        name="Strategy Car",
        hoverinfo="text",
        hovertext="Your Strategy"
    ))
    
    # Create frames for animation
    num_points = len(track_df)
    frames = []
    
    # If we have lap time data, calculate position based on accumulated time
    if lap_times_strategy is not None and lap_times_baseline is not None and total_laps is not None:
        # Calculate cumulative times
        cumulative_strategy = [0] + [sum(lap_times_strategy[:i+1]) for i in range(len(lap_times_strategy))]
        cumulative_baseline = [0] + [sum(lap_times_baseline[:i+1]) for i in range(len(lap_times_baseline))]
        
        # Total race time
        total_race_time = max(cumulative_strategy[-1], cumulative_baseline[-1])
        
        # Create time steps for smooth animation
        time_steps = 200  # Number of animation frames
        
        for step in range(time_steps):
            current_time = (step / time_steps) * total_race_time
            
            # Find strategy car position
            strategy_lap = 0
            for i, t in enumerate(cumulative_strategy):
                if current_time >= t:
                    strategy_lap = i
                else:
                    break
            
            # Interpolate position within lap
            if strategy_lap < total_laps:
                lap_start_time = cumulative_strategy[strategy_lap]
                lap_end_time = cumulative_strategy[strategy_lap + 1] if strategy_lap + 1 < len(cumulative_strategy) else total_race_time
                lap_progress = (current_time - lap_start_time) / (lap_end_time - lap_start_time) if lap_end_time > lap_start_time else 0
                strategy_idx = int(lap_progress * num_points) % num_points
            else:
                strategy_idx = 0
            
            # Find baseline car position
            baseline_lap = 0
            for i, t in enumerate(cumulative_baseline):
                if current_time >= t:
                    baseline_lap = i
                else:
                    break
            
            # Interpolate position within lap
            if baseline_lap < total_laps:
                lap_start_time = cumulative_baseline[baseline_lap]
                lap_end_time = cumulative_baseline[baseline_lap + 1] if baseline_lap + 1 < len(cumulative_baseline) else total_race_time
                lap_progress = (current_time - lap_start_time) / (lap_end_time - lap_start_time) if lap_end_time > lap_start_time else 0
                baseline_idx = int(lap_progress * num_points) % num_points
            else:
                baseline_idx = 0
            
            # Check if strategy car is in pit
            in_pit = laps_pit is not None and strategy_lap + 1 in laps_pit
            pit_marker = " 🔧 PIT" if in_pit else ""
            
            frames.append(go.Frame(
                data=[
                    # Ghost car
                    go.Scatter(
                        x=[track_df["X"].iloc[baseline_idx]],
                        y=[track_df["Y"].iloc[baseline_idx]],
                        hovertext=f"Ghost Car - Lap {baseline_lap + 1}/{total_laps}"
                    ),
                    # Strategy car
                    go.Scatter(
                        x=[track_df["X"].iloc[strategy_idx]],
                        y=[track_df["Y"].iloc[strategy_idx]],
                        hovertext=f"Strategy Car - Lap {strategy_lap + 1}/{total_laps}{pit_marker}"
                    )
                ],
                traces=[3, 4],  # Update only the car traces
                name=str(step)
            ))
    else:
        # Simple continuous loop animation without lap data
        for i in range(num_points):
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=[track_df["X"].iloc[i]], y=[track_df["Y"].iloc[i]]),
                    go.Scatter(x=[track_df["X"].iloc[i]], y=[track_df["Y"].iloc[i]])
                ],
                traces=[3, 4],
                name=str(i)
            ))
    
    fig.frames = frames
    
    # Animation controls
    updatemenus = [{
        "type": "buttons",
        "showactive": True,
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 50, "redraw": True},
                        "fromcurrent": True,
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ]
            }
        ],
        "x": 0.5,
        "xanchor": "center",
        "y": -0.05,
        "yanchor": "top",
        "bgcolor": "#1a1a1a",
        "bordercolor": "#e10600",
        "borderwidth": 2,
        "font": {"color": "#ffffff", "family": "Orbitron"}
    }]
    
    fig.update_layout(
        updatemenus=updatemenus,
        xaxis=dict(
            visible=False,
            fixedrange=True,
            range=[x_min - x_padding, x_max + x_padding]
        ),
        yaxis=dict(
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
            range=[y_min - y_padding, y_max + y_padding]
        ),
        margin=dict(l=20, r=20, t=20, b=60),
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(20, 20, 20, 0.8)",
            bordercolor="#e10600",
            borderwidth=2,
            font=dict(color="#ffffff", family="Rajdhani")
        ),
        autosize=True,
        height=600
    )
    
    return fig

## SIMULATION LOGIC ##
def simulate_race(strategy, track, track_length, total_laps, model):
    """
    Simulates a sequence of lap predictions based on strategy decisions.
    
    PREDICTION LOGIC REVIEW:
    - Each lap increments tire_age by 1
    - When a pit stop occurs:
      * PIT_STOP_LOSS (time penalty) is added to race_time
      * Tire compound changes to new compound
      * Tire age resets to 0
    - Features are built for each lap including:
      * track name
      * compound (one-hot encoded)
      * tire_age and tire_age_squared (degradation modeling)
      * circuit_length
    - Model predicts lap time for current state
    - Lap time is accumulated into race_time
    
    Args:
        strategy: list of dicts with 'start_lap' and 'compound' keys
        track: circuit name
        track_length: circuit length in km
        total_laps: total laps in the race
        model: trained XGBoost lap-time predictor
    
    Returns:
        race_time: total race time in seconds
        lap_times: list of predicted lap times
        laps_pit: list of lap numbers where pit stops occurred
    """
    # Initialize
    current_compound = strategy[0]["compound"]
    next_pit_index = 1
    tire_age = 0
    race_time = 0.0
    lap_times = []
    laps_pit = []
    
    for lap in range(1, total_laps + 1):
        # Check if we need to pit at this lap
        if next_pit_index < len(strategy) and lap == strategy[next_pit_index]["start_lap"]:
            # Pit stop occurs
            current_compound = strategy[next_pit_index]["compound"]
            race_time += PIT_STOP_LOSS
            tire_age = 0  # Fresh tires
            next_pit_index += 1
            laps_pit.append(lap)
        else:
            # No pit stop, age the tires
            tire_age += 1
        
        # Build features for prediction
        # Note: The compound needs to be one-hot encoded
        new_feature_vals = pd.DataFrame(data={
            "track": [track],
            f"compound_{current_compound}": [1],
            "tire_age": [tire_age],
            "tire_age_squared": [tire_age ** 2],
            "circuit_length(km)": [track_length]
        })
        
        features = data_cleaning(new_feature_vals)
        
        # Predict lap time
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
    st.markdown("<h1>🏎️ RACE STRATEGY SIMULATOR</h1>", unsafe_allow_html=True)
    st.markdown('<div class="subtitle">POWERED BY MACHINE LEARNING TELEMETRY</div>', unsafe_allow_html=True)
    
    # Track selection
    col_track, col_laps, col_corners = st.columns([3, 1, 1])
    
    with col_track:
        track_choice = st.selectbox(
            "SELECT CIRCUIT",
            TRACKS,
            index=0,
            help="Choose the Grand Prix circuit"
        )
    
    # Pull track data
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
    
    with col_corners:
        show_corners = st.checkbox("Show Corners", value=True)
    
    # Main dashboard layout
    col1, col2 = st.columns([3, 2])
    
    # LEFT PANEL - Track Visualization
    with col1:
        st.markdown("### 📡 TRACK TELEMETRY")
        
        # Build strategy first to pass to animation
        with col2:
            st.markdown("### ⚙️ STRATEGY CONFIGURATION")
            
            # Number of stints
            num_stints = st.number_input(
                "NUMBER OF STINTS",
                min_value=1,
                max_value=4,
                value=2,
                help="A stint is a continuous period between pit stops"
            )
            
            # Build strategy
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
            show_corners,
            lap_times_strategy=lap_times_strategy,
            lap_times_baseline=lap_times_baseline,
            total_laps=total_laps,
            laps_pit=laps_pit,
            auto_play=True
        )
        
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )
        
        # Track status
        st.markdown(f"""
            <div class="telemetry-bar">
                <div class="telemetry-item">
                    <div class="telemetry-label">Circuit Length</div>
                    <div class="telemetry-value">{track_length:.2f} km</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">Pit Stops</div>
                    <div class="telemetry-value">{len(laps_pit)}</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">Status</div>
                    <div class="telemetry-value"><span class="status-indicator"></span> LIVE</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # RIGHT PANEL - Strategy builder and results (already rendered above)
    with col2:
        # Results section
        st.markdown("### 📊 PERFORMANCE ANALYSIS")
        
        # Calculate delta
        delta_val = baseline_time - race_time
        is_faster = delta_val > 0
        delta_display = f"+{abs(delta_val):.2f}s" if not is_faster else f"-{abs(delta_val):.2f}s"
        
        # Format times
        baseline_formatted = time.strftime("%H:%M:%S", time.gmtime(baseline_time))
        predicted_formatted = time.strftime("%H:%M:%S", time.gmtime(race_time))
        
        # Display metrics in engineer style
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                "BASELINE TIME",
                baseline_formatted,
                help="Flat average lap time, no pit stops"
            )
        
        with metric_col2:
            st.metric(
                "PREDICTED TIME",
                predicted_formatted,
                delta=delta_display,
                delta_color="inverse",
                help="Your strategy's predicted race time"
            )
        
        # Performance verdict
        if is_faster:
            st.success(f"✅ **FASTER BY {abs(delta_val):.2f} SECONDS** - Excellent strategy!")
        else:
            st.error(f"⚠️ **SLOWER BY {abs(delta_val):.2f} SECONDS** - Consider adjusting your strategy")
        
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