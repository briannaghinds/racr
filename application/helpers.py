"""
@author: Brianna Hinds
Description: helper funcitons for the F1 application
"""
from constants import INPUT_COLS, DEFAULT_VALS, TIRE_COLORS
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go



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


def plot_circuit_animated(track_choice, lap_times_strategy=None, lap_times_baseline=None, total_laps=None, laps_pit=None, tire_strategy=None):
    """
    Create animated track visualization with strategy car and ghost baseline car.
    """
    # load track data 
    df = pd.read_csv("../data/circuit_info.csv")
    track_df = df[(df["track"] == track_choice) & (df["type"]=="track")].sort_values("seq")
    corner_df = df[(df["track"] == track_choice) & (df["type"]=="corner")]
    
    # Calculate bounds for consistent framing
    x_min, x_max = track_df["X"].min(), track_df["X"].max()
    y_min, y_max = track_df["Y"].min(), track_df["Y"].max()
    x_padding = (x_max - x_min) * 0.15
    y_padding = (y_max - y_min) * 0.15

    # Create a wider "Canvas" so the scoreboard has its own corner
    sidebar_x = x_max + (x_max - x_min) * 0.4  # 40% extra space for telemetry
    
    fig = go.Figure()
    
    # add static traces (track, line, corner markers)
    ## TRACK LINE
    fig.add_trace(go.Scatter(
        x=track_df["X"], y=track_df["Y"],
        mode="lines", line=dict(color="#ffffff", width=6),
        hoverinfo="skip", name="Circuit"
    ))
    
    ## START/FINISH LINE
    fig.add_trace(go.Scatter(
        x=[track_df["X"].iloc[0]], y=[track_df["Y"].iloc[0]],
        mode="markers", marker=dict(symbol="star", size=15, color="#00ff00"),
        name="Start/Finish"
    ))
    
    ## CORNER MARKERS
    fig.add_trace(go.Scatter(
        x=corner_df["X"] if not corner_df.empty else [None],
        y=corner_df["Y"] if not corner_df.empty else [None],
        mode="markers+text",
        text=corner_df["corner"] if not corner_df.empty else "",
        textposition="top center",
        textfont=dict(color="#ffaa00", size=8, family="Orbitron"),
        marker=dict(color="#ffaa00", size=4),
        name="Corners"
    ))
    
    # add the dynamic traces (cars)
    ## GHOST CAR
    fig.add_trace(go.Scatter(
        x=[track_df["X"].iloc[0]], y=[track_df["Y"].iloc[0]],
        mode="markers",
        marker=dict(size=16, color="rgba(100, 100, 255, 0.5)", symbol="diamond", line=dict(color="#6666ff", width=2)),
        name="Ghost Car (baseline)",
        hoverinfo="text",
        hovertext="Ghost Car (baseline)"
    ))
    
    # STRATEGY CAR
    fig.add_trace(go.Scatter(
        x=[track_df["X"].iloc[0]], y=[track_df["Y"].iloc[0]],
        mode="markers",
        marker=dict(size=18, color="#e10600", symbol="circle", line=dict(color="#ffffff", width=2)),
        name="Strategy Car",
        hoverinfo="text",
        hovertext="Your Strategy"
    ))

    ## SCOREBOARD
    fig.add_trace(go.Scatter(
        x=[sidebar_x],
        y=[y_max], 
        mode="text",
        text=["Initializing..."],
        textposition="bottom center", # grows text DOWN and LEFT so it stays in view
        textfont=dict(family="Orbitron", size=16, color="#ffffff"),
        showlegend=False,
        name="Telemetry",
        cliponaxis=False
    ))

    # animation logic
    num_points = len(track_df)
    time_steps = 1250 
    frames = []
    tire_colors = TIRE_COLORS
    
    if lap_times_strategy and lap_times_baseline:
        cumulative_strategy = [0] + [sum(lap_times_strategy[:i+1]) for i in range(len(lap_times_strategy))]
        cumulative_baseline = [0] + [sum(lap_times_baseline[:i+1]) for i in range(len(lap_times_baseline))]
        total_race_time = max(cumulative_strategy[-1], cumulative_baseline[-1])
        
        for step in range(time_steps):
            current_time = (step / time_steps) * total_race_time

            # strategy car position
            strategy_lap = 0
            for i, t in enumerate(cumulative_strategy):
                if current_time >= t: 
                    strategy_lap = i
                else: 
                    break

            if strategy_lap < total_laps:
                lp = (current_time - cumulative_strategy[strategy_lap]) / (cumulative_strategy[strategy_lap+1] - cumulative_strategy[strategy_lap])
                f_idx = lp * (num_points - 1)
                b_idx = int(f_idx)
                rem = f_idx - b_idx
                n_idx = (b_idx + 1) % num_points
                strat_x = track_df["X"].iloc[b_idx] + rem * (track_df["X"].iloc[n_idx] - track_df["X"].iloc[b_idx])
                strat_y = track_df["Y"].iloc[b_idx] + rem * (track_df["Y"].iloc[n_idx] - track_df["Y"].iloc[b_idx])
            else:
                strat_x, strat_y = track_df["X"].iloc[0], track_df["Y"].iloc[0]

            # ghost car position
            baseline_lap = 0
            for i, t in enumerate(cumulative_baseline):
                if current_time >= t: 
                    baseline_lap = i
                else: 
                    break

            if baseline_lap < total_laps:
                lp_b = (current_time - cumulative_baseline[baseline_lap]) / (cumulative_baseline[baseline_lap+1] - cumulative_baseline[baseline_lap])
                f_idx_b = lp_b * (num_points - 1)
                rem_b = f_idx_b - int(f_idx_b)
                n_idx_b = (int(f_idx_b) + 1) % num_points
                gh_x = track_df["X"].iloc[int(f_idx_b)] + rem_b * (track_df["X"].iloc[n_idx_b] - track_df["X"].iloc[int(f_idx_b)])
                gh_y = track_df["Y"].iloc[int(f_idx_b)] + rem_b * (track_df["Y"].iloc[n_idx_b] - track_df["Y"].iloc[int(f_idx_b)])
            else:
                gh_x, gh_y = track_df["X"].iloc[0], track_df["Y"].iloc[0]

            # telemetry formatting
            gap = cumulative_baseline[strategy_lap] - cumulative_strategy[strategy_lap]
            current_tire = tire_strategy[strategy_lap].upper() if tire_strategy else "SOFT"
            car_color = tire_colors.get(current_tire, "#e10600")
            status = "IN PIT" if (laps_pit and (strategy_lap + 1) in laps_pit) else "ON TRACK"
            
            # Formatted Scoreboard Text
            scoreboard_text = (
                f"<b>RACE TELEMETRY</b><br>"
                f"LAP: {strategy_lap + 1}/{total_laps}<br>"
                f"GAP: <span style='color:{'#00ff00' if gap > 0 else '#ff0000'}'>{gap:+.2f}s</span><br>"
                f"TIRE: {current_tire}<br>"
                f"STATUS: {status}"
            )

            frames.append(go.Frame(
                data=[
                    go.Scatter(x=[gh_x], y=[gh_y]),  # ghost car 
                    go.Scatter(x=[strat_x], y=[strat_y], marker=dict(color=car_color)),  # strategy car
                    go.Scatter(x=[sidebar_x], y=[y_max], text=[scoreboard_text]) # Scoreboard (Fixed Position)
                ],
                traces=[3, 4, 5],
                name=str(step),
            ))
    
    fig.frames = frames
    
    # final layout and animation settings
    fig.update_layout(
        xaxis=dict(visible=False, fixedrange=True, range=[x_min - x_padding, sidebar_x + x_padding]),
        yaxis=dict(visible=False, fixedrange=True, scaleanchor="x", scaleratio=1, range=[y_min - y_padding, y_max + y_padding]),
        margin=dict(l=20, r=100, t=50, b=20),
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        height=750,
        showlegend=True,
        legend=dict(x=0.02, y=0.02, bgcolor="rgba(20, 20, 20, 0.8)", bordercolor="#e10600", borderwidth=2,font=dict(color="#ffffff", family="Rajdhani")),
        updatemenus=[{
            "type": "buttons", "direction": "right", "x": 0.5, "y": -0.05, "xanchor": "center", "font": {"color": "#669bbc", "family": "Orbitron"},
            "buttons": [
                {
                    "label": "Play", "method": "animate", 
                    "args": [None, {"frame": {"duration": 100, "redraw": False}, "fromcurrent": True, "transition": {"duration": 100, "easing": "linear"}}]
                },
                {
                    "label": "Pause", "method": "animate", 
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }
            ]
        }]
    )
    
    return fig

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