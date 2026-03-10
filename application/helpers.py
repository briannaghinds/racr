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

## PIT WALL VISUALS ##
def _tire_segments(strategy: list[dict], total_laps: int) -> list[dict]:
    segs = []
    for i, stint in enumerate(strategy):
        end = strategy[i + 1]["start_lap"] - 1 if i + 1 < len(strategy) else total_laps
        for lap in range(stint["start_lap"], end + 1):
            segs.append({"lap": lap, "compound": stint["compound"]})
    return segs

def race_explain_visualizer(
    lap_times: list[float],
    pits: list[int],
    lap_times_baseline: list[float] | None = None,
    strategy: list[dict] | None = None,
    lap_compounds: list[str] | None = None,
    track: str | None = None,
    tel_profiles: pd.DataFrame | None = None,
) -> plt.Figure:
    """
    6-panel pit-wall telemetry dashboard.

    Layout
    ──────
    Row 0 (full width) │ Lap time progression + baseline
    Row 1 left         │ Tire degradation by compound
    Row 1 right        │ Cumulative gap vs baseline
    Row 2 left         │ Avg speed per lap  (real data if profiles available)
    Row 2 right        │ Throttle % per lap (real data if profiles available)
    Row 3 left         │ Braking % per lap  (real data if profiles available)
    Row 3 right        │ Gear changes per lap (real data if profiles available)

    Args
        lap_times          : predicted lap times (seconds)
        pits               : lap numbers with pit stops
        lap_times_baseline : baseline lap times; auto-built if None
        strategy           : list of {"start_lap", "compound"} dicts
        lap_compounds      : per-lap compound list (len == total_laps)
        track              : circuit name for profile lookup
        tel_profiles       : DataFrame from load_telemetry_profiles()
    """
    total_laps = len(lap_times)
    if lap_times_baseline is None:
        lap_times_baseline = [np.mean(lap_times)] * total_laps

    laps_x = list(range(1, total_laps + 1))
    gap    = np.cumsum(np.array(lap_times) - np.array(lap_times_baseline)).tolist()

    # ── get telemetry series (real or synthetic) ──────────────────────────
    telem: dict[str, list] = {}
    if tel_profiles is not None and lap_compounds is not None and track and strategy:
        telem = build_telemetry_series(strategy, lap_compounds, total_laps, track, tel_profiles)

    data_source_label = "(real)" if tel_profiles is not None else "(simulated)"

    # ── compound-lap index for deg plot ───────────────────────────────────
    compound_laps:  dict[str, list] = {}
    compound_times: dict[str, list] = {}
    if strategy:
        for s in _tire_segments(strategy, total_laps):
            c = s["compound"]
            compound_laps.setdefault(c, []).append(s["lap"])
            compound_times.setdefault(c, []).append(lap_times[s["lap"] - 1])

    # ── figure setup ──────────────────────────────────────────────────────
    BG   = "#0d0d0d"
    GRID = "#1e1e1e"
    ACC  = "#e10600"
    TXT  = "#cccccc"
    SPD  = "#00bfff"
    THR  = "#44ff88"
    BRK  = "#ff6b35"
    GER  = "#c084fc"

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs  = GridSpec(
        4, 2, figure=fig,
        hspace=0.58, wspace=0.32,
        top=0.93, bottom=0.05, left=0.07, right=0.97,
    )

    ax_lap = fig.add_subplot(gs[0, :])   # row 0 — full width
    ax_deg = fig.add_subplot(gs[1, 0])   # row 1 left
    ax_gap = fig.add_subplot(gs[1, 1])   # row 1 right
    ax_spd = fig.add_subplot(gs[2, 0])   # row 2 left
    ax_thr = fig.add_subplot(gs[2, 1])   # row 2 right
    ax_brk = fig.add_subplot(gs[3, 0])   # row 3 left
    ax_ger = fig.add_subplot(gs[3, 1])   # row 3 right

    def _style(ax, title):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        ax.set_title(title, color=TXT, fontsize=9, fontfamily="monospace", pad=6)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--")

    def _pit_lines(ax, y_max_val=None):
        for p in pits:
            ax.axvline(p, color="#ffaa00", linewidth=0.8, linestyle=":", alpha=0.8)
            if y_max_val:
                ax.text(p + 0.3, y_max_val * 0.995, "PIT",
                        color="#ffaa00", fontsize=6, fontfamily="monospace")

    # ── row 0: lap time ───────────────────────────────────────────────────
    _style(ax_lap, "LAP TIME PROGRESSION  (s)")
    ax_lap.plot(laps_x, lap_times_baseline, color="#555", linewidth=1.2,
                linestyle="--", label="Baseline (degradation model)", zorder=2)
    ax_lap.plot(laps_x, lap_times, color=ACC, linewidth=1.6,
                label="Your Strategy", zorder=3)

    if strategy:
        segs = _tire_segments(strategy, total_laps)
        for i in range(len(segs) - 1):
            c   = segs[i]["compound"]
            col = TIRE_COLORS.get(c, ACC)
            ax_lap.plot(
                [segs[i]["lap"], segs[i + 1]["lap"]],
                [lap_times[segs[i]["lap"] - 1], lap_times[segs[i + 1]["lap"] - 1]],
                color=col, linewidth=2.2, zorder=4,
            )

    _pit_lines(ax_lap, max(lap_times))
    ax_lap.set_xlabel("Lap", fontsize=8)
    ax_lap.set_ylabel("Time (s)", fontsize=8)
    ax_lap.legend(facecolor="#1a1a1a", edgecolor=GRID, labelcolor=TXT, fontsize=8)

    # ── row 1 left: tire degradation ─────────────────────────────────────
    _style(ax_deg, "TIRE DEGRADATION BY COMPOUND")
    if compound_laps:
        for c, lps in compound_laps.items():
            col   = TIRE_COLORS.get(c, "#888")
            times = compound_times[c]
            ax_deg.scatter(lps, times, color=col, s=14, label=c, zorder=3)
            if len(lps) > 1:
                z    = np.polyfit(lps, times, 1)
                xfit = np.linspace(min(lps), max(lps), 60)
                ax_deg.plot(xfit, np.poly1d(z)(xfit), color=col,
                            linewidth=1.2, alpha=0.6)
        ax_deg.legend(facecolor="#1a1a1a", edgecolor=GRID,
                      labelcolor=TXT, fontsize=7)
    else:
        ax_deg.plot(laps_x, lap_times, color=ACC, linewidth=1.2)
    ax_deg.set_xlabel("Lap", fontsize=8)
    ax_deg.set_ylabel("Lap Time (s)", fontsize=8)

    # ── row 1 right: cumulative gap ───────────────────────────────────────
    _style(ax_gap, "CUMULATIVE GAP vs BASELINE  (s)")
    pos_g = [g if g >= 0 else np.nan for g in gap]
    neg_g = [g if g <  0 else np.nan for g in gap]
    ax_gap.plot(laps_x, pos_g, color="#ff4444", linewidth=1.4, label="Slower")
    ax_gap.plot(laps_x, neg_g, color="#44ff88", linewidth=1.4, label="Faster")
    ax_gap.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax_gap.fill_between(laps_x, gap, 0, where=[g >= 0 for g in gap],
                        color="#ff4444", alpha=0.12)
    ax_gap.fill_between(laps_x, gap, 0, where=[g < 0 for g in gap],
                        color="#44ff88", alpha=0.12)
    _pit_lines(ax_gap)
    ax_gap.set_xlabel("Lap", fontsize=8)
    ax_gap.set_ylabel("Gap (s)", fontsize=8)
    ax_gap.legend(facecolor="#1a1a1a", edgecolor=GRID, labelcolor=TXT, fontsize=7)

    # ── row 2 left: avg speed ─────────────────────────────────────────────
    spd = telem.get("avg_speed_kmh", [])
    _style(ax_spd, f"AVG SPEED PER LAP  (km/h)  {data_source_label}")
    if spd:
        ax_spd.fill_between(laps_x, spd, alpha=0.2, color=SPD)
        ax_spd.plot(laps_x, spd, color=SPD, linewidth=1.3)
        _pit_lines(ax_spd)
    ax_spd.set_xlabel("Lap", fontsize=8)
    ax_spd.set_ylabel("km/h", fontsize=8)

    # ── row 2 right: throttle ─────────────────────────────────────────────
    thr_raw = telem.get("throttle_pct", [])
    # profiles store fractions; scale to % for display
    thr = [v * 100 if v <= 1.0 else v for v in thr_raw]
    _style(ax_thr, f"FULL-THROTTLE %  {data_source_label}")
    if thr:
        ax_thr.fill_between(laps_x, thr, alpha=0.25, color=THR)
        ax_thr.plot(laps_x, thr, color=THR, linewidth=1.3)
        ax_thr.set_ylim(0, 110)
        _pit_lines(ax_thr)
    ax_thr.set_xlabel("Lap", fontsize=8)
    ax_thr.set_ylabel("%", fontsize=8)

    # ── row 3 left: braking ───────────────────────────────────────────────
    brk_raw = telem.get("braking_pct", [])
    brk = [v * 100 if v <= 1.0 else v for v in brk_raw]
    _style(ax_brk, f"BRAKING %  {data_source_label}")
    if brk:
        ax_brk.fill_between(laps_x, brk, alpha=0.25, color=BRK)
        ax_brk.plot(laps_x, brk, color=BRK, linewidth=1.3)
        ax_brk.set_ylim(0, 60)
        _pit_lines(ax_brk)
    ax_brk.set_xlabel("Lap", fontsize=8)
    ax_brk.set_ylabel("%", fontsize=8)

    # ── row 3 right: gear changes ─────────────────────────────────────────
    gcg = telem.get("gear_changes", [])
    _style(ax_ger, f"GEAR CHANGES PER LAP  {data_source_label}")
    if gcg:
        ax_ger.bar(laps_x, gcg, color=GER, alpha=0.55, width=0.8)
        ax_ger.plot(laps_x, gcg, color=GER, linewidth=1.0, alpha=0.9)
        _pit_lines(ax_ger)
    ax_ger.set_xlabel("Lap", fontsize=8)
    ax_ger.set_ylabel("Shifts", fontsize=8)

    fig.suptitle(
        "PIT WALL TELEMETRY DASHBOARD",
        color=TXT, fontsize=14, fontfamily="monospace", fontweight="bold",
    )
    return fig

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

            strategy_lap = min(strategy_lap, total_laps - 1)
            denom_s = cumulative_strategy[strategy_lap+1] - cumulative_strategy[strategy_lap]
            lp      = (current_time - cumulative_strategy[strategy_lap]) / denom_s if denom_s else 0.0
            lp      = max(0.0, min(1.0, lp))
            f_idx   = lp * (num_points - 1)
            b_idx   = min(int(f_idx), num_points - 2)
            rem     = f_idx - b_idx
            n_idx   = b_idx + 1
            strat_x = track_df["X"].iloc[b_idx] + rem * (track_df["X"].iloc[n_idx] - track_df["X"].iloc[b_idx])
            strat_y = track_df["Y"].iloc[b_idx] + rem * (track_df["Y"].iloc[n_idx] - track_df["Y"].iloc[b_idx])

            # ghost car position
            baseline_lap = 0
            for i, t in enumerate(cumulative_baseline):
                if current_time >= t: 
                    baseline_lap = i
                else: 
                    break

            baseline_lap = min(baseline_lap, total_laps - 1)
            denom_b = cumulative_baseline[baseline_lap+1] - cumulative_baseline[baseline_lap]
            lp_b    = (current_time - cumulative_baseline[baseline_lap]) / denom_b if denom_b else 0.0
            lp_b    = max(0.0, min(1.0, lp_b))
            f_idx_b = lp_b * (num_points - 1)
            b_idx_b = min(int(f_idx_b), num_points - 2)
            rem_b   = f_idx_b - b_idx_b
            n_idx_b = b_idx_b + 1
            gh_x    = track_df["X"].iloc[b_idx_b] + rem_b * (track_df["X"].iloc[n_idx_b] - track_df["X"].iloc[b_idx_b])
            gh_y    = track_df["Y"].iloc[b_idx_b] + rem_b * (track_df["Y"].iloc[n_idx_b] - track_df["Y"].iloc[b_idx_b])

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