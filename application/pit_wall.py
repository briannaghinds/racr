"""
build_pit_wall_figure()
───────────────────────
Single Plotly figure that contains the track map AND all four telemetry
panels in one unified animation.  Every frame advances:

  • Both car positions on the track map
  • A growing lap-time line  (panel 1)
  • A growing gap line       (panel 2)
  • A growing speed line     (panel 3)
  • Throttle + brake gauges  (panel 4)
  • A vertical cursor line across all four panels
  • Live tile annotations showing current-lap values

The Plotly slider and Play/Pause buttons both drive the same frame list,
so scrubbing and auto-play are always in sync.

Layout  (using make_subplots with specs)
───────
Row 1, col 1  (rowspan 2) │  Track map
Row 1, col 2              │  Panel 1 — Lap time  (line builds)
Row 1, col 3              │  Panel 2 — Gap vs baseline  (line builds)
Row 2, col 2              │  Panel 3 — Avg speed  (line builds)
Row 2, col 3              │  Panel 4 — Throttle / brake gauges

Trace index map (fixed, never changes order)
────────────────────────────────────────────
 0  track line          (static)
 1  start/finish marker (static)
 2  corner markers      (static)
 3  ghost car           (animated)
 4  strategy car        (animated)
 5  lap-time full trace (static reference, grey)
 6  lap-time built line (animated, compound-coloured segments handled via
                         colour array update)
 7  gap full trace      (static reference, grey)
 8  gap built line      (animated)
 9  speed full trace    (static, grey)
10  speed built line    (animated)
11  throttle gauge bar  (animated)
12  brake gauge bar     (animated)
13  cursor line panel 1 (animated) — shared x across panels via annotation
    (annotations are updated per-frame via layout patch)

Annotations (updated per frame via layout.annotations)
  ann[0]  cursor line panel 1  (lap-time chart)
  ann[1]  cursor line panel 2  (gap chart)
  ann[2]  cursor line panel 3  (speed chart)
  ann[3]  live tile: LAP N/TOTAL
  ann[4]  live tile: gap value
  ann[5]  live tile: speed value
  ann[6]  live tile: throttle %
  ann[7]  live tile: compound / tire color
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
def _data(f): return os.path.join(_DATA, f)

try:
    from constants import TIRE_COLORS
except ImportError:
    TIRE_COLORS = {
        "SOFT": "#e8002d", "MEDIUM": "#ffd700", "HARD": "#eeeeee",
        "INTERMEDIATE": "#39b54a", "WET": "#0067ff",
    }

# ── colour palette ─────────────────────────────────────────────────────────
BG       = "#0a0a0a"
GRID     = "#1c1c1c"
TXT      = "#cccccc"
DIM      = "#444444"
ACC      = "#e10600"
SPD_COL  = "#00bfff"
THR_COL  = "#44ff88"
BRK_COL  = "#ff6b35"
GAP_POS  = "#ff4444"   # slower than baseline
GAP_NEG  = "#44ff88"   # faster than baseline


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_float_list(series) -> list:
    """Convert pandas Series or list to plain Python float list."""
    if hasattr(series, "tolist"):
        return [float(v) for v in series.tolist()]
    return [float(v) for v in series]


def _cumulative_times(lap_times: list[float]) -> list[float]:
    """[0, t1, t1+t2, ...]  length = total_laps + 1"""
    cum = [0.0]
    for t in lap_times:
        cum.append(cum[-1] + t)
    return cum


def _car_pos(current_time: float,
             cum: list[float],
             total_laps: int,
             track_x: list, track_y: list) -> tuple[float, float, int]:
    """Interpolate car (x, y, lap_index) at current_time.

    cum has length total_laps + 1: [0, t1, t1+t2, ..., total_race_time].
    We find which lap the car is currently on, then interpolate its
    position along the track spline for the fractional progress within
    that lap.

    Guards:
      - lap clamped to [0, total_laps-1] so cum[lap+1] is always valid
      - frac clamped to [0, 1] to absorb floating-point overshoot
      - bi clamped to [0, n-2] so both bi and ni=(bi+1) are valid indices
    """
    n = len(track_x)
    lap = 0
    for i, t in enumerate(cum):
        if current_time >= t:
            lap = i
        else:
            break
    # clamp: cum has total_laps+1 entries (indices 0..total_laps)
    # lap+1 must be a valid index, so lap <= total_laps-1
    lap = min(lap, total_laps - 1)

    denom = cum[lap + 1] - cum[lap]
    frac  = (current_time - cum[lap]) / denom if denom else 0.0
    frac  = max(0.0, min(1.0, frac))   # clamp float overshoot

    fi = frac * (n - 1)
    bi = min(int(fi), n - 2)           # clamp so ni = bi+1 is always valid
    rem = fi - bi
    ni  = bi + 1                       # no modulo needed — bi <= n-2

    x = track_x[bi] + rem * (track_x[ni] - track_x[bi])
    y = track_y[bi] + rem * (track_y[ni] - track_y[bi])
    return float(x), float(y), lap


def _axis_cfg() -> dict:
    """Shared dark-theme axis style for telemetry panels."""
    return dict(
        showgrid=True, gridcolor=GRID, gridwidth=0.5,
        zeroline=False, showline=True, linecolor=GRID,
        tickfont=dict(family="Rajdhani", color=TXT, size=9),
        title_font=dict(family="Rajdhani", color=TXT, size=9),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_pit_wall_figure(
    track_choice: str,
    lap_times_strategy: list[float],
    lap_times_baseline: list[float],
    total_laps: int,
    laps_pit: list[int],
    lap_compounds: list[str],
    telem_series: dict[str, list],
    strategy: list[dict],
    time_steps: int = 1250,
) -> go.Figure:
    """
    Build the unified pit-wall figure.

    Parameters
    ──────────
    track_choice        : circuit name (must exist in circuit_info.csv)
    lap_times_strategy  : per-lap predicted times (seconds)
    lap_times_baseline  : per-lap baseline times (seconds)
    total_laps          : race distance in laps
    laps_pit            : list of lap numbers where pit stops occur
    lap_compounds       : per-lap compound string (len == total_laps)
    telem_series        : dict from build_telemetry_series() —
                          keys: avg_speed_kmh, throttle_pct, braking_pct, …
    strategy            : list of {"start_lap", "compound"} dicts
    time_steps          : animation resolution (frames); 300 is smooth enough
                          without making the figure too large to serialise
    """

    # ── 1. load track geometry ─────────────────────────────────────────────
    circuit_path = _data("circuit_info.csv")
    df_circ  = pd.read_csv(circuit_path)
    track_df = (df_circ[(df_circ["track"] == track_choice) &
                         (df_circ["type"]  == "track")]
                .sort_values("seq"))
    corner_df = df_circ[(df_circ["track"] == track_choice) &
                         (df_circ["type"]  == "corner")]

    track_x = _to_float_list(track_df["X"])
    track_y = _to_float_list(track_df["Y"])
    num_pts  = len(track_x)

    # ── 2. derived data series ────────────────────────────────────────────
    laps_x    = list(range(1, total_laps + 1))
    gap_series = list(np.cumsum(
        np.array(lap_times_strategy) - np.array(lap_times_baseline)
    ))
    speed_series  = telem_series.get("avg_speed_kmh", [0.0] * total_laps)
    throttle_ser  = telem_series.get("throttle_pct",  [0.5] * total_laps)
    brake_ser     = telem_series.get("braking_pct",   [0.2] * total_laps)

    # scale fractions → percent if needed
    throttle_pct = [v * 100 if v <= 1.0 else v for v in throttle_ser]
    brake_pct    = [v * 100 if v <= 1.0 else v for v in brake_ser]

    cum_s = _cumulative_times(lap_times_strategy)
    cum_b = _cumulative_times(lap_times_baseline)
    total_race_time = max(cum_s[-1], cum_b[-1])

    # Y-axis ranges (computed once, used for cursor line extent)
    lt_min = min(min(lap_times_strategy), min(lap_times_baseline)) * 0.998
    lt_max = max(max(lap_times_strategy), max(lap_times_baseline)) * 1.002
    gap_min = min(gap_series) * 1.05 if min(gap_series) < 0 else min(gap_series) * 0.95
    gap_max = max(gap_series) * 1.05 if max(gap_series) > 0 else 1.0
    spd_min = min(speed_series) * 0.98 if speed_series else 0
    spd_max = max(speed_series) * 1.02 if speed_series else 350

    # ── 3. subplot layout ────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "xy", "rowspan": 2}, {"type": "xy"}, {"type": "xy"}],
            [None,                          {"type": "xy"}, {"type": "xy"}],
        ],
        column_widths=[0.38, 0.31, 0.31],
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )

    # ── 4. static traces ─────────────────────────────────────────────────
    # trace 0 — track outline
    fig.add_trace(go.Scatter(
        x=track_x, y=track_y,
        mode="lines",
        line=dict(color="#ffffff", width=5),
        hoverinfo="skip", showlegend=False, name="Circuit",
    ), row=1, col=1)

    # trace 1 — start/finish
    fig.add_trace(go.Scatter(
        x=[track_x[0]], y=[track_y[0]],
        mode="markers",
        marker=dict(symbol="star", size=14, color="#00ff00"),
        hoverinfo="skip", showlegend=False, name="S/F",
    ), row=1, col=1)

    # trace 2 — corner markers
    cx = _to_float_list(corner_df["X"]) if not corner_df.empty else [None]
    cy = _to_float_list(corner_df["Y"]) if not corner_df.empty else [None]
    ct = corner_df["corner"].tolist()    if not corner_df.empty else []
    fig.add_trace(go.Scatter(
        x=cx, y=cy,
        mode="markers+text",
        text=ct,
        textposition="top center",
        textfont=dict(color="#ffaa00", size=7, family="Orbitron"),
        marker=dict(color="#ffaa00", size=3),
        hoverinfo="skip", showlegend=False, name="Corners",
    ), row=1, col=1)

    # trace 3 — ghost car (baseline)
    fig.add_trace(go.Scatter(
        x=[track_x[0]], y=[track_y[0]],
        mode="markers",
        marker=dict(size=14, color="rgba(100,100,255,0.5)",
                    symbol="diamond", line=dict(color="#6666ff", width=2)),
        name="Baseline", showlegend=True,
    ), row=1, col=1)

    # trace 4 — strategy car
    fig.add_trace(go.Scatter(
        x=[track_x[0]], y=[track_y[0]],
        mode="markers",
        marker=dict(size=17, color=ACC, symbol="circle",
                    line=dict(color="#ffffff", width=2)),
        name="Your Strategy", showlegend=True,
    ), row=1, col=1)

    # ── Panel 1: Lap time ─────────────────────────────────────────────────
    # trace 5 — full lap-time reference (static, dimmed)
    fig.add_trace(go.Scatter(
        x=laps_x, y=lap_times_baseline,
        mode="lines",
        line=dict(color=DIM, width=1, dash="dot"),
        name="Baseline pace", showlegend=True,
    ), row=1, col=2)

    # trace 6 — strategy lap-time line (builds lap by lap)
    fig.add_trace(go.Scatter(
        x=[laps_x[0]], y=[lap_times_strategy[0]],
        mode="lines+markers",
        line=dict(color=ACC, width=2),
        marker=dict(size=4, color=ACC),
        name="Strategy pace", showlegend=True,
    ), row=1, col=2)

    # ── Panel 2: Gap ──────────────────────────────────────────────────────
    # trace 7 — zero reference
    fig.add_trace(go.Scatter(
        x=[laps_x[0], laps_x[-1]], y=[0, 0],
        mode="lines",
        line=dict(color=DIM, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=3)

    # trace 8 — gap builds lap by lap
    fig.add_trace(go.Scatter(
        x=[laps_x[0]], y=[gap_series[0]],
        mode="lines",
        line=dict(color=GAP_NEG if gap_series[0] < 0 else GAP_POS, width=2),
        fill="tozeroy",
        fillcolor="rgba(68,255,136,0.08)" if gap_series[0] < 0 else "rgba(255,68,68,0.08)",
        name="Gap to baseline", showlegend=True,
    ), row=1, col=3)

    # ── Panel 3: Speed ────────────────────────────────────────────────────
    # trace 9 — full speed reference (static, dimmed)
    fig.add_trace(go.Scatter(
        x=laps_x, y=speed_series,
        mode="lines",
        line=dict(color=DIM, width=1),
        hoverinfo="skip", showlegend=False,
    ), row=2, col=2)

    # trace 10 — speed builds lap by lap
    fig.add_trace(go.Scatter(
        x=[laps_x[0]], y=[speed_series[0]],
        mode="lines+markers",
        line=dict(color=SPD_COL, width=2),
        marker=dict(size=4, color=SPD_COL),
        name="Avg speed", showlegend=True,
    ), row=2, col=2)

    # ── Panel 4: Throttle / Brake gauge bars ─────────────────────────────
    # trace 11 — throttle bar (horizontal bar chart with 1 bar)
    fig.add_trace(go.Bar(
        x=[throttle_pct[0]],
        y=["THROTTLE"],
        orientation="h",
        marker=dict(color=THR_COL, opacity=0.85),
        name="Throttle %",
        showlegend=False,
        width=0.35,
    ), row=2, col=3)

    # trace 12 — brake bar
    fig.add_trace(go.Bar(
        x=[brake_pct[0]],
        y=["BRAKE"],
        orientation="h",
        marker=dict(color=BRK_COL, opacity=0.85),
        name="Brake %",
        showlegend=False,
        width=0.35,
    ), row=2, col=3)

    # ── 5. pit stop vertical lines (static shapes) ────────────────────────
    # xref uses the panel data axis so x0/x1 are lap numbers.
    # yref must be "<axis> domain" so y0=0/y1=1 = full subplot height.
    shapes = []
    for p in laps_pit:
        for panel_xref, yref_domain in [
            ("x2", "y2 domain"),
            ("x3", "y3 domain"),
            ("x4", "y4 domain"),
        ]:
            shapes.append(dict(
                type="line",
                xref=panel_xref,
                yref=yref_domain,
                x0=p, x1=p,
                y0=0, y1=1,
                line=dict(color="#ffaa00", width=1.2, dash="dot"),
                layer="below",
            ))

    # ── 6. cursor shapes + live-tile annotations ─────────────────────────
    # Cursor lines are shapes (not annotations) — Plotly shapes with
    # yref="<axis> domain" reliably span the full subplot height.
    # Three cursor shapes are appended AFTER the pit-stop shapes so their
    # indices are predictable: len(shapes), len(shapes)+1, len(shapes)+2.
    # Each frame updates all shapes via layout.shapes (full replacement).

    def _cursor_shape(x_val, xref, yref_domain):
        return dict(
            type="line",
            xref=xref, yref=yref_domain,
            x0=x_val, x1=x_val,
            y0=0, y1=1,
            line=dict(color="#ffffff", width=1.5, dash="solid"),
            opacity=0.55,
            layer="above",
        )

    cursor_shapes = [
        _cursor_shape(1, "x2", "y2 domain"),
        _cursor_shape(1, "x3", "y3 domain"),
        _cursor_shape(1, "x4", "y4 domain"),
    ]
    all_static_shapes = shapes + cursor_shapes  # pit shapes first, then cursors
    # Remember pit shape count so frame builder can rebuild correctly
    n_pit_shapes = len(shapes)

    def _tile_ann(x, y, text, xref="paper", yref="paper", size=18):
        return dict(
            x=x, y=y, xref=xref, yref=yref,
            text=text, showarrow=False,
            font=dict(family="Orbitron", size=size, color=TXT),
            align="center",
            bgcolor="rgba(20,20,20,0.7)",
            bordercolor=GRID,
            borderwidth=1,
            borderpad=4,
        )

    # Live tile annotations (paper coords, updated every frame)
    # Indices 0-4
    init_annotations = [
        _tile_ann(0.20, 0.97, "LAP 1"),
        _tile_ann(0.20, 0.91, "GAP —"),
        _tile_ann(0.20, 0.85, "— km/h"),
        _tile_ann(0.20, 0.79, "THROTTLE —"),
        _tile_ann(0.20, 0.73, "—"),
    ]

    # Static panel title annotations — combined with init_annotations in
    # update_layout so they're never overwritten by frame annotation patches
    # (frame patches only replace the 5 live-tile indices 0-4).
    panel_titles = [
        dict(x=0.445, y=1.02, text="LAP TIME",         showarrow=False,
             xref="paper", yref="paper",
             font=dict(family="Orbitron", size=9, color=ACC)),
        dict(x=0.755, y=1.02, text="GAP vs BASELINE",  showarrow=False,
             xref="paper", yref="paper",
             font=dict(family="Orbitron", size=9, color=ACC)),
        dict(x=0.445, y=0.47, text="AVG SPEED",        showarrow=False,
             xref="paper", yref="paper",
             font=dict(family="Orbitron", size=9, color=ACC)),
        dict(x=0.755, y=0.47, text="THROTTLE / BRAKE", showarrow=False,
             xref="paper", yref="paper",
             font=dict(family="Orbitron", size=9, color=ACC)),
    ]

    # ── 7. build frames ───────────────────────────────────────────────────
    frames = []

    for step in range(time_steps):
        ct = (step / (time_steps - 1)) * total_race_time

        # car positions
        sx, sy, s_lap = _car_pos(ct, cum_s, total_laps, track_x, track_y)
        gx, gy, _     = _car_pos(ct, cum_b, total_laps, track_x, track_y)

        # lap index (0-based) — clamp
        li = min(s_lap, total_laps - 1)

        # current values
        compound   = lap_compounds[li] if li < len(lap_compounds) else "MEDIUM"
        car_color  = TIRE_COLORS.get(compound.upper(), ACC)
        gap_val    = gap_series[li]
        spd_val    = speed_series[li]  if li < len(speed_series)  else 0.0
        thr_val    = throttle_pct[li]  if li < len(throttle_pct)  else 0.0
        brk_val    = brake_pct[li]     if li < len(brake_pct)     else 0.0
        gap_color  = GAP_NEG if gap_val < 0 else GAP_POS
        lap_num    = li + 1

        # built series up to current lap (inclusive)
        built_x   = laps_x[:lap_num]
        built_lt  = lap_times_strategy[:lap_num]
        built_gap = gap_series[:lap_num]
        built_spd = speed_series[:lap_num]

        # compound-coloured lap-time line
        # build a colour array — one colour per point based on compound
        lt_colors = [
            TIRE_COLORS.get(lap_compounds[i].upper(), ACC)
            for i in range(lap_num)
        ]

        # frame data — must match trace order exactly
        frame_data = [
            # trace 3 — ghost car
            go.Scatter(x=[gx], y=[gy]),
            # trace 4 — strategy car
            go.Scatter(
                x=[sx], y=[sy],
                marker=dict(size=17, color=car_color, symbol="circle",
                            line=dict(color="#ffffff", width=2)),
            ),
            # trace 6 — lap-time built line
            go.Scatter(
                x=built_x,
                y=built_lt,
                marker=dict(size=4, color=lt_colors),
                line=dict(color=ACC, width=2),
            ),
            # trace 8 — gap built line
            go.Scatter(
                x=built_x,
                y=built_gap,
                line=dict(color=gap_color, width=2),
                fill="tozeroy",
                fillcolor=("rgba(68,255,136,0.08)"
                           if gap_val < 0 else "rgba(255,68,68,0.08)"),
            ),
            # trace 10 — speed built line
            go.Scatter(x=built_x, y=built_spd),
            # trace 11 — throttle gauge
            go.Bar(x=[thr_val], y=["THROTTLE"]),
            # trace 12 — brake gauge
            go.Bar(x=[brk_val], y=["BRAKE"]),
        ]

        # Per-frame layout patch:
        #   shapes  — rebuild pit shapes (static) + updated cursor shapes
        #   annotations — live tile values only (5 paper-coord tiles)
        gap_sign = f"{gap_val:+.2f}s"

        frame_cursor_shapes = [
            _cursor_shape(lap_num, "x2", "y2 domain"),
            _cursor_shape(lap_num, "x3", "y3 domain"),
            _cursor_shape(lap_num, "x4", "y4 domain"),
        ]
        frame_shapes = shapes + frame_cursor_shapes

        frame_annotations = [
            _tile_ann(0.20, 0.97, f"LAP {lap_num}/{total_laps}"),
            _tile_ann(0.20, 0.91,
                      f'<b style="color:{gap_color}">{gap_sign}</b>'),
            _tile_ann(0.20, 0.85, f"{spd_val:.0f} km/h"),
            _tile_ann(0.20, 0.79, f"THROTTLE {thr_val:.0f}%"),
            _tile_ann(0.20, 0.73,
                      f'<b style="color:{car_color}">{compound}</b>'),
        ]

        frames.append(go.Frame(
            data=frame_data,
            traces=[3, 4, 6, 8, 10, 11, 12],
            layout=go.Layout(
                shapes=frame_shapes,
                annotations=frame_annotations,
            ),
            name=str(step),
        ))

    fig.frames = frames

    # ── 8. slider definition ──────────────────────────────────────────────
    slider_steps = []
    # one slider step per lap (not per animation frame — more usable)
    for lap_i in range(total_laps):
        # find the frame closest to this lap completing
        target_time = cum_s[lap_i + 1]
        target_step = min(
            round((target_time / total_race_time) * (time_steps - 1)),
            time_steps - 1,
        )
        slider_steps.append(dict(
            method="animate",
            label=str(lap_i + 1),
            args=[
                [str(target_step)],
                {"frame": {"duration": 0, "redraw": False},
                 "mode": "immediate",
                 "transition": {"duration": 0}},
            ],
        ))

    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix="LAP ",
            visible=True,
            font=dict(family="Orbitron", size=11, color=TXT),
        ),
        pad=dict(t=10, b=5),
        bgcolor="#1a1a1a",
        bordercolor=GRID,
        tickcolor=DIM,
        font=dict(family="Rajdhani", size=9, color=TXT),
        steps=slider_steps,
        len=0.50,
        x=0.50,
        y=0.0,
    )]

    # ── 9. layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Rajdhani", color=TXT),
        height=800,
        margin=dict(l=20, r=20, t=55, b=80),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.02,
            # orientation="h",
            bgcolor="rgba(0,0,0,0)",
            bordercolor=ACC, borderwidth=2,
            font=dict(family="Rajdhani", size=10, color=TXT),
        ),
        annotations=init_annotations + panel_titles,
        shapes=all_static_shapes,

        # play / pause buttons
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.1, y=-0.05,
            xanchor="center",
            font=dict(family="Orbitron", color="#669bbc"),
            bgcolor="#1a1a1a",
            bordercolor=GRID,
            buttons=[
                dict(
                    label="▶  PLAY",
                    method="animate",
                    args=[
                        None,
                        {"frame":      {"duration": 75, "redraw": True},
                         "fromcurrent": True,
                         "transition":  {"duration": 75, "easing": "linear"}},
                    ],
                ),
                dict(
                    label="⏸  PAUSE",
                    method="animate",
                    args=[
                        [None],
                        {"frame":      {"duration": 0, "redraw": False},
                         "mode":       "immediate",
                         "transition": {"duration": 0}},
                    ],
                ),
            ],
        )],
        sliders=sliders,
    )

    # ── 10. axis styling per subplot ──────────────────────────────────────

    # track map — no axes
    fig.update_xaxes(visible=False, fixedrange=True, row=1, col=1)
    fig.update_yaxes(visible=False, fixedrange=True,
                     scaleanchor="x", scaleratio=1, row=1, col=1)

    # panel 1 — lap time
    fig.update_xaxes(title_text="Lap", **_axis_cfg(), row=1, col=2,
                     range=[0.5, total_laps + 0.5])
    fig.update_yaxes(title_text="Time (s)", **_axis_cfg(), row=1, col=2,
                     range=[lt_min, lt_max])
    # panel 2 — gap
    fig.update_xaxes(title_text="Lap", **_axis_cfg(), row=1, col=3,
                     range=[0.5, total_laps + 0.5])
    fig.update_yaxes(title_text="Gap (s)", **_axis_cfg(), row=1, col=3,
                     range=[gap_min - 1, gap_max + 1])

    # panel 3 — speed
    fig.update_xaxes(title_text="Lap", **_axis_cfg(), row=2, col=2,
                     range=[0.5, total_laps + 0.5])
    fig.update_yaxes(title_text="km/h", **_axis_cfg(), row=2, col=2,
                     range=[spd_min, spd_max])

    # panel 4 — gauge
    fig.update_xaxes(title_text="%", **_axis_cfg(), row=2, col=3,
                     range=[0, 105])
    gauge_yaxis_cfg = {**_axis_cfg(),
                        "tickfont": dict(family="Orbitron", size=9, color=TXT)}
    fig.update_yaxes(**gauge_yaxis_cfg, row=2, col=3)
    fig.update_layout(barmode="group")



    return fig