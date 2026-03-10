"""
build_pit_wall_figure()
───────────────────────
WHY NO make_subplots?
  make_subplots forces redraw=True on every animation frame, which redraws
  the entire SVG from scratch each tick — causing visible shaking and
  choppiness.  A plain go.Figure() with redraw=False only repaints the
  trace xy-data that actually changed, giving the same smooth result as
  the original plot_circuit_animated().

LAYOUT
  Left half  (xaxis/yaxis)                   ← track map, same as original
  Right half  — four stacked mini-panels using secondary axes:
    xaxis2/yaxis2  (domain y 0.76–1.00)      ← Lap time
    xaxis3/yaxis3  (domain y 0.51–0.75)      ← Gap vs baseline
    xaxis4/yaxis4  (domain y 0.26–0.50)      ← Avg speed
    xaxis5/yaxis5  (domain y 0.02–0.25)      ← Throttle / Brake gauge bars

TRACE INDEX MAP  (order is fixed — frames reference by index)
  0  track outline        static   xaxis/yaxis
  1  start/finish star    static   xaxis/yaxis
  2  corner markers       static   xaxis/yaxis
  3  ghost car            animated xaxis/yaxis
  4  strategy car         animated xaxis/yaxis
  5  scoreboard text      animated xaxis/yaxis  (top-right of track canvas)
  --- telemetry panels ---
  6  lap-time baseline    static   xaxis2/yaxis2
  7  lap-time built line  animated xaxis2/yaxis2
  8  gap zero ref         static   xaxis3/yaxis3
  9  gap built line       animated xaxis3/yaxis3
  10 speed ref            static   xaxis4/yaxis4
  11 speed built line     animated xaxis4/yaxis4
  12 throttle bar         animated xaxis5/yaxis5
  13 brake bar            animated xaxis5/yaxis5

ANIMATION STRATEGY
  Every frame  → traces 3, 4, 5  (cars + scoreboard — tiny payload)
  Lap change   → traces 3–13 + layout.shapes  (telemetry panels update)

  This gives 1250 frames for smooth car motion while telemetry panels
  only update ~total_laps times regardless of time_steps.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
def _data(f): return os.path.join(_DATA, f)

try:
    from constants import TIRE_COLORS
except ImportError:
    TIRE_COLORS = {
        "SOFT":         "#e8002d",
        "MEDIUM":       "#ffd700",
        "HARD":         "#eeeeee",
        "INTERMEDIATE": "#39b54a",
        "WET":          "#0067ff",
    }

# ── palette (matches original helpers.py aesthetic) ────────────────────────
BG      = "#0a0a0a"
GRID    = "#1c1c1c"
TXT     = "#cccccc"
DIM     = "#444444"
ACC     = "#e10600"
SPD_C   = "#00bfff"
THR_C   = "#44ff88"
BRK_C   = "#ff6b35"
GAP_AHD = "#44ff88"   # strategy is ahead of baseline (gap < 0)
GAP_BHD = "#ff4444"   # strategy is behind baseline  (gap > 0)

# ── panel column: right half of the canvas ─────────────────────────────────
_PX0, _PX1 = 0.50, 1.00   # x domain for all four telemetry panels
_PANELS = {                 # name → (y_domain_start, y_domain_end)
    "laptime": (0.76, 1.00),
    "gap":     (0.51, 0.75),
    "speed":   (0.26, 0.50),
    "gauge":   (0.02, 0.25),
}

# axis numbers for each panel (xaxis2=laptime … xaxis5=gauge)
_AXIS_NUM = {"laptime": 2, "gap": 3, "speed": 4, "gauge": 5}


def _cumulative_times(lap_times):
    cum = [0.0]
    for t in lap_times:
        cum.append(cum[-1] + t)
    return cum


def _car_pos(ct, cum, total_laps, track_x, track_y):
    """Smooth interpolated car position — all three index bounds clamped."""
    n   = len(track_x)
    lap = 0
    for i, t in enumerate(cum):
        if ct >= t:
            lap = i
        else:
            break
    lap   = min(lap, total_laps - 1)
    denom = cum[lap + 1] - cum[lap]
    frac  = (ct - cum[lap]) / denom if denom else 0.0
    frac  = max(0.0, min(1.0, frac))
    fi    = frac * (n - 1)
    bi    = min(int(fi), n - 2)
    rem   = fi - bi
    ni    = bi + 1
    x = track_x[bi] + rem * (track_x[ni] - track_x[bi])
    y = track_y[bi] + rem * (track_y[ni] - track_y[bi])
    return float(x), float(y), lap


def _axis_style():
    return dict(
        showgrid=True, gridcolor=GRID, gridwidth=0.5,
        zeroline=False, showline=True, linecolor=GRID,
        tickfont=dict(family="Rajdhani", color=TXT, size=8),
        title_font=dict(family="Rajdhani", color=TXT, size=8),
    )


def _pit_shapes(laps_pit, total_laps):
    """Static vertical pit-stop lines across all four panel y-domains."""
    shapes = []
    for p in laps_pit:
        for panel, ax_n in _AXIS_NUM.items():
            y0, y1 = _PANELS[panel]
            shapes.append(dict(
                type="line",
                xref=f"x{ax_n}", yref="paper",
                x0=p, x1=p, y0=y0, y1=y1,
                line=dict(color="#ffaa00", width=1, dash="dot"),
                layer="below",
            ))
    return shapes


def _cursor_shapes(lap_num):
    """Vertical cursor lines at the current lap across all panels."""
    shapes = []
    for panel, ax_n in _AXIS_NUM.items():
        if panel == "gauge":
            continue   # bar chart — no cursor
        y0, y1 = _PANELS[panel]
        shapes.append(dict(
            type="line",
            xref=f"x{ax_n}", yref="paper",
            x0=lap_num, x1=lap_num, y0=y0, y1=y1,
            line=dict(color="#ffffff", width=1.2, dash="solid"),
            opacity=0.4,
        ))
    return shapes


# ═══════════════════════════════════════════════════════════════════════════
def build_pit_wall_figure(
    track_choice:        str,
    lap_times_strategy:  list,
    lap_times_baseline:  list,
    total_laps:          int,
    laps_pit:            list,
    lap_compounds:       list,
    telem_series:        dict,
    strategy:            list,
    time_steps:          int = 1250,   # match original helpers.py
) -> go.Figure:
    """
    Build the smooth unified pit-wall figure.

    time_steps=1250 matches the original plot_circuit_animated — ~20 frames
    per lap for a 60-lap race, which is enough for smooth car motion.
    redraw=False in the Play button means only changed trace data repaints.
    """

    # ── track geometry ────────────────────────────────────────────────────
    df       = pd.read_csv(_data("circuit_info.csv"))
    track_df = (df[(df["track"] == track_choice) & (df["type"] == "track")]
                .sort_values("seq"))
    corner_df = df[(df["track"] == track_choice) & (df["type"] == "corner")]

    track_x = [float(v) for v in track_df["X"]]
    track_y = [float(v) for v in track_df["Y"]]

    x_min, x_max = min(track_x), max(track_x)
    y_min, y_max = min(track_y), max(track_y)
    x_pad = (x_max - x_min) * 0.12
    y_pad = (y_max - y_min) * 0.12

    # sidebar_x: scoreboard text anchor (right of track, left of panels)
    # matches the original helpers.py pattern exactly
    sidebar_x = x_max + (x_max - x_min) * 0.15

    # ── telemetry series ──────────────────────────────────────────────────
    laps_x  = list(range(1, total_laps + 1))
    gap_s   = list(np.cumsum(
        np.array(lap_times_strategy) - np.array(lap_times_baseline)
    ))

    # accept both key name variants
    spd_s = (telem_series.get("avg_speed_kmh") or
             telem_series.get("avg_speed") or [150.0] * total_laps)
    thr_s = (telem_series.get("throttle_percent") or
             telem_series.get("throttle_pct") or [0.5] * total_laps)
    brk_s = (telem_series.get("braking_percent") or
             telem_series.get("braking_pct") or [0.2] * total_laps)

    # normalise fractions → percent if needed
    def _to_pct(v): return v * 100 if (v is not None and v <= 1.0) else (v or 0.0)
    thr_pct = [_to_pct(v) for v in thr_s]
    brk_pct = [_to_pct(v) for v in brk_s]

    cum_s = _cumulative_times(lap_times_strategy)
    cum_b = _cumulative_times(lap_times_baseline)
    total_race_time = max(cum_s[-1], cum_b[-1])

    # axis ranges
    lt_min  = min(min(lap_times_strategy), min(lap_times_baseline)) * 0.997
    lt_max  = max(max(lap_times_strategy), max(lap_times_baseline)) * 1.003
    gap_abs = max(abs(min(gap_s)), abs(max(gap_s)))
    gap_min, gap_max = -(gap_abs * 1.1 + 1), (gap_abs * 1.1 + 1)
    spd_min = min(spd_s) * 0.97 if spd_s else 0
    spd_max = max(spd_s) * 1.03 if spd_s else 350

    static_pit_shapes = _pit_shapes(laps_pit, total_laps)

    # ── figure (plain go.Figure — NO make_subplots) ───────────────────────
    fig = go.Figure()

    # ── static traces ─────────────────────────────────────────────────────
    # 0: track outline
    fig.add_trace(go.Scatter(
        x=track_x, y=track_y,
        mode="lines", line=dict(color="#ffffff", width=6),
        hoverinfo="skip", name="Circuit", showlegend=True,
    ))
    # 1: start/finish
    fig.add_trace(go.Scatter(
        x=[track_x[0]], y=[track_y[0]],
        mode="markers", marker=dict(symbol="star", size=15, color="#00ff00"),
        name="Start/Finish", showlegend=True,
    ))
    # 2: corners
    cx = list(corner_df["X"]) if not corner_df.empty else [None]
    cy = list(corner_df["Y"]) if not corner_df.empty else [None]
    ct_ = list(corner_df["corner"]) if not corner_df.empty else []
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers+text", text=ct_,
        textposition="top center",
        textfont=dict(color="#ffaa00", size=8, family="Orbitron"),
        marker=dict(color="#ffaa00", size=4),
        hoverinfo="skip", showlegend=False,
    ))

    # ── animated track traces ─────────────────────────────────────────────
    # 3: ghost car
    fig.add_trace(go.Scatter(
        x=[track_x[0]], y=[track_y[0]], mode="markers",
        marker=dict(size=16, color="rgba(100,100,255,0.5)", symbol="diamond",
                    line=dict(color="#6666ff", width=2)),
        name="Ghost Car (baseline)", showlegend=True,
    ))
    # 4: strategy car
    fig.add_trace(go.Scatter(
        x=[track_x[0]], y=[track_y[0]], mode="markers",
        marker=dict(size=18, color=ACC, symbol="circle",
                    line=dict(color="#ffffff", width=2)),
        name="Strategy Car", showlegend=True,
    ))
    # 5: scoreboard text (same pattern as original helpers.py)
    fig.add_trace(go.Scatter(
        x=[sidebar_x], y=[y_max],
        mode="text",
        text=["Initializing..."],
        textposition="bottom center",
        textfont=dict(family="Orbitron", size=13, color="#ffffff"),
        showlegend=False, name="Telemetry", cliponaxis=False,
    ))

    # ── telemetry panel traces (secondary axes) ───────────────────────────
    # Panel A — Lap time  (xaxis2 / yaxis2)
    # 6: baseline ref (static grey dotted)
    fig.add_trace(go.Scatter(
        x=laps_x, y=lap_times_baseline,
        mode="lines", line=dict(color=DIM, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
        xaxis="x2", yaxis="y2",
    ))
    # 7: strategy line (animated, builds lap-by-lap)
    fig.add_trace(go.Scatter(
        x=[laps_x[0]], y=[lap_times_strategy[0]],
        mode="lines+markers", line=dict(color=ACC, width=2),
        marker=dict(size=4, color=ACC),
        showlegend=False, xaxis="x2", yaxis="y2",
    ))

    # Panel B — Gap  (xaxis3 / yaxis3)
    # 8: zero ref
    fig.add_trace(go.Scatter(
        x=[laps_x[0], laps_x[-1]], y=[0, 0],
        mode="lines", line=dict(color=DIM, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
        xaxis="x3", yaxis="y3",
    ))
    # 9: gap line (animated)
    fig.add_trace(go.Scatter(
        x=[laps_x[0]], y=[gap_s[0]],
        mode="lines",
        line=dict(color=GAP_AHD if gap_s[0] < 0 else GAP_BHD, width=2),
        fill="tozeroy",
        fillcolor=("rgba(68,255,136,0.08)" if gap_s[0] < 0
                   else "rgba(255,68,68,0.08)"),
        showlegend=False, xaxis="x3", yaxis="y3",
    ))

    # Panel C — Speed  (xaxis4 / yaxis4)
    # 10: speed ref (static grey)
    fig.add_trace(go.Scatter(
        x=laps_x, y=spd_s,
        mode="lines", line=dict(color=DIM, width=1),
        hoverinfo="skip", showlegend=False,
        xaxis="x4", yaxis="y4",
    ))
    # 11: speed line (animated)
    fig.add_trace(go.Scatter(
        x=[laps_x[0]], y=[spd_s[0] if spd_s else 150],
        mode="lines+markers", line=dict(color=SPD_C, width=2),
        marker=dict(size=4, color=SPD_C),
        showlegend=False, xaxis="x4", yaxis="y4",
    ))

    # Panel D — Throttle / Brake  (xaxis5 / yaxis5)
    # 12: throttle bar (animated)
    fig.add_trace(go.Bar(
        x=[thr_pct[0]], y=["THR"],
        orientation="h", marker=dict(color=THR_C, opacity=0.85),
        showlegend=False, width=0.4,
        xaxis="x5", yaxis="y5",
    ))
    # 13: brake bar (animated)
    fig.add_trace(go.Bar(
        x=[brk_pct[0]], y=["BRK"],
        orientation="h", marker=dict(color=BRK_C, opacity=0.85),
        showlegend=False, width=0.4,
        xaxis="x5", yaxis="y5",
    ))

    # ── panel title annotations (static) ─────────────────────────────────
    def _panel_title(text, panel):
        y0, y1 = _PANELS[panel]
        return dict(
            x=(_PX0 + _PX1) / 2, y=y1 + 0.005,
            xref="paper", yref="paper",
            text=text, showarrow=False,
            font=dict(family="Orbitron", size=8, color=ACC),
            xanchor="center", yanchor="bottom",
        )

    static_annotations = [
        _panel_title("LAP TIME",         "laptime"),
        _panel_title("GAP vs BASELINE",  "gap"),
        _panel_title("AVG SPEED",        "speed"),
        _panel_title("THROTTLE / BRAKE", "gauge"),
    ]

    # ── build frames ──────────────────────────────────────────────────────
    frames   = []
    prev_li  = -1

    for step in range(time_steps):
        ct = (step / max(time_steps - 1, 1)) * total_race_time

        # car positions — every frame
        sx, sy, s_lap = _car_pos(ct, cum_s, total_laps, track_x, track_y)
        gx, gy, _     = _car_pos(ct, cum_b, total_laps, track_x, track_y)
        li        = min(s_lap, total_laps - 1)
        compound  = lap_compounds[li] if li < len(lap_compounds) else "MEDIUM"
        car_color = TIRE_COLORS.get(compound.upper(), ACC)
        lap_num   = li + 1
        lap_changed = (li != prev_li)
        prev_li = li

        # scoreboard text (matches original helpers.py format)
        gap_val  = gap_s[li]
        gap_col  = "#00ff00" if gap_val < 0 else "#ff0000"
        status   = "IN PIT" if (laps_pit and lap_num in laps_pit) else "ON TRACK"
        scoreboard = (
            f"<b>RACE TELEMETRY</b><br>"
            f"LAP: {lap_num}/{total_laps}<br>"
            f"GAP: <span style='color:{gap_col}'>{gap_val:+.2f}s</span><br>"
            f"TIRE: {compound}<br>"
            f"STATUS: {status}"
        )

        if lap_changed:
            # full update: cars + scoreboard + all telemetry panels
            spd_val  = spd_s[li]   if li < len(spd_s)   else 0.0
            thr_val  = thr_pct[li] if li < len(thr_pct) else 0.0
            brk_val  = brk_pct[li] if li < len(brk_pct) else 0.0
            gap_line = GAP_AHD if gap_val < 0 else GAP_BHD

            bx  = laps_x[:lap_num]
            blt = lap_times_strategy[:lap_num]
            bgp = gap_s[:lap_num]
            bsp = spd_s[:lap_num]
            ltc = [TIRE_COLORS.get(lap_compounds[j].upper(), ACC)
                   for j in range(lap_num)]

            frame_data = [
                go.Scatter(x=[gx], y=[gy]),                             # 3 ghost
                go.Scatter(x=[sx], y=[sy],                              # 4 strategy
                           marker=dict(size=18, color=car_color,
                                       symbol="circle",
                                       line=dict(color="#ffffff", width=2))),
                go.Scatter(x=[sidebar_x], y=[y_max],                   # 5 scoreboard
                           text=[scoreboard]),
                go.Scatter(x=bx, y=blt,                                # 7 lap-time
                           marker=dict(size=4, color=ltc),
                           line=dict(color=ACC, width=2)),
                go.Scatter(x=bx, y=bgp,                                # 9 gap
                           line=dict(color=gap_line, width=2),
                           fill="tozeroy",
                           fillcolor=("rgba(68,255,136,0.08)" if gap_val < 0
                                      else "rgba(255,68,68,0.08)")),
                go.Scatter(x=bx, y=bsp),                               # 11 speed
                go.Bar(x=[thr_val], y=["THR"]),                        # 12 throttle
                go.Bar(x=[brk_val], y=["BRK"]),                        # 13 brake
            ]
            frame_traces = [3, 4, 5, 7, 9, 11, 12, 13]
            frame_layout = go.Layout(
                shapes=static_pit_shapes + _cursor_shapes(lap_num),
                annotations=static_annotations,
            )

        else:
            # car + scoreboard only — no panel redraws
            frame_data = [
                go.Scatter(x=[gx], y=[gy]),
                go.Scatter(x=[sx], y=[sy],
                           marker=dict(size=18, color=car_color,
                                       symbol="circle",
                                       line=dict(color="#ffffff", width=2))),
                go.Scatter(x=[sidebar_x], y=[y_max], text=[scoreboard]),
            ]
            frame_traces = [3, 4, 5]
            frame_layout = go.Layout()

        frames.append(go.Frame(
            data=frame_data,
            traces=frame_traces,
            layout=frame_layout,
            name=str(step),
        ))

    fig.frames = frames

    # ── slider — one step per lap ─────────────────────────────────────────
    slider_steps = []
    for lap_i in range(total_laps):
        target_step = min(
            round((cum_s[lap_i + 1] / total_race_time) * (time_steps - 1)),
            time_steps - 1,
        )
        slider_steps.append(dict(
            method="animate",
            label=str(lap_i + 1),
            args=[[str(target_step)],
                  {"frame": {"duration": 0, "redraw": False},
                   "mode": "immediate",
                   "transition": {"duration": 0}}],
        ))

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="LAP ", visible=True,
                          font=dict(family="Orbitron", size=11, color=TXT)),
        pad=dict(t=10, b=5),
        bgcolor="#1a1a1a", bordercolor=GRID,
        font=dict(family="Rajdhani", size=9, color=TXT),
        steps=slider_steps,
        len=0.50, x=0.50, y=0.0,
    )]

    # ── layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Rajdhani", color=TXT),
        height=750,
        margin=dict(l=20, r=20, t=50, b=70),
        showlegend=True,
        legend=dict(x=0.02, y=0.02,
                    bgcolor="rgba(20,20,20,0.8)",
                    bordercolor=ACC, borderwidth=2,
                    font=dict(color="#ffffff", family="Rajdhani")),
        barmode="overlay",
        annotations=static_annotations,
        shapes=static_pit_shapes + _cursor_shapes(1),

        # ── track map — left half ─────────────────────────────────────────
        xaxis=dict(
            visible=False, fixedrange=True,
            domain=[0.0, 0.46],
            range=[x_min - x_pad, sidebar_x + x_pad],
        ),
        yaxis=dict(
            visible=False, fixedrange=True,
            domain=[0.05, 1.0],
            scaleanchor="x", scaleratio=1,
            range=[y_min - y_pad, y_max + y_pad],
        ),

        # ── Panel A: Lap time  ────────────────────────────────────────────
        xaxis2=dict(domain=[_PX0, _PX1], range=[0.5, total_laps + 0.5],
                    **_axis_style()),
        yaxis2=dict(domain=list(_PANELS["laptime"]),
                    range=[lt_min, lt_max], title_text="s", **_axis_style()),

        # ── Panel B: Gap ──────────────────────────────────────────────────
        xaxis3=dict(domain=[_PX0, _PX1], range=[0.5, total_laps + 0.5],
                    **_axis_style()),
        yaxis3=dict(domain=list(_PANELS["gap"]),
                    range=[gap_min, gap_max], title_text="gap (s)",
                    **_axis_style()),

        # ── Panel C: Speed ────────────────────────────────────────────────
        xaxis4=dict(domain=[_PX0, _PX1], range=[0.5, total_laps + 0.5],
                    **_axis_style()),
        yaxis4=dict(domain=list(_PANELS["speed"]),
                    range=[spd_min, spd_max], title_text="km/h",
                    **_axis_style()),

        # ── Panel D: Gauge ────────────────────────────────────────────────
        xaxis5=dict(domain=[_PX0, _PX1], range=[0, 105],
                    title_text="%", **_axis_style()),
        yaxis5=dict(domain=list(_PANELS["gauge"]),
                    tickfont=dict(family="Orbitron", size=8, color=TXT)),

        # ── Play / Pause ──────────────────────────────────────────────────
        # redraw=False is THE key difference from the make_subplots version.
        # It tells Plotly to only repaint changed trace data, not the full SVG.
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.5, y=-0.05, xanchor="center",
            font=dict(family="Orbitron", color="#669bbc"),
            buttons=[
                dict(label="Play", method="animate",
                     args=[None,
                           {"frame": {"duration": 100, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": 100, "easing": "linear"}}]),
                dict(label="Pause", method="animate",
                     args=[[None],
                           {"frame": {"duration": 0, "redraw": False},
                            "mode": "immediate"}]),
            ],
        )],
        sliders=sliders,
    )

    return fig