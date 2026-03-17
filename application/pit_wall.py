"""
build_pit_wall_figure()
───────────────────────
Single Plotly figure: track map + four telemetry panels, one shared
animation loop.  Supports two layouts selected by the `mobile` flag:

  mobile=False  (desktop ≥ 768px)
    make_subplots 2×3 grid — track left, 4 panels right
    height 800px

  mobile=True   (phone < 768px)
    make_subplots 5×1 stack — track top, 4 panels stacked below
    height 1300px  (tall but every panel is full device-width)
    larger fonts, bigger car markers, scoreboard on track panel

The `mobile` flag is set by a JS screen-width detector injected in
simulator.py that writes to st.session_state["screen_width"].

Trace index map — SAME for both layouts (order never changes)
─────────────────────────────────────────────────────────────
  0   track outline        static
  1   start/finish star    static
  2   corner markers       static
  3   ghost car            animated
  4   strategy car         animated
  5   lap-time baseline    static grey
  6   lap-time built line  animated
  7   gap zero ref         static grey
  8   gap built line       animated
  9   speed ref            static grey
 10   speed built line     animated
 11   throttle bar         animated
 12   brake bar            animated
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

BG      = "#0a0a0a"
GRID    = "#1c1c1c"
TXT     = "#cccccc"
DIM     = "#444444"
ACC     = "#e10600"
SPD_COL = "#00bfff"
THR_COL = "#44ff88"
BRK_COL = "#ff6b35"
GAP_POS = "#ff4444"
GAP_NEG = "#44ff88"


# ── shared helpers ──────────────────────────────────────────────────────────

def _to_float_list(s):
    if hasattr(s, "tolist"):
        return [float(v) for v in s.tolist()]
    return [float(v) for v in s]

def _cumulative_times(lap_times):
    cum = [0.0]
    for t in lap_times:
        cum.append(cum[-1] + t)
    return cum

def _car_pos(ct, cum, total_laps, tx, ty):
    n = len(tx)
    lap = 0
    for i, t in enumerate(cum):
        if ct >= t: lap = i
        else: break
    lap  = min(lap, total_laps - 1)
    denom = cum[lap + 1] - cum[lap]
    frac  = (ct - cum[lap]) / denom if denom else 0.0
    frac  = max(0.0, min(1.0, frac))
    fi    = frac * (n - 1)
    bi    = min(int(fi), n - 2)
    rem   = fi - bi
    ni    = bi + 1
    return float(tx[bi] + rem * (tx[ni] - tx[bi])), \
           float(ty[bi] + rem * (ty[ni] - ty[bi])), lap

def _axis_cfg(fs=9):
    return dict(
        showgrid=True, gridcolor=GRID, gridwidth=0.5,
        zeroline=False, showline=True, linecolor=GRID,
        tickfont=dict(family="Rajdhani", color=TXT, size=fs),
        title_font=dict(family="Rajdhani", color=TXT, size=fs),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PREP  (shared by both layouts)
# ══════════════════════════════════════════════════════════════════════════════

def _prep(track_choice, lap_times_strategy, lap_times_baseline,
          total_laps, telem_series):
    """Load geometry + derive all series. Returns a single dict."""
    df       = pd.read_csv(_data("circuit_info.csv"))
    track_df = (df[(df["track"] == track_choice) & (df["type"] == "track")]
                .sort_values("seq"))
    corner_df = df[(df["track"] == track_choice) & (df["type"] == "corner")]

    tx = _to_float_list(track_df["X"])
    ty = _to_float_list(track_df["Y"])

    laps_x    = list(range(1, total_laps + 1))
    gap_s     = list(np.cumsum(
        np.array(lap_times_strategy) - np.array(lap_times_baseline)))

    spd_s = telem_series.get("avg_speed_kmh",    [0.0] * total_laps)
    thr_s = telem_series.get("throttle_pct",
            telem_series.get("throttle_percent",  [0.5] * total_laps))
    brk_s = telem_series.get("braking_pct",
            telem_series.get("braking_percent",   [0.2] * total_laps))
    thr_pct = [v * 100 if v <= 1.0 else v for v in thr_s]
    brk_pct = [v * 100 if v <= 1.0 else v for v in brk_s]

    cum_s = _cumulative_times(lap_times_strategy)
    cum_b = _cumulative_times(lap_times_baseline)
    trt   = max(cum_s[-1], cum_b[-1])

    lt_min = min(min(lap_times_strategy), min(lap_times_baseline)) * 0.998
    lt_max = max(max(lap_times_strategy), max(lap_times_baseline)) * 1.002
    gap_mn = min(gap_s) * 1.05 if min(gap_s) < 0 else min(gap_s) * 0.95
    gap_mx = max(gap_s) * 1.05 if max(gap_s) > 0 else 1.0
    spd_mn = min(spd_s) * 0.98 if spd_s else 0
    spd_mx = max(spd_s) * 1.02 if spd_s else 350

    cx = _to_float_list(corner_df["X"]) if not corner_df.empty else [None]
    cy = _to_float_list(corner_df["Y"]) if not corner_df.empty else [None]
    ct = corner_df["corner"].tolist()   if not corner_df.empty else []

    return dict(
        tx=tx, ty=ty, cx=cx, cy=cy, ct=ct,
        laps_x=laps_x, gap_s=gap_s,
        spd_s=spd_s, thr_pct=thr_pct, brk_pct=brk_pct,
        cum_s=cum_s, cum_b=cum_b, trt=trt,
        lt_min=lt_min, lt_max=lt_max,
        gap_mn=gap_mn, gap_mx=gap_mx,
        spd_mn=spd_mn, spd_mx=spd_mx,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DESKTOP LAYOUT  (2×3 grid, current design)
# ══════════════════════════════════════════════════════════════════════════════

def _build_desktop(d, lap_times_strategy, lap_times_baseline,
                   total_laps, laps_pit, lap_compounds, strategy, time_steps):
    tx, ty = d["tx"], d["ty"]
    laps_x = d["laps_x"]
    gap_s, spd_s = d["gap_s"], d["spd_s"]
    thr_pct, brk_pct = d["thr_pct"], d["brk_pct"]
    cum_s, cum_b, trt = d["cum_s"], d["cum_b"], d["trt"]

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

    # ── static traces ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines",
        line=dict(color="#ffffff", width=5),
        hoverinfo="skip", showlegend=False, name="Circuit"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[tx[0]], y=[ty[0]], mode="markers",
        marker=dict(symbol="star", size=14, color="#00ff00"),
        hoverinfo="skip", showlegend=False, name="S/F"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["cx"], y=d["cy"],
        mode="markers+text", text=d["ct"], textposition="top center",
        textfont=dict(color="#ffaa00", size=7, family="Orbitron"),
        marker=dict(color="#ffaa00", size=3),
        hoverinfo="skip", showlegend=False, name="Corners"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[tx[0]], y=[ty[0]], mode="markers",
        marker=dict(size=14, color="rgba(100,100,255,0.5)", symbol="diamond",
                    line=dict(color="#6666ff", width=2)),
        name="Baseline", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[tx[0]], y=[ty[0]], mode="markers",
        marker=dict(size=17, color=ACC, symbol="circle",
                    line=dict(color="#ffffff", width=2)),
        name="Your Strategy", showlegend=True), row=1, col=1)

    # panel 1 — lap time
    fig.add_trace(go.Scatter(x=laps_x, y=lap_times_baseline, mode="lines",
        line=dict(color=DIM, width=1, dash="dot"),
        name="Baseline pace", showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=[laps_x[0]], y=[lap_times_strategy[0]],
        mode="lines+markers", line=dict(color=ACC, width=2),
        marker=dict(size=4, color=ACC),
        name="Strategy pace", showlegend=True), row=1, col=2)

    # panel 2 — gap
    fig.add_trace(go.Scatter(x=[laps_x[0], laps_x[-1]], y=[0, 0],
        mode="lines", line=dict(color=DIM, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[laps_x[0]], y=[gap_s[0]], mode="lines",
        line=dict(color=GAP_NEG if gap_s[0] < 0 else GAP_POS, width=2),
        fill="tozeroy",
        fillcolor="rgba(68,255,136,0.08)" if gap_s[0] < 0 else "rgba(255,68,68,0.08)",
        name="Gap to baseline", showlegend=True), row=1, col=3)

    # panel 3 — speed
    fig.add_trace(go.Scatter(x=laps_x, y=spd_s, mode="lines",
        line=dict(color=DIM, width=1),
        hoverinfo="skip", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[laps_x[0]], y=[spd_s[0]],
        mode="lines+markers", line=dict(color=SPD_COL, width=2),
        marker=dict(size=4, color=SPD_COL),
        name="Avg speed", showlegend=True), row=2, col=2)

    # panel 4 — gauge
    fig.add_trace(go.Bar(x=[thr_pct[0]], y=["THROTTLE"], orientation="h",
        marker=dict(color=THR_COL, opacity=0.85),
        showlegend=False, width=0.35), row=2, col=3)
    fig.add_trace(go.Bar(x=[brk_pct[0]], y=["BRAKE"], orientation="h",
        marker=dict(color=BRK_COL, opacity=0.85),
        showlegend=False, width=0.35), row=2, col=3)

    # ── shapes ────────────────────────────────────────────────────────────
    def _cursor(x, xref, yref_d):
        return dict(type="line", xref=xref, yref=yref_d,
                    x0=x, x1=x, y0=0, y1=1,
                    line=dict(color="#ffffff", width=1.5), opacity=0.55, layer="above")

    shapes = []
    for p in laps_pit:
        for xr, yr in [("x2","y2 domain"),("x3","y3 domain"),("x4","y4 domain")]:
            shapes.append(dict(type="line", xref=xr, yref=yr,
                x0=p, x1=p, y0=0, y1=1,
                line=dict(color="#ffaa00", width=1.2, dash="dot"), layer="below"))
    init_shapes = shapes + [_cursor(1,"x2","y2 domain"),
                            _cursor(1,"x3","y3 domain"),
                            _cursor(1,"x4","y4 domain")]

    # ── annotations ───────────────────────────────────────────────────────
    def _tile(x, y, text, size=18):
        return dict(x=x, y=y, xref="paper", yref="paper", text=text,
                    showarrow=False, font=dict(family="Orbitron", size=size, color=TXT),
                    align="center", bgcolor="rgba(20,20,20,0.7)",
                    bordercolor=GRID, borderwidth=1, borderpad=4)

    init_ann = [_tile(0.20,0.97,"LAP 1"), _tile(0.20,0.91,"GAP —"),
                _tile(0.20,0.85,"— km/h"), _tile(0.20,0.79,"THROTTLE —"),
                _tile(0.20,0.73,"—")]
    panel_titles = [
        dict(x=0.445,y=1.02,text="LAP TIME",showarrow=False,
             xref="paper",yref="paper",font=dict(family="Orbitron",size=9,color=ACC)),
        dict(x=0.755,y=1.02,text="GAP vs BASELINE",showarrow=False,
             xref="paper",yref="paper",font=dict(family="Orbitron",size=9,color=ACC)),
        dict(x=0.445,y=0.47,text="AVG SPEED",showarrow=False,
             xref="paper",yref="paper",font=dict(family="Orbitron",size=9,color=ACC)),
        dict(x=0.755,y=0.47,text="THROTTLE / BRAKE",showarrow=False,
             xref="paper",yref="paper",font=dict(family="Orbitron",size=9,color=ACC)),
    ]

    # ── frames ────────────────────────────────────────────────────────────
    frames = []
    for step in range(time_steps):
        ct  = (step / (time_steps - 1)) * trt
        sx, sy, s_lap = _car_pos(ct, cum_s, total_laps, tx, ty)
        gx, gy, _     = _car_pos(ct, cum_b, total_laps, tx, ty)
        li  = min(s_lap, total_laps - 1)
        cmp = lap_compounds[li] if li < len(lap_compounds) else "MEDIUM"
        col = TIRE_COLORS.get(cmp.upper(), ACC)
        gv  = gap_s[li]
        sv  = spd_s[li]  if li < len(spd_s)  else 0.0
        tv  = thr_pct[li] if li < len(thr_pct) else 0.0
        bv  = brk_pct[li] if li < len(brk_pct) else 0.0
        gc  = GAP_NEG if gv < 0 else GAP_POS
        ln  = li + 1
        bx  = laps_x[:ln]
        ltc = [TIRE_COLORS.get(lap_compounds[j].upper(), ACC) for j in range(ln)]

        fr_shapes = shapes + [_cursor(ln,"x2","y2 domain"),
                               _cursor(ln,"x3","y3 domain"),
                               _cursor(ln,"x4","y4 domain")]
        fr_ann = [
            _tile(0.20,0.97,f"LAP {ln}/{total_laps}"),
            _tile(0.20,0.91,f'<b style="color:{gc}">{gv:+.2f}s</b>'),
            _tile(0.20,0.85,f"{sv:.0f} km/h"),
            _tile(0.20,0.79,f"THROTTLE {tv:.0f}%"),
            _tile(0.20,0.73,f'<b style="color:{col}">{cmp}</b>'),
        ]
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[gx], y=[gy]),
                go.Scatter(x=[sx], y=[sy],
                    marker=dict(size=17,color=col,symbol="circle",
                                line=dict(color="#ffffff",width=2))),
                go.Scatter(x=bx, y=lap_times_strategy[:ln],
                    marker=dict(size=4,color=ltc), line=dict(color=ACC,width=2)),
                go.Scatter(x=bx, y=gap_s[:ln],
                    line=dict(color=gc,width=2), fill="tozeroy",
                    fillcolor="rgba(68,255,136,0.08)" if gv<0 else "rgba(255,68,68,0.08)"),
                go.Scatter(x=bx, y=spd_s[:ln]),
                go.Bar(x=[tv], y=["THROTTLE"]),
                go.Bar(x=[bv], y=["BRAKE"]),
            ],
            traces=[3,4,6,8,10,11,12],
            layout=go.Layout(shapes=fr_shapes, annotations=fr_ann),
            name=str(step),
        ))
    fig.frames = frames

    # ── slider ────────────────────────────────────────────────────────────
    slider_steps = []
    for i in range(total_laps):
        ts = min(round((cum_s[i+1]/trt)*(time_steps-1)), time_steps-1)
        slider_steps.append(dict(method="animate", label=str(i+1),
            args=[[str(ts)], {"frame":{"duration":0,"redraw":False},
                              "mode":"immediate","transition":{"duration":0}}]))

    # ── layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Rajdhani", color=TXT),
        height=800, margin=dict(l=20,r=20,t=55,b=80),
        showlegend=True,
        legend=dict(x=0.02,y=0.02,bgcolor="rgba(0,0,0,0)",
                    bordercolor=ACC,borderwidth=2,
                    font=dict(family="Rajdhani",size=10,color=TXT)),
        annotations=init_ann + panel_titles,
        shapes=init_shapes,
        updatemenus=[dict(type="buttons",direction="right",
            x=0.1,y=-0.05,xanchor="center",
            font=dict(family="Orbitron",color="#669bbc"),
            bgcolor="#1a1a1a",bordercolor=GRID,
            buttons=[
                dict(label="▶  PLAY",method="animate",
                     args=[None,{"frame":{"duration":75,"redraw":True},
                                 "fromcurrent":True,
                                 "transition":{"duration":75,"easing":"linear"}}]),
                dict(label="⏸  PAUSE",method="animate",
                     args=[[None],{"frame":{"duration":0,"redraw":False},
                                   "mode":"immediate","transition":{"duration":0}}]),
            ])],
        sliders=[dict(active=0,
            currentvalue=dict(prefix="LAP ",visible=True,
                              font=dict(family="Orbitron",size=11,color=TXT)),
            pad=dict(t=10,b=5),bgcolor="#1a1a1a",bordercolor=GRID,
            font=dict(family="Rajdhani",size=9,color=TXT),
            steps=slider_steps, len=0.50, x=0.50, y=0.0)],
    )
    fig.update_xaxes(visible=False, fixedrange=True, row=1, col=1)
    fig.update_yaxes(visible=False, fixedrange=True,
                     scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="Lap", **_axis_cfg(), row=1, col=2,
                     range=[0.5, total_laps+0.5])
    fig.update_yaxes(title_text="Time (s)", **_axis_cfg(), row=1, col=2,
                     range=[d["lt_min"], d["lt_max"]])
    fig.update_xaxes(title_text="Lap", **_axis_cfg(), row=1, col=3,
                     range=[0.5, total_laps+0.5])
    fig.update_yaxes(title_text="Gap (s)", **_axis_cfg(), row=1, col=3,
                     range=[d["gap_mn"]-1, d["gap_mx"]+1])
    fig.update_xaxes(title_text="Lap", **_axis_cfg(), row=2, col=2,
                     range=[0.5, total_laps+0.5])
    fig.update_yaxes(title_text="km/h", **_axis_cfg(), row=2, col=2,
                     range=[d["spd_mn"], d["spd_mx"]])
    fig.update_xaxes(title_text="%", **_axis_cfg(), row=2, col=3, range=[0,105])
    fig.update_yaxes(**{**_axis_cfg(),
        "tickfont":dict(family="Orbitron",size=9,color=TXT)}, row=2, col=3)
    fig.update_layout(barmode="group")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MOBILE LAYOUT  (5-row portrait stack — every panel full device width)
# ══════════════════════════════════════════════════════════════════════════════

def _build_mobile(d, lap_times_strategy, lap_times_baseline,
                  total_laps, laps_pit, lap_compounds, strategy, time_steps):
    """
    5-row single-column stack.  Each panel fills the full device width so
    nothing is squished.  Fonts are larger, car markers bigger, scoreboard
    annotations repositioned to sit inside the track map row.

    Row heights: track 28% | lap-time 18% | gap 18% | speed 18% | gauge 18%
    """
    tx, ty = d["tx"], d["ty"]
    laps_x = d["laps_x"]
    gap_s, spd_s = d["gap_s"], d["spd_s"]
    thr_pct, brk_pct = d["thr_pct"], d["brk_pct"]
    cum_s, cum_b, trt = d["cum_s"], d["cum_b"], d["trt"]

    fig = make_subplots(
        rows=5, cols=1,
        specs=[[{"type":"xy"}]]*5,
        row_heights=[0.28, 0.18, 0.18, 0.18, 0.18],
        vertical_spacing=0.04,
    )
    FS = 11   # font size for mobile axis labels (up from 9)

    # ── static traces ──────────────────────────────────────────────────────
    # 0: track
    fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines",
        line=dict(color="#ffffff", width=4),
        hoverinfo="skip", showlegend=False, name="Circuit"), row=1, col=1)
    # 1: S/F
    fig.add_trace(go.Scatter(x=[tx[0]], y=[ty[0]], mode="markers",
        marker=dict(symbol="star", size=16, color="#00ff00"),
        hoverinfo="skip", showlegend=False, name="S/F"), row=1, col=1)
    # 2: corners — omit text on mobile (too small), markers only
    fig.add_trace(go.Scatter(x=d["cx"], y=d["cy"], mode="markers",
        marker=dict(color="#ffaa00", size=5),
        hoverinfo="skip", showlegend=False, name="Corners"), row=1, col=1)
    # 3: ghost car
    fig.add_trace(go.Scatter(x=[tx[0]], y=[ty[0]], mode="markers",
        marker=dict(size=18, color="rgba(100,100,255,0.5)", symbol="diamond",
                    line=dict(color="#6666ff", width=2)),
        name="Baseline", showlegend=True), row=1, col=1)
    # 4: strategy car
    fig.add_trace(go.Scatter(x=[tx[0]], y=[ty[0]], mode="markers",
        marker=dict(size=22, color=ACC, symbol="circle",
                    line=dict(color="#ffffff", width=2)),
        name="Your Strategy", showlegend=True), row=1, col=1)

    # 5: lap-time baseline ref
    fig.add_trace(go.Scatter(x=laps_x, y=lap_times_baseline, mode="lines",
        line=dict(color=DIM, width=1, dash="dot"),
        name="Baseline pace", showlegend=True), row=2, col=1)
    # 6: lap-time built
    fig.add_trace(go.Scatter(x=[laps_x[0]], y=[lap_times_strategy[0]],
        mode="lines+markers", line=dict(color=ACC, width=2),
        marker=dict(size=5, color=ACC),
        name="Strategy pace", showlegend=True), row=2, col=1)

    # 7: gap zero ref
    fig.add_trace(go.Scatter(x=[laps_x[0], laps_x[-1]], y=[0, 0],
        mode="lines", line=dict(color=DIM, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False), row=3, col=1)
    # 8: gap built
    fig.add_trace(go.Scatter(x=[laps_x[0]], y=[gap_s[0]], mode="lines",
        line=dict(color=GAP_NEG if gap_s[0]<0 else GAP_POS, width=2),
        fill="tozeroy",
        fillcolor="rgba(68,255,136,0.08)" if gap_s[0]<0 else "rgba(255,68,68,0.08)",
        name="Gap to baseline", showlegend=True), row=3, col=1)

    # 9: speed ref
    fig.add_trace(go.Scatter(x=laps_x, y=spd_s, mode="lines",
        line=dict(color=DIM, width=1),
        hoverinfo="skip", showlegend=False), row=4, col=1)
    # 10: speed built
    fig.add_trace(go.Scatter(x=[laps_x[0]], y=[spd_s[0]],
        mode="lines+markers", line=dict(color=SPD_COL, width=2),
        marker=dict(size=5, color=SPD_COL),
        name="Avg speed", showlegend=True), row=4, col=1)

    # 11: throttle bar
    fig.add_trace(go.Bar(x=[thr_pct[0]], y=["THR"], orientation="h",
        marker=dict(color=THR_COL, opacity=0.85),
        showlegend=False, width=0.4), row=5, col=1)
    # 12: brake bar
    fig.add_trace(go.Bar(x=[brk_pct[0]], y=["BRK"], orientation="h",
        marker=dict(color=BRK_COL, opacity=0.85),
        showlegend=False, width=0.4), row=5, col=1)

    # ── shapes ────────────────────────────────────────────────────────────
    # On mobile all panels share xaxis2 (row2)…xaxis5 (row5)
    # yref domain strings: "y2 domain" … "y5 domain"
    def _cursor(x, xref, yref_d):
        return dict(type="line", xref=xref, yref=yref_d,
                    x0=x, x1=x, y0=0, y1=1,
                    line=dict(color="#ffffff", width=1.2), opacity=0.5, layer="above")

    shapes = []
    for p in laps_pit:
        for xr, yr in [("x2","y2 domain"),("x3","y3 domain"),
                       ("x4","y4 domain")]:
            shapes.append(dict(type="line", xref=xr, yref=yr,
                x0=p, x1=p, y0=0, y1=1,
                line=dict(color="#ffaa00", width=1.2, dash="dot"), layer="below"))
    init_shapes = shapes + [_cursor(1,"x2","y2 domain"),
                            _cursor(1,"x3","y3 domain"),
                            _cursor(1,"x4","y4 domain")]

    # ── annotations ───────────────────────────────────────────────────────
    # On mobile the scoreboard tiles sit at the RIGHT of the track map row
    # (paper x ~0.72–0.98) so they don't obscure the track.
    def _tile(x, y, text, size=13):
        return dict(x=x, y=y, xref="paper", yref="paper", text=text,
                    showarrow=False, font=dict(family="Orbitron",size=size,color=TXT),
                    align="left", bgcolor="rgba(10,10,10,0.85)",
                    bordercolor=GRID, borderwidth=1, borderpad=3)

    # y positions within the top 28% of paper (track row)
    # paper y 1.0 = top, 0.72 = bottom of track row (approx)
    TX = 0.68   # left edge of tile column (right side of track map)
    init_ann = [
        _tile(TX, 0.98, "LAP 1"),
        _tile(TX, 0.91, "GAP —"),
        _tile(TX, 0.84, "— km/h"),
        _tile(TX, 0.77, "THROTTLE —"),
        _tile(TX, 0.70, "—"),
    ]
    panel_titles = [
        dict(x=0.5,y=0.715,text="LAP TIME",showarrow=False,
             xref="paper",yref="paper",
             font=dict(family="Orbitron",size=FS,color=ACC),xanchor="center"),
        dict(x=0.5,y=0.530,text="GAP vs BASELINE",showarrow=False,
             xref="paper",yref="paper",
             font=dict(family="Orbitron",size=FS,color=ACC),xanchor="center"),
        dict(x=0.5,y=0.345,text="AVG SPEED",showarrow=False,
             xref="paper",yref="paper",
             font=dict(family="Orbitron",size=FS,color=ACC),xanchor="center"),
        dict(x=0.5,y=0.155,text="THROTTLE / BRAKE",showarrow=False,
             xref="paper",yref="paper",
             font=dict(family="Orbitron",size=FS,color=ACC),xanchor="center"),
    ]

    # ── frames ────────────────────────────────────────────────────────────
    frames = []
    for step in range(time_steps):
        ct  = (step / (time_steps - 1)) * trt
        sx, sy, s_lap = _car_pos(ct, cum_s, total_laps, tx, ty)
        gx, gy, _     = _car_pos(ct, cum_b, total_laps, tx, ty)
        li  = min(s_lap, total_laps - 1)
        cmp = lap_compounds[li] if li < len(lap_compounds) else "MEDIUM"
        col = TIRE_COLORS.get(cmp.upper(), ACC)
        gv  = gap_s[li]
        sv  = spd_s[li]  if li < len(spd_s)  else 0.0
        tv  = thr_pct[li] if li < len(thr_pct) else 0.0
        bv  = brk_pct[li] if li < len(brk_pct) else 0.0
        gc  = GAP_NEG if gv < 0 else GAP_POS
        ln  = li + 1
        bx  = laps_x[:ln]
        ltc = [TIRE_COLORS.get(lap_compounds[j].upper(), ACC) for j in range(ln)]

        fr_shapes = shapes + [_cursor(ln,"x2","y2 domain"),
                               _cursor(ln,"x3","y3 domain"),
                               _cursor(ln,"x4","y4 domain")]
        fr_ann = [
            _tile(TX, 0.98, f"LAP {ln}/{total_laps}"),
            _tile(TX, 0.91, f'<span style="color:{gc}">{gv:+.2f}s</span>'),
            _tile(TX, 0.84, f"{sv:.0f} km/h"),
            _tile(TX, 0.77, f"THR {tv:.0f}%"),
            _tile(TX, 0.70, f'<span style="color:{col}">{cmp}</span>'),
        ]
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[gx], y=[gy]),
                go.Scatter(x=[sx], y=[sy],
                    marker=dict(size=22,color=col,symbol="circle",
                                line=dict(color="#ffffff",width=2))),
                go.Scatter(x=bx, y=lap_times_strategy[:ln],
                    marker=dict(size=5,color=ltc), line=dict(color=ACC,width=2)),
                go.Scatter(x=bx, y=gap_s[:ln],
                    line=dict(color=gc,width=2), fill="tozeroy",
                    fillcolor="rgba(68,255,136,0.08)" if gv<0 else "rgba(255,68,68,0.08)"),
                go.Scatter(x=bx, y=spd_s[:ln]),
                go.Bar(x=[tv], y=["THR"]),
                go.Bar(x=[bv], y=["BRK"]),
            ],
            traces=[3,4,6,8,10,11,12],
            layout=go.Layout(shapes=fr_shapes, annotations=fr_ann),
            name=str(step),
        ))
    fig.frames = frames

    # ── slider ────────────────────────────────────────────────────────────
    slider_steps = []
    for i in range(total_laps):
        ts = min(round((cum_s[i+1]/trt)*(time_steps-1)), time_steps-1)
        slider_steps.append(dict(method="animate", label=str(i+1),
            args=[[str(ts)], {"frame":{"duration":0,"redraw":False},
                              "mode":"immediate","transition":{"duration":0}}]))

    # ── layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Rajdhani", color=TXT),
        height=1300, margin=dict(l=10,r=10,t=30,b=60),
        showlegend=True,
        legend=dict(x=0.0,y=-0.04,orientation="h",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Rajdhani",size=10,color=TXT)),
        annotations=init_ann + panel_titles,
        shapes=init_shapes,
        updatemenus=[dict(type="buttons",direction="right",
            x=0.5,y=-0.03,xanchor="center",
            font=dict(family="Orbitron",color="#669bbc"),
            bgcolor="#1a1a1a",bordercolor=GRID,
            buttons=[
                dict(label="▶ PLAY",method="animate",
                     args=[None,{"frame":{"duration":75,"redraw":True},
                                 "fromcurrent":True,
                                 "transition":{"duration":75,"easing":"linear"}}]),
                dict(label="⏸ PAUSE",method="animate",
                     args=[[None],{"frame":{"duration":0,"redraw":False},
                                   "mode":"immediate","transition":{"duration":0}}]),
            ])],
        sliders=[dict(active=0,
            currentvalue=dict(prefix="LAP ",visible=True,
                              font=dict(family="Orbitron",size=12,color=TXT)),
            pad=dict(t=8,b=4), bgcolor="#1a1a1a", bordercolor=GRID,
            font=dict(family="Rajdhani",size=10,color=TXT),
            steps=slider_steps, len=1.0, x=0.0, y=0.0)],
    )

    # track map — no axes, preserve aspect ratio
    fig.update_xaxes(visible=False, fixedrange=True, row=1, col=1)
    fig.update_yaxes(visible=False, fixedrange=True,
                     scaleanchor="x", scaleratio=1, row=1, col=1)
    # panel axes — all share col=1, rows 2-5
    fig.update_xaxes(title_text="Lap", **_axis_cfg(FS), row=2, col=1,
                     range=[0.5, total_laps+0.5])
    fig.update_yaxes(title_text="s", **_axis_cfg(FS), row=2, col=1,
                     range=[d["lt_min"], d["lt_max"]])
    fig.update_xaxes(title_text="Lap", **_axis_cfg(FS), row=3, col=1,
                     range=[0.5, total_laps+0.5])
    fig.update_yaxes(title_text="gap(s)", **_axis_cfg(FS), row=3, col=1,
                     range=[d["gap_mn"]-1, d["gap_mx"]+1])
    fig.update_xaxes(title_text="Lap", **_axis_cfg(FS), row=4, col=1,
                     range=[0.5, total_laps+0.5])
    fig.update_yaxes(title_text="km/h", **_axis_cfg(FS), row=4, col=1,
                     range=[d["spd_mn"], d["spd_mx"]])
    fig.update_xaxes(title_text="%", **_axis_cfg(FS), row=5, col=1,
                     range=[0, 105])
    fig.update_yaxes(**{**_axis_cfg(FS),
        "tickfont":dict(family="Orbitron",size=FS,color=TXT)}, row=5, col=1)
    fig.update_layout(barmode="group")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def build_pit_wall_figure(
    track_choice:       str,
    lap_times_strategy: list,
    lap_times_baseline: list,
    total_laps:         int,
    laps_pit:           list,
    lap_compounds:      list,
    telem_series:       dict,
    strategy:           list,
    mobile:             bool = False,
    time_steps:         int  = 1250,
) -> go.Figure:
    """
    Build the unified pit-wall figure.

    Pass mobile=True (from st.session_state["mobile"]) to get the
    portrait-stack layout optimised for phone screens.
    """
    d = _prep(track_choice, lap_times_strategy, lap_times_baseline,
              total_laps, telem_series)
    fn = _build_mobile if mobile else _build_desktop
    return fn(d, lap_times_strategy, lap_times_baseline,
              total_laps, laps_pit, lap_compounds, strategy, time_steps)