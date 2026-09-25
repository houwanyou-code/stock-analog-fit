"""Interactive Plotly charts for the web UI (English labels).

Each builder takes the structured output of a script's run() and returns a plotly Figure:
  kline_chart(res)            ← kline_fit.run()
  analog_overlay(plot)        ← analog.run()["plot"]   (prices, best match, hover shows reference's real date/price)
  norm_overlay(plot, n)       ← screen.run()["plot"]   (top-n matches normalized to target's last close)
  fan_chart(plot)             ← analog/screen ["plot"] (forward paths, median, 25–75% band)
  scenario_chart(plot)        ← scenario.run()["plot"]
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import log_ticks

UP, DOWN = "#26a69a", "#ef5350"
TARGET = "#c2185b"
REF = "#1f77b4"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#7f7f7f"]
LAYOUT = dict(template="plotly_white", hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10),
              title=dict(x=0, xanchor="left", font=dict(size=16)),
              legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0, font=dict(size=11)))


def _layout(fig, title, height, legend_rows=2):
    """Title on top, legend under the plot; reserve bottom space for the legend rows."""
    fig.update_layout(**LAYOUT, height=height + 22 * legend_rows)
    fig.update_layout(title_text=title, margin=dict(b=40 + 22 * legend_rows))


def _price_axis(fig, lo, hi, **kw):
    """Log price axis with readable round-number ticks (plotly's default log labels are cryptic)."""
    vals = log_ticks(lo * 0.9, hi * 1.1, 10)
    fig.update_yaxes(type="log", tickvals=vals, ticktext=[f"{v:g}" for v in vals], **kw)



def _ratio_axis(fig, lo, hi, **kw):
    """Log axis for ratios (1 = today's close) labelled as +/- percent."""
    cands = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 7, 10])
    vals = cands[(cands >= lo * 0.95) & (cands <= hi * 1.05)]
    fig.update_yaxes(type="log", tickvals=vals, ticktext=[f"{v - 1:+.0%}" for v in vals], **kw)


# ---------------------------------------------------------------- trend
def kline_chart(res):
    df, r, ma = res["df"], res["result"], res["ma"]
    n = len(df)
    x = np.arange(n)
    d = df.index
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="Price", increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
    L = r["long"]
    mid = L["k"] * x + L["b"]
    fig.add_trace(go.Scatter(x=d, y=np.exp(mid + 2 * L["sd"]), line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=d, y=np.exp(mid - 2 * L["sd"]), line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(128,128,128,0.12)", name="±2σ channel", hoverinfo="skip"), 1, 1)
    short = n < 126
    tag = (lambda q: f"{q['chg']:+.0%}") if short else (lambda q: f"{q['ann']:+.0%}/yr")
    fig.add_trace(go.Scatter(x=d, y=np.exp(mid), name=f"Regression {tag(L)} (R² {L['r2']:.2f})",
                             line=dict(color="black", dash="dash", width=1.2)), 1, 1)
    fig.add_trace(go.Scatter(x=d, y=np.exp(np.polyval(r["poly"]["coef"], x / n)), name="Cubic fit",
                             line=dict(color="#666", width=1)), 1, 1)
    for w, c in zip(sorted(ma)[-3:], ["#e9a23b", "#3b82c4", "#8e5cc4"]):
        fig.add_trace(go.Scatter(x=d, y=ma[w], name=f"MA{w}", line=dict(color=c, width=1.2)), 1, 1)
    S, sw = r["short"], r["short_win"]
    xs = x[-sw:]
    fig.add_trace(go.Scatter(x=d[-sw:], y=np.exp(S["k"] * xs + S["b"]), name=f"Last {sw}d {tag(S)}",
                             line=dict(color=TARGET, width=2.5)), 1, 1)
    for line, c, name in ((r["res_line"], DOWN, "Resistance"), (r["sup_line"], UP, "Support")):
        if line:
            k, b, idx = line
            xx = np.arange(idx[0], n)
            fig.add_trace(go.Scatter(x=d[idx[0]:], y=k * xx + b, name=name, line=dict(color=c, dash="dot", width=1.6)), 1, 1)
            fig.add_trace(go.Scatter(x=d[idx], y=k * idx + b, mode="markers", marker=dict(color=c, size=7),
                                     name=f"{name} pivots", showlegend=False), 1, 1)
    if "Volume" in df:
        colors = np.where(df["Close"] >= df["Open"], UP, DOWN)
        fig.add_trace(go.Bar(x=d, y=df["Volume"], marker_color=colors, name="Volume", showlegend=False), 2, 1)
    _layout(fig, f"{r['ticker']} daily chart with trend fits" + (" (listed < 6 months)" if short else ""), 700, 2)
    fig.update_layout(xaxis_rangeslider_visible=False)
    _price_axis(fig, df["Low"].min(), df["High"].max(), title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=1, label="1M", step="month", stepmode="backward"),
                                                 dict(count=3, label="3M", step="month", stepmode="backward"),
                                                 dict(count=6, label="6M", step="month", stepmode="backward"),
                                                 dict(count=1, label="1Y", step="year", stepmode="backward"),
                                                 dict(step="all", label="All")]), row=1, col=1)
    return fig


# ---------------------------------------------------------------- analogs
def _offsets(plot):
    L, H = plot["L"], plot["H"]
    return np.arange(-L + 1, 1), np.arange(0, H + 1), np.arange(-L + 1, H + 1)


def analog_overlay(plot):
    """Target price vs. the best match scaled to the target's price (hover shows the reference's real date/price)."""
    L = plot["L"]
    xt, xf, xa = _offsets(plot)
    seg = plot["segs"][0]
    k = plot["last"]
    cd = np.column_stack([[str(x) for x in seg["dates"]], seg["px"]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xt, y=plot["tgt"], name=f"{plot['target']} actual", line=dict(color=TARGET, width=3),
                             customdata=[str(x) for x in plot["tgt_dates"]],
                             hovertemplate="%{customdata}: %{y:.2f}"))
    lab = f"{seg['ticker']} {seg['start']} → {seg['end']} ({seg['speed']:g}x, r={seg['corr']:.2f})"
    fig.add_trace(go.Scatter(x=xt, y=seg["p"][:L] * k, name=lab, line=dict(color=REF, width=2),
                             customdata=cd[:L], hovertemplate="%{y:.2f} (" + seg["ticker"] + " %{customdata[0]}: $%{customdata[1]:.2f})"))
    fig.add_trace(go.Scatter(x=xf, y=seg["p"][L - 1:] * k, name=f"{seg['ticker']} afterwards → projection",
                             line=dict(color=REF, width=2, dash="dash"), customdata=cd[L - 1:],
                             hovertemplate="%{y:.2f} (" + seg["ticker"] + " %{customdata[0]}: $%{customdata[1]:.2f})"))
    fig.add_vline(x=0, line_dash="dot", line_color="gray", annotation_text="today", annotation_position="top")
    _layout(fig, f"{plot['target']} vs. most similar {seg['ticker']} window (aligned at today)", 500, 2)
    ys = np.concatenate([plot["tgt"], seg["p"] * k])
    _price_axis(fig, ys.min(), ys.max(), title_text=f"{plot['target']} price ({seg['ticker']} rescaled)")
    fig.update_xaxes(title_text="Trading days (0 = latest close; reference time-scaled to match)", hoverformat="+d")
    return fig


def norm_overlay(plot, n=5):
    """Top-n matching windows (different tickers) normalized to the target's last close."""
    L = plot["L"]
    xt, xf, xa = _offsets(plot)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xt, y=plot["tgt"] / plot["last"], name=f"{plot['target']} actual",
                             line=dict(color=TARGET, width=3.5), customdata=[str(x) for x in plot["tgt_dates"]],
                             hovertemplate="%{customdata}: %{y:.2f}×"))
    allp = [plot["tgt"] / plot["last"]]
    for seg, c in zip(plot["segs"][:n], PALETTE):
        cd = np.column_stack([[str(x) for x in seg["dates"]], seg["px"]])
        lab = f"{seg['ticker']} {seg['start']} → {seg['end']} ({seg['speed']:g}x, r={seg['corr']:.2f})"
        ht = seg["ticker"] + " %{customdata[0]}: $%{customdata[1]:.2f} (%{y:.2f}×)"
        fig.add_trace(go.Scatter(x=xt, y=seg["p"][:L], name=lab, line=dict(color=c, width=1.6), legendgroup=lab,
                                 customdata=cd[:L], hovertemplate=ht))
        fig.add_trace(go.Scatter(x=xf, y=seg["p"][L - 1:], name=lab, line=dict(color=c, width=1.6, dash="dash"),
                                 legendgroup=lab, showlegend=False, customdata=cd[L - 1:], hovertemplate=ht))
        allp.append(seg["p"])
    allp = np.concatenate(allp)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", annotation_text="today", annotation_position="top")
    fig.add_hline(y=1, line_color="lightgray")
    _layout(fig, f"Most similar {min(n, len(plot['segs']))} windows (dashed = what happened next)", 540, 3)
    _ratio_axis(fig, allp.min(), allp.max(), title_text=f"vs. {plot['target']} latest close")
    fig.update_xaxes(title_text="Trading days (0 = latest close; references time-scaled to match)", hoverformat="+d")
    return fig


def fan_chart(plot):
    """Forward paths of all matches after 'today', with median and interquartile band."""
    L = plot["L"]
    xt, xf, xa = _offsets(plot)
    P = np.array([s["p"][L - 1:] for s in plot["segs"]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xt, y=plot["tgt"] / plot["last"], name=plot["target"], line=dict(color=TARGET, width=3),
                             customdata=[str(x) for x in plot["tgt_dates"]], hovertemplate="%{customdata}: %{y:.2f}×"))
    for s in plot["segs"]:
        fig.add_trace(go.Scatter(x=xa, y=s["p"], line=dict(color=REF, width=0.8), opacity=0.35, showlegend=False,
                                 name=f"{s['ticker']} {s['start']}", hovertemplate=f"{s['ticker']} {s['start']}: " + "%{y:.2f}×"))
    fig.add_trace(go.Scatter(x=xf, y=P.max(0), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=P.min(0), line=dict(width=0), fill="tonexty", fillcolor="rgba(31,119,180,0.08)",
                             name="min–max", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=np.quantile(P, .75, 0), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=np.quantile(P, .25, 0), line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.25)", name="25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=np.median(P, 0), name="Median path", line=dict(color="black", width=2.5, dash="dash"),
                             hovertemplate="median: %{y:.2f}×"))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.add_hline(y=1, line_color="lightgray")
    allp = np.concatenate([P.ravel(), plot["tgt"] / plot["last"]])
    _layout(fig, f"What happened after the {len(plot['segs'])} similar windows (scenarios, not a forecast)", 500, 1)
    _ratio_axis(fig, allp.min(), allp.max(), title_text=f"vs. {plot['target']} latest close")
    fig.update_xaxes(title_text="Trading days (0 = latest close)", hoverformat="+d")
    return fig


# ---------------------------------------------------------------- scenario
def scenario_chart(plot):
    t, r = plot["target"], plot["ref"]
    h = np.asarray(plot["is_hist"])
    dates = np.array(plot["dates"])
    cd = np.column_stack([[str(x) for x in plot["ref_dates"]], plot["ref_px"]])
    ht = "%{y:.2f} (" + r + " %{customdata[0]}: $%{customdata[1]:.2f})"
    f = ~h
    f[np.argmax(~h) - 1 if (~h).any() else -1] = True  # connect projection to the anchor point
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot["tgt_dates"], y=plot["tgt"], name=f"{t} actual close", line=dict(color=TARGET, width=3),
                             hovertemplate="%{y:.2f}"))
    fig.add_trace(go.Scatter(x=dates[h], y=plot["proj"][h], name=f"{r} {plot['fit_start']} → {plot['anchor']} (fit, r={plot['corr']:.2f})",
                             line=dict(color=REF, width=1.8), opacity=0.8, customdata=cd[h], hovertemplate=ht))
    fig.add_trace(go.Scatter(x=dates[f], y=plot["proj"][f],
                             name=f"{r} {plot['anchor']} → {plot['ext_day']} path → {t} {'bull' if plot['up'] else 'bear'} case",
                             line=dict(color=REF, width=2.2, dash="dash"), customdata=cd[f], hovertemplate=ht))
    fig.add_trace(go.Scatter(x=[plot["target_date"]], y=[plot["target_px"]], mode="markers+text",
                             marker=dict(color="#e65100", size=12), text=[f"target ≈ {plot['target_px']:.4g}"],
                             textposition="middle left", name=f"Scenario target ({r} {'high' if plot['up'] else 'low'} {plot['ext_px']:.2f} on {plot['ext_day']})",
                             hovertemplate="target %{y:.2f} on %{x|%Y-%m-%d}"))
    fig.add_trace(go.Scatter(x=[plot["worst_date"]], y=[plot["worst_px"]], mode="markers", marker=dict(color="#555", size=9, symbol="x"),
                             name=f"Largest {'drawdown' if plot['up'] else 'rebound'} {plot['worst_pct']:+.0%} ({r} {plot['top_date']} → {plot['bottom_date']})",
                             hovertemplate="%{y:.2f}"))
    fig.add_vline(x=plot["today"], line_dash="dot", line_color="gray")
    _layout(fig, f"{'Bull' if plot['up'] else 'Bear'} case: {t} following {r}'s path "
                 f"(anchor {r} {plot['anchor']} ↔ {t} today, time scale {plot['speed']:g}x)", 600, 3)
    ys = np.concatenate([plot["tgt"], plot["proj"], [plot["target_px"]]])
    _price_axis(fig, ys.min(), ys.max(), title_text=f"{t} price")
    return fig
