"""Interactive Plotly charts for the web UI (English labels).

Design follows the dataviz method: the target stock is the accent (categorical slot 1), references take
the following slots in fixed order, context is gray; light and dark each use their own validated steps
(see PALETTE below — validated with validate_palette.js, adjacent and all-pairs where marks share a plot).

Each builder takes the structured output of a script's run() plus mode ("light"/"dark") and returns
(figure, table DataFrame) — the table is the chart's accessible twin.
  kline_chart(res)          ← kline_fit.run()
  analog_overlay(plot)      ← analog.run()["plot"]
  norm_overlay(plot, n)     ← screen.run()["plot"]
  fan_chart(plot)           ← analog/screen ["plot"]
  scenario_chart(plot)      ← scenario.run()["plot"]
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import log_ticks

# ---------------------------------------------------------------- tokens
PALETTE = {
    "light": dict(
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        up="#1baf7a", down="#e34948",
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", border="rgba(11,11,11,0.10)",
        wash="rgba(42,120,214,0.14)", wash_faint="rgba(42,120,214,0.06)", gray_wash="rgba(137,135,129,0.12)",
    ),
    "dark": dict(
        series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        up="#199e70", down="#e66767",
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)",
        wash="rgba(57,135,229,0.22)", wash_faint="rgba(57,135,229,0.09)", gray_wash="rgba(137,135,129,0.16)",
    ),
}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def _t(mode):
    return PALETTE["dark" if mode == "dark" else "light"]


def _base(fig, t, title, height, legend_rows=1):
    """Shared chrome: recessive hairline grid, crosshair, one tooltip for every series, legend under the plot."""
    fig.update_layout(
        height=height + 22 * legend_rows,
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(family=FONT, size=12, color=t["ink2"]),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=t["ink"])),
        margin=dict(l=12, r=12, t=48, b=36 + 22 * legend_rows),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"], font=dict(family=FONT, size=12, color=t["ink"])),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
                    font=dict(size=12, color=t["ink2"]), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor=t["axis"], linewidth=1, ticks="outside", tickcolor=t["axis"],
                     tickfont=dict(color=t["muted"]), title_font=dict(color=t["muted"], size=12),
                     automargin=True, showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
                     spikecolor=t["axis"], spikedash="solid", zeroline=False)
    fig.update_yaxes(automargin=True, gridcolor=t["grid"], gridwidth=1, linecolor=t["axis"], zeroline=False,
                     tickfont=dict(color=t["muted"]), title_font=dict(color=t["muted"], size=12))
    return fig


def _price_axis(fig, lo, hi, **kw):
    """Log price axis with round-number ticks (plotly's default log labels are cryptic)."""
    vals = log_ticks(lo * 0.9, hi * 1.1, 10)
    fig.update_yaxes(type="log", tickvals=vals, ticktext=[f"{v:g}" for v in vals], **kw)


def _ratio_axis(fig, lo, hi, **kw):
    """Log axis for ratios (1 = today's close) labelled as +/- percent."""
    cands = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 7, 10])
    vals = cands[(cands >= lo * 0.95) & (cands <= hi * 1.05)]
    fig.update_yaxes(type="log", tickvals=vals, ticktext=[f"{v - 1:+.0%}" for v in vals], **kw)


def _today(fig, t, x):
    """'Today' reference: a solid hairline in the axis tone (not dashed — dashing reads as a threshold)."""
    fig.add_vline(x=x, line_width=1, line_color=t["axis"])
    fig.add_annotation(x=x, y=1, yref="paper", text="today", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color=t["muted"]))


def _end_label(fig, t, x, y, text, xanchor="left", row=None, col=None):
    """Selective direct label at a line end, in text ink (never the series color).
    All our value axes are log scale, where Plotly positions annotations in log10 units."""
    fig.add_annotation(x=x, y=float(np.log10(y)), text=text, row=row, col=col, showarrow=False, xanchor=xanchor, xshift=6 if xanchor == "left" else -6,
                       font=dict(size=12, color=t["ink"]), bgcolor=t["surface"])


def _dot(t, color):
    return dict(color=color, size=9, line=dict(color=t["surface"], width=2))  # 2px surface ring


def _ht(value_fmt, extra=""):
    """Tooltip row: value first (bold), label after; <extra></extra> stops Plotly prefixing the trace name."""
    return f"<b>{value_fmt}</b>{extra}<extra></extra>"


# ---------------------------------------------------------------- trend
def kline_chart(res, mode="light"):
    t = _t(mode)
    df, r, ma = res["df"], res["result"], res["ma"]
    n = len(df)
    x = np.arange(n)
    d = df.index
    accent = t["series"][0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
    L = r["long"]
    mid = L["k"] * x + L["b"]
    hi_ch, lo_ch = np.exp(mid + 2 * L["sd"]), np.exp(mid - 2 * L["sd"])
    short = n < 126
    tag = (lambda q: f"{q['chg']:+.0%}") if short else (lambda q: f"{q['ann']:+.0%}/yr")
    # context first (drawn underneath): channel wash + whole-period regression in gray
    fig.add_trace(go.Scatter(x=d, y=hi_ch, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=d, y=lo_ch, line=dict(width=0), fill="tonexty", fillcolor=t["gray_wash"],
                             name="±2σ channel", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=d, y=np.exp(mid), name=f"Whole-period trend {tag(L)} (R² {L['r2']:.2f})",
                             line=dict(color=t["muted"], width=1.5),
                             hovertemplate=_ht("%{y:.2f}", " trend")), 1, 1)
    fig.add_trace(go.Scatter(x=d, y=np.exp(np.polyval(r["poly"]["coef"], x / n)), name="Cubic fit",
                             line=dict(color=t["muted"], width=1.5, dash="dot"), visible="legendonly",
                             hovertemplate=_ht("%{y:.2f}", " cubic")), 1, 1)
    for w in sorted(ma)[-3:]:
        first = w == sorted(ma)[-3:][0]
        fig.add_trace(go.Scatter(x=d, y=ma[w], name=f"MA{w}", line=dict(color=t["ink2"] if first else t["muted"], width=1.5),
                                 visible=True if first else "legendonly", hovertemplate=_ht("%{y:.2f}", f" MA{w}")), 1, 1)
    for line, name, sym in ((r["res_line"], "Resistance", "triangle-down"), (r["sup_line"], "Support", "triangle-up")):
        if line:
            k, b, idx = line
            xx = np.arange(idx[0], n)
            fig.add_trace(go.Scatter(x=d[idx[0]:], y=k * xx + b, name=name, legendgroup=name,
                                     line=dict(color=t["ink2"], width=1.2, dash="dot"),
                                     hovertemplate=_ht("%{y:.2f}", f" {name.lower()}")), 1, 1)
            fig.add_trace(go.Scatter(x=d[idx], y=k * idx + b, mode="markers", legendgroup=name, showlegend=False,
                                     marker=dict(symbol=sym, color=t["ink2"], size=9, line=dict(color=t["surface"], width=2)),
                                     hoverinfo="skip"), 1, 1)
    # candles: hollow up / filled down = secondary encoding for the aqua↔red pair (CVD ΔE 6.5–6.9)
    fig.add_trace(go.Candlestick(x=d, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price",
                                 increasing=dict(line=dict(color=t["up"], width=1), fillcolor=t["surface"]),
                                 decreasing=dict(line=dict(color=t["down"], width=1), fillcolor=t["down"]),
                                 whiskerwidth=0), 1, 1)
    # the story: recent trend in the accent
    S, sw = r["short"], r["short_win"]
    xs = x[-sw:]
    fig.add_trace(go.Scatter(x=d[-sw:], y=np.exp(S["k"] * xs + S["b"]), name=f"Last {sw}d trend {tag(S)}",
                             line=dict(color=accent, width=2.5), hovertemplate=_ht("%{y:.2f}", f" last-{sw}d trend")), 1, 1)
    if "Volume" in df:
        fig.add_trace(go.Bar(x=d, y=df["Volume"], marker=dict(color=t["muted"], line_width=0), opacity=0.55,
                             name="Volume", showlegend=False, hovertemplate=_ht("%{y:,.0f}", " volume")), 2, 1)
    _end_label(fig, t, d[-1], r["last"], f"{r['last']:.2f}", row=1, col=1)
    _base(fig, t, f"{r['ticker']} daily · trend fits" + (" · listed < 6 months" if short else ""), 640, 2)
    fig.update_layout(xaxis_rangeslider_visible=False, bargap=0.2)
    _price_axis(fig, df["Low"].min(), df["High"].max(), row=1, col=1)
    fig.update_yaxes(title_text=None, row=2, col=1, tickformat="~s", nticks=3)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_xaxes(rangeselector=dict(
        buttons=[dict(count=1, label="1M", step="month", stepmode="backward"),
                 dict(count=3, label="3M", step="month", stepmode="backward"),
                 dict(count=6, label="6M", step="month", stepmode="backward"),
                 dict(count=1, label="1Y", step="year", stepmode="backward"),
                 dict(step="all", label="All")],
        bgcolor=t["surface"], activecolor=t["grid"], bordercolor=t["axis"], borderwidth=1,
        font=dict(color=t["ink2"], size=11), x=0, xanchor="left", y=1.0, yanchor="bottom"), row=1, col=1)
    table = pd.DataFrame({"date": [x.date() for x in d], "open": df["Open"].values, "high": df["High"].values,
                          "low": df["Low"].values, "close": df["Close"].values,
                          **{f"MA{w}": ma[w] for w in sorted(ma)[-3:]},
                          "trend": np.exp(mid), "channel_low": lo_ch, "channel_high": hi_ch})
    if "Volume" in df:
        table.insert(5, "volume", df["Volume"].values)
    return fig, table.iloc[::-1].reset_index(drop=True)


# ---------------------------------------------------------------- analogs
def _offsets(plot):
    L, H = plot["L"], plot["H"]
    return np.arange(-L + 1, 1), np.arange(0, H + 1), np.arange(-L + 1, H + 1)


def _seg_label(seg):
    return f"{seg['ticker']} {seg['start']} → {seg['end']} ({seg['speed']:g}x, r {seg['corr']:.2f})"


def analog_overlay(plot, mode="light"):
    """Target price vs. the best match rescaled to the target's price; hover shows the reference's real date/price."""
    t = _t(mode)
    L = plot["L"]
    xt, xf, _ = _offsets(plot)
    seg = plot["segs"][0]
    k = plot["last"]
    tgt_c, ref_c = t["series"][0], t["series"][1]
    cd = np.column_stack([[str(x) for x in seg["dates"]], seg["px"]])
    ref_ht = _ht("%{y:.2f}", f" {seg['ticker']} rescaled · real " + "%{customdata[0]} $%{customdata[1]:.2f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xt, y=seg["p"][:L] * k, name=_seg_label(seg), legendgroup="ref",
                             line=dict(color=ref_c, width=2), customdata=cd[:L], hovertemplate=ref_ht))
    fig.add_trace(go.Scatter(x=xf, y=seg["p"][L - 1:] * k, name=f"{seg['ticker']} afterwards (projection)",
                             line=dict(color=ref_c, width=2, dash="dash"), customdata=cd[L - 1:], hovertemplate=ref_ht))
    fig.add_trace(go.Scatter(x=xt, y=plot["tgt"], name=f"{plot['target']}", line=dict(color=tgt_c, width=2.5),
                             customdata=[str(x) for x in plot["tgt_dates"]],
                             hovertemplate=_ht("%{y:.2f}", f" {plot['target']} · " + "%{customdata}")))
    fig.add_trace(go.Scatter(x=[0], y=[plot["tgt"][-1]], mode="markers", marker=_dot(t, tgt_c), showlegend=False,
                             hoverinfo="skip"))
    _end_label(fig, t, xf[-1], seg["p"][-1] * k, f"{seg['p'][-1] * k:.2f}")
    _today(fig, t, 0)
    _base(fig, t, f"{plot['target']} vs. the most similar {seg['ticker']} window", 480, 2)
    ys = np.concatenate([plot["tgt"], seg["p"] * k])
    _price_axis(fig, ys.min(), ys.max(), title_text=f"{plot['target']} price ({seg['ticker']} rescaled)")
    fig.update_xaxes(title_text="Trading days from today (reference time-scaled to match)", hoverformat="+d")
    table = pd.DataFrame({"day": np.r_[xt, xf[1:]],
                          f"{plot['target']}_date": [str(x) for x in plot["tgt_dates"]] + [""] * (len(xf) - 1),
                          f"{plot['target']}_close": np.r_[plot["tgt"], [np.nan] * (len(xf) - 1)],
                          f"{seg['ticker']}_date": [str(x) for x in seg["dates"]],
                          f"{seg['ticker']}_close": seg["px"],
                          f"{seg['ticker']}_rescaled": seg["p"] * k})
    return fig, table


def norm_overlay(plot, n=5, mode="light"):
    """Target plus top-n matches (different tickers) normalized to the target's last close."""
    t = _t(mode)
    L = plot["L"]
    xt, xf, _ = _offsets(plot)
    fig = go.Figure()
    allp = [plot["tgt"] / plot["last"]]
    rows = {}
    for seg, c in zip(plot["segs"][:n], t["series"][1:]):  # references take slots 2.. in fixed order
        cd = np.column_stack([[str(x) for x in seg["dates"]], seg["px"]])
        ht = _ht("%{y:.2f}×", f" {seg['ticker']} · real " + "%{customdata[0]} $%{customdata[1]:.2f}")
        lab = _seg_label(seg)
        fig.add_trace(go.Scatter(x=xt, y=seg["p"][:L], name=lab, legendgroup=lab, line=dict(color=c, width=2),
                                 customdata=cd[:L], hovertemplate=ht))
        fig.add_trace(go.Scatter(x=xf, y=seg["p"][L - 1:], name=lab, legendgroup=lab, showlegend=False,
                                 line=dict(color=c, width=2, dash="dash"), customdata=cd[L - 1:], hovertemplate=ht))
        allp.append(seg["p"])
        rows[f"{seg['ticker']} {seg['start']}"] = seg["p"]
    fig.add_trace(go.Scatter(x=xt, y=plot["tgt"] / plot["last"], name=plot["target"],
                             line=dict(color=t["series"][0], width=3), customdata=[str(x) for x in plot["tgt_dates"]],
                             hovertemplate=_ht("%{y:.2f}×", f" {plot['target']} · " + "%{customdata}")))
    fig.add_trace(go.Scatter(x=[0], y=[1], mode="markers", marker=_dot(t, t["series"][0]), showlegend=False, hoverinfo="skip"))
    allp = np.concatenate(allp)
    _today(fig, t, 0)
    fig.add_hline(y=1, line_width=1, line_color=t["axis"])
    _base(fig, t, f"{plot['target']} and its {min(n, len(plot['segs']))} most similar windows · dashed = what followed", 520, 3)
    _ratio_axis(fig, allp.min(), allp.max(), title_text=f"vs. {plot['target']} latest close")
    fig.update_xaxes(title_text="Trading days from today (references time-scaled to match)", hoverformat="+d")
    days = np.r_[xt, xf[1:]]
    table = pd.DataFrame({"day": days, plot["target"]: np.r_[plot["tgt"] / plot["last"], [np.nan] * (len(xf) - 1)],
                          **rows})
    for c in table.columns[1:]:
        table[c] = table[c] - 1
    return fig, table


def fan_chart(plot, mode="light"):
    """Forward paths of all matches after today: individual paths in gray, IQR band and median emphasised."""
    t = _t(mode)
    L = plot["L"]
    xt, xf, xa = _offsets(plot)
    P = np.array([s["p"][L - 1:] for s in plot["segs"]])
    q1, med, q3 = np.quantile(P, .25, 0), np.median(P, 0), np.quantile(P, .75, 0)
    fig = go.Figure()
    for s in plot["segs"]:
        fig.add_trace(go.Scatter(x=xa, y=s["p"], line=dict(color=t["muted"], width=1), opacity=0.45, showlegend=False,
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=P.max(0), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=P.min(0), line=dict(width=0), fill="tonexty", fillcolor=t["wash_faint"],
                             name="Min–max", hovertemplate=_ht("%{y:.2f}×", " min")))
    fig.add_trace(go.Scatter(x=xf, y=q3, line=dict(width=0), showlegend=False, hovertemplate=_ht("%{y:.2f}×", " 75th pct")))
    fig.add_trace(go.Scatter(x=xf, y=q1, line=dict(width=0), fill="tonexty", fillcolor=t["wash"],
                             name="Middle 50%", hovertemplate=_ht("%{y:.2f}×", " 25th pct")))
    fig.add_trace(go.Scatter(x=xf, y=med, name="Median path", line=dict(color=t["series"][0], width=2.5, dash="dash"),
                             hovertemplate=_ht("%{y:.2f}×", " median")))
    fig.add_trace(go.Scatter(x=xt, y=plot["tgt"] / plot["last"], name=plot["target"], line=dict(color=t["series"][0], width=2.5),
                             customdata=[str(x) for x in plot["tgt_dates"]],
                             hovertemplate=_ht("%{y:.2f}×", f" {plot['target']} · " + "%{customdata}")))
    _end_label(fig, t, xf[-1], med[-1], f"median {med[-1] - 1:+.0%}")
    _today(fig, t, 0)
    fig.add_hline(y=1, line_width=1, line_color=t["axis"])
    allp = np.concatenate([P.ravel(), plot["tgt"] / plot["last"]])
    _base(fig, t, f"What followed the {len(plot['segs'])} similar windows · scenarios, not a forecast", 460, 1)
    _ratio_axis(fig, allp.min(), allp.max(), title_text=f"vs. {plot['target']} latest close")
    fig.update_xaxes(title_text="Trading days from today", hoverformat="+d")
    table = pd.DataFrame({"day": xf, "min": P.min(0) - 1, "25th pct": q1 - 1, "median": med - 1,
                          "75th pct": q3 - 1, "max": P.max(0) - 1,
                          "median price": med * plot["last"]})
    return fig, table


# ---------------------------------------------------------------- scenario
def scenario_chart(plot, mode="light"):
    tk = _t(mode)
    t_, r = plot["target"], plot["ref"]
    tgt_c, ref_c = tk["series"][0], tk["series"][1]
    h = np.asarray(plot["is_hist"])
    dates = np.array(plot["dates"])
    cd = np.column_stack([[str(x) for x in plot["ref_dates"]], plot["ref_px"]])
    ht = _ht("%{y:.2f}", f" {t_} scenario · {r} real " + "%{customdata[0]} $%{customdata[1]:.2f}")
    f = ~h
    if f.any():
        f[max(np.argmax(f) - 1, 0)] = True  # connect the projection to the anchor point
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[h], y=plot["proj"][h], name=f"{r} {plot['fit_start']} → {plot['anchor']} (fit, r {plot['corr']:.2f})",
                             line=dict(color=ref_c, width=2), opacity=0.85, customdata=cd[h], hovertemplate=ht))
    fig.add_trace(go.Scatter(x=dates[f], y=plot["proj"][f],
                             name=f"{r} {plot['anchor']} → {plot['ext_day']} path ({'bull' if plot['up'] else 'bear'} case)",
                             line=dict(color=ref_c, width=2, dash="dash"), customdata=cd[f], hovertemplate=ht))
    fig.add_trace(go.Scatter(x=plot["tgt_dates"], y=plot["tgt"], name=f"{t_} actual", line=dict(color=tgt_c, width=2.5),
                             hovertemplate=_ht("%{y:.2f}", f" {t_} actual")))
    fig.add_trace(go.Scatter(x=[plot["target_date"]], y=[plot["target_px"]], mode="markers", marker=_dot(tk, ref_c),
                             name=f"Scenario target ({r} {'high' if plot['up'] else 'low'} {plot['ext_px']:.2f} on {plot['ext_day']})",
                             hovertemplate=_ht("%{y:.2f}", " scenario target")))
    fig.add_trace(go.Scatter(x=[plot["worst_date"]], y=[plot["worst_px"]], mode="markers",
                             marker=dict(symbol="x-thin", size=12, line=dict(color=tk["ink2"], width=2)),
                             name=f"Largest {'drawdown' if plot['up'] else 'rebound'} {plot['worst_pct']:+.0%} ({r} {plot['top_date']} → {plot['bottom_date']})",
                             hovertemplate=_ht("%{y:.2f}", " after largest " + ("drawdown" if plot["up"] else "rebound"))))
    _end_label(fig, tk, plot["target_date"], plot["target_px"], f"target {plot['target_px']:.4g}", xanchor="right")
    _today(fig, tk, plot["today"])
    _base(fig, tk, f"{'Bull' if plot['up'] else 'Bear'} case · {t_} following {r}'s path "
                   f"(anchor {r} {plot['anchor']} = {t_} today, time scale {plot['speed']:g}x)", 560, 3)
    ys = np.concatenate([plot["tgt"], plot["proj"], [plot["target_px"]]])
    _price_axis(fig, ys.min(), ys.max(), title_text=f"{t_} price")
    table = pd.DataFrame({"date": [x.date() for x in dates], "phase": np.where(h, "fit", "projection"),
                          f"{t_}_scenario": plot["proj"], f"{r}_date": [str(x) for x in plot["ref_dates"]],
                          f"{r}_close": plot["ref_px"]})
    return fig, table
