"""Web UI for stock-analog-fit.

Run:  streamlit run app.py
"""
import os
import sys
import tempfile
from datetime import date
from types import SimpleNamespace

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import analog  # noqa: E402
import charts  # noqa: E402
import kline_fit  # noqa: E402
import scenario  # noqa: E402
import screen  # noqa: E402

st.set_page_config(page_title="Stock Analog Fit", page_icon="📈", layout="wide")

PAGES = {
    "📈 Trend fit": "Candlestick chart with regression channel, moving averages, support/resistance, pattern and breakout checks.",
    "🔁 Analog (one reference)": "Find the windows in one reference stock's history that most resemble the target's recent path, overlay them, and see what happened next.",
    "🔎 Analog screen (many stocks)": "Search a basket of stocks for the windows most similar to the target and summarize the spread of what followed.",
    "🎯 Scenario projection": "Project the target along a chosen historical run of a reference stock (bull or bear case): target price, timing, drawdowns.",
}
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
PLOTLY_CFG = {"displaylogo": False, "scrollZoom": True,
              "toImageButtonOptions": {"format": "png", "scale": 2}}

# Messages raised by the scripts (Chinese, for the CLI) → English for the UI.
ERRORS = {
    "没取到数据": "No data returned for this ticker. Check the symbol, your network access to Yahoo Finance, or upload a CSV instead.",
    "数据不足": "Not enough price history (need at least 20 trading days).",
    "没有找到满足振幅条件": "No window with a comparable price swing was found. Widen the swing-ratio range, allow more time scales, or add more tickers.",
    "锚点区间内没有振幅可比": "No anchor with a comparable price swing in the search range. Widen the swing-ratio range or change the anchor search dates.",
    "锚点落在极值点之后": "The anchor falls after the path's extreme point. Move the anchor search end date earlier.",
}


# ---------- helpers ----------
def upload(label, key):
    """Optional CSV upload; returns a temp file path or None."""
    f = st.file_uploader(label, type="csv", key=key,
                         help="Daily CSV with Date, Open, High, Low, Close columns (e.g. Yahoo Finance → Historical Data → Download).")
    if f is None:
        return None
    path = os.path.join(tempfile.gettempdir(), f"saf_{key}.csv")
    with open(path, "wb") as fh:
        fh.write(f.getvalue())
    return path


def mode():
    """Viewer's current theme; charts use that mode's own validated palette steps."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001
        return "light"


def chart(fig_table, key, name, pct_cols=(), num_fmt="{:.2f}"):
    """A chart with its table twin (every value reachable without hovering)."""
    fig, table = fig_table
    tab_chart, tab_table = st.tabs(["Chart", "Table"])
    with tab_chart:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG, key=key, theme=None)
    with tab_table:
        fmt = {c: num_fmt for c in table.select_dtypes("number").columns if c != "day"}
        fmt.update({c: "{:+.1%}" for c in pct_cols})
        if "volume" in table:
            fmt["volume"] = "{:,.0f}"
        st.dataframe(table.style.format(fmt, na_rep=""), use_container_width=True, hide_index=True, height=360)
        st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"), file_name=name,
                           mime="text/csv", key=f"dl_{key}")


def show_table(df, name, num_cols=()):
    fmt = {c: "{:+.1%}" for c in df.columns if str(c).startswith(("+", "max"))}
    fmt.update({c: "{:.2f}" for c in num_cols if c in df.columns})
    st.dataframe(df.style.format(fmt), use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=name,
                       mime="text/csv", key=f"dl_{name}")


def run_safely(fn, *args, **kw):
    """Run an analysis; turn script errors and network failures into readable messages."""
    try:
        with st.spinner("Fetching data and computing… (the multi-stock screen takes ~15–30 s the first time)"):
            return fn(*args, **kw)
    except SystemExit as e:
        msg = str(e)
        st.error(next((v for k, v in ERRORS.items() if k in msg), msg))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if any(x in msg for x in ("403", "Connection", "curl", "resolve")):
            st.error("Cannot reach Yahoo Finance. Allow query1/query2.finance.yahoo.com, fc.yahoo.com and "
                     "guce.yahoo.com in your network settings, or upload CSV files instead.\n\n" + msg[:300])
        else:
            st.exception(e)
    return None


def kpi(col, label, value, note=None):
    col.metric(label, value)
    if note:
        col.caption(note)


def horizon_metrics(summary, last, target):
    cols = st.columns(len(summary))
    for c, (h, v) in zip(cols, summary.items()):
        c.metric(f"Median price in {h} trading days", f"{v['price']:.2f}", f"{v['median']:+.1%}")
        extra = f"{v['up']:.0%} of cases up"
        if "q1" in v:
            extra += f" · IQR {v['q1']:+.0%} to {v['q3']:+.0%}"
        c.caption(extra)
    st.caption(f"Relative to {target}'s latest close of {last:.2f}.")


NUM = ["corr", "amp", "p0", "p1"]
TABLE_HELP = ("**days** = trading days the reference actually took · **speed** = time-scale factor · "
              "**corr** = shape correlation · **amp** = swing ratio vs. target · **p0/p1** = reference price at "
              "window start/end · **+Nd** = move N target-trading-days later · **maxup/maxdd** = best/worst point afterwards")


def analog_params(prefix, default_window=60):
    with st.expander("Advanced settings"):
        c1, c2, c3 = st.columns(3)
        window = c1.number_input("Comparison window (target's last N trading days)", 15, 500, default_window, key=f"{prefix}_w")
        horizon = c2.number_input("Projection horizon (trading days)", 5, 120, 40, key=f"{prefix}_h")
        amp = c3.slider("Swing-ratio range", 0.2, 3.0, (0.6, 1.6), 0.05, key=f"{prefix}_amp",
                        help="Reference window's price swing ÷ target's swing. A wide range lets flat windows qualify on shape alone.")
        scales = st.multiselect("Allowed time scales", SCALES, [0.5, 0.75, 1.0, 1.5, 2.0, 3.0], key=f"{prefix}_sc",
                                help="How many times as many days the reference may take to trace the same shape. Pick only 1 for same-speed matches.")
    return window, horizon, amp, scales or [1.0]


# ---------- pages ----------
def page_trend():
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        tickers = c1.text_input("Tickers (comma-separated)", "NIO, SECZ").upper()
        period = c2.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2)
        csv = upload("Or upload a CSV for the first ticker (optional)", "trend_csv")
        if st.button("Run", type="primary", key="go_trend"):
            out = []
            for i, t in enumerate([x.strip() for x in tickers.split(",") if x.strip()]):
                res = run_safely(kline_fit.run, t, period, csv if i == 0 else None)
                if res:
                    out.append((t, res))
            st.session_state["trend"] = out
    for t, res in st.session_state.get("trend", []):
        st.subheader(t)
        r = res["result"]
        short = r["n"] < 126
        tag = "change" if short else "annualized"
        c = st.columns(4)
        kpi(c[0], "Latest close", f"{r['last']:.2f}", f"{r['start']} → {r['end']} · {r['n']} bars")
        kpi(c[1], f"Last {r['short_win']} days ({tag})",
            f"{(r['short']['chg'] if short else r['short']['ann']):+.1%}", f"R² {r['short']['r2']:.2f}")
        kpi(c[2], f"Whole period ({tag})", f"{(r['long']['chg'] if short else r['long']['ann']):+.1%}",
            f"R² {r['long']['r2']:.2f}")
        kpi(c[3], "Position in ±2σ channel", f"{r['long']['z']:+.2f}σ")
        st.markdown("\n".join(f"- {x}" for x in res["lines_en"]))
        chart(charts.kline_chart(res, mode()), f"k_{t}", f"{t}_prices_and_fits.csv")
        st.divider()


def page_analog():
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        target = c1.text_input("Target ticker", "SECZ").upper().strip()
        ref = c2.text_input("Reference ticker (full history)", "NIO").upper().strip()
        top = c3.number_input("Top N windows", 1, 10, 5)
        window, horizon, amp, scales = analog_params("an")
        u1, u2 = st.columns(2)
        with u1:
            tcsv = upload("Target CSV (optional)", "an_t")
        with u2:
            rcsv = upload("Reference CSV (optional)", "an_r")
        if st.button("Find analogs", type="primary", key="go_an"):
            a = SimpleNamespace(target=target, ref=ref, window=window, horizon=horizon, top=top,
                                scales=",".join(f"{x:g}" for x in scales), amp_min=amp[0], amp_max=amp[1],
                                target_csv=tcsv, ref_csv=rcsv)
            r = run_safely(analog.run, a)
            st.session_state["analog"] = (a, r) if r else None
    res = st.session_state.get("analog")
    if res:
        a, r = res
        horizon_metrics(r["summary"], r["last"], a.target)
        chart(charts.analog_overlay(r["plot"], mode()), "an_overlay", f"{a.target}_vs_{a.ref}_overlay.csv")
        chart(charts.fan_chart(r["plot"], mode()), "an_fan", f"{a.target}_vs_{a.ref}_paths.csv",
              pct_cols=["min", "25th pct", "median", "75th pct", "max"])
        st.markdown("**Matching windows**")
        show_table(r["table"], f"{a.target}_vs_{a.ref}_analog.csv", NUM)
        st.caption(TABLE_HELP)


def page_screen():
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        target = c1.text_input("Target ticker", "SECZ", key="sc_t").upper().strip()
        since = c2.date_input("Candidate history since", date(2012, 1, 1), key="sc_since")
        top = c3.number_input("Pool top N windows", 3, 30, 10, key="sc_top")
        universe = st.text_area("Candidate tickers (comma-separated, editable)", screen.DEFAULT_UNIVERSE.replace(",", ", "), height=110)
        c4, c5 = st.columns(2)
        per = c4.number_input("Max windows per ticker", 1, 5, 2)
        min_corr = c5.slider("Minimum correlation to include", 0.5, 0.99, 0.75, 0.01)
        window, horizon, amp, scales = analog_params("sc")
        tcsv = upload("Target CSV (optional)", "sc_csv")
        if st.button("Run screen", type="primary", key="go_sc"):
            a = SimpleNamespace(target=target, universe=universe.replace(" ", "").replace("\n", ","),
                                since=str(since), window=window, horizon=horizon, per_ticker=per, top=top,
                                min_corr=min_corr, scales=",".join(f"{x:g}" for x in scales),
                                amp_min=amp[0], amp_max=amp[1], target_csv=tcsv)
            r = run_safely(screen.run, a)
            st.session_state["screen"] = (a, r) if r else None
    res = st.session_state.get("screen")
    if res:
        a, r = res
        horizon_metrics(r["summary"], r["last"], a.target)
        fig_t = charts.norm_overlay(r["plot"], 5, mode())
        chart(fig_t, "sc_overlay", f"{a.target}_screen_overlay.csv", pct_cols=list(fig_t[1].columns[1:]))
        chart(charts.fan_chart(r["plot"], mode()), "sc_fan", f"{a.target}_screen_paths.csv",
              pct_cols=["min", "25th pct", "median", "75th pct", "max"])
        st.markdown("**Pooled windows**")
        show_table(r["table"], f"{a.target}_screen_top.csv", NUM)
        st.caption(TABLE_HELP)
        st.markdown("**Best match per ticker** (sorted by correlation)")
        show_table(r["all_best"], f"{a.target}_screen_all.csv", NUM)
        st.info("Caveats: very short, nearly one-directional windows correlate highly with almost any rally, so they "
                "discriminate poorly. The default basket only contains stocks that became well known (survivorship "
                "bias), which can make results look too optimistic.")


def page_scenario():
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 2])
        target = c1.text_input("Target ticker", "SECZ", key="sn_t").upper().strip()
        ref = c2.text_input("Reference ticker", "NIO", key="sn_r").upper().strip()
        direction = c3.radio("Case", ["up", "down"], horizontal=True,
                             format_func=lambda x: "Bull (to the period high)" if x == "up" else "Bear (to the period low)")
        d1, d2 = st.columns(2)
        ps = d1.date_input("Reference path start", date(2020, 6, 1), min_value=date(1990, 1, 1))
        pe = d2.date_input("Reference path end (approx.)", date(2021, 1, 11), min_value=date(1990, 1, 1),
                           help="The highest (bull) or lowest (bear) price between start and this date becomes the end point.")
        with st.expander("Advanced settings"):
            e1, e2, e3 = st.columns(3)
            af = e1.date_input("Anchor search from (optional)", None, key="sn_af")
            at = e2.date_input("Anchor search to (optional)", None, key="sn_at",
                               help="Defaults to the first half of the path (start → extreme point).")
            window = e3.number_input("Comparison window (target's last N days)", 15, 500, 60, key="sn_w")
            e4, e5 = st.columns(2)
            sc = e4.slider("Time-scale range", 0.25, 4.0, (0.5, 3.0), 0.25, key="sn_sc")
            amp = e5.slider("Swing-ratio range", 0.2, 3.0, (0.8, 1.25), 0.05, key="sn_amp")
        u1, u2 = st.columns(2)
        with u1:
            tcsv = upload("Target CSV (optional)", "sn_tc")
        with u2:
            rcsv = upload("Reference CSV (optional)", "sn_rc")
        if st.button("Project scenario", type="primary", key="go_sn"):
            a = SimpleNamespace(target=target, ref=ref, path_start=str(ps), path_end=str(pe), direction=direction,
                                anchor_from=str(af) if af else None, anchor_to=str(at) if at else None,
                                window=window, scales=f"{sc[0]}:{sc[1]}:0.25", amp_min=amp[0], amp_max=amp[1],
                                target_csv=tcsv, ref_csv=rcsv)
            r = run_safely(scenario.run, a)
            st.session_state["scenario"] = (a, r) if r else None
    res = st.session_state.get("scenario")
    if res:
        a, r = res
        k = r["key"]
        if k["corr"] < 0.6:
            st.warning(f"Weak fit (correlation {k['corr']:.2f} < 0.6): {a.target}'s current shape does not really "
                       f"resemble this {a.ref} period. Read the projection as a 'what if it copied this path' exercise.")
        c = st.columns(4)
        c[0].metric("Scenario target", f"{k['target']:.2f}", f"{k['mult'] - 1:+.0%} vs. latest close")
        c[0].caption(f"Latest close {k['last']:.2f}")
        kpi(c[1], "Reached around", str(k["target_date"]), f"≈ {k['days']} trading days from now")
        kpi(c[2], "Anchor", str(k["anchor"]), f"{a.ref} close {k['anchor_px']:.2f} · r {k['corr']:.2f} · {k['speed']:g}x")
        kpi(c[3], "Largest " + ("drawdown" if a.direction == "up" else "rebound") + " on the way",
            f"{k['worst']:+.0%}", f"{k['worst_from']:.2f} → {k['worst_to']:.2f}")
        chart(charts.scenario_chart(r["plot"], mode()), "sn_chart", f"{a.target}_{a.ref}_scenario_path.csv")
        st.markdown("**Anchor sensitivity**: target prices when aligned to other candidate dates. Treat the result as a range.")
        show_table(r["sensitivity"], f"{a.target}_{a.ref}_scenario_sensitivity.csv", ["corr", "amp_ratio", "target"])


# ---------- layout ----------
with st.sidebar:
    st.title("📈 Stock Analog Fit")
    page = st.radio("Feature", list(PAGES), label_visibility="collapsed")
    st.caption(PAGES[page])
    st.divider()
    st.caption("Charts: drag to zoom, double-click to reset, hover for values, camera icon to save a PNG. "
               "Every chart has a Table tab with the same numbers.")
    st.caption("Light or dark: follows your system; change it under ⋮ → Settings.")
    st.caption("Data: Yahoo Finance daily bars via yfinance, or your own CSV uploads.")
    st.caption("Scenarios based on historical price patterns only. Not a forecast and not investment advice.")

st.header(page)
{"📈 Trend fit": page_trend, "🔁 Analog (one reference)": page_analog,
 "🔎 Analog screen (many stocks)": page_screen, "🎯 Scenario projection": page_scenario}[page]()
