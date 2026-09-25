"""stock-analog-fit 网页界面。

启动:  streamlit run app.py
"""
import io
import os
import sys
import tempfile
from datetime import date
from types import SimpleNamespace

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import analog  # noqa: E402
import kline_fit  # noqa: E402
import scenario  # noqa: E402
import screen  # noqa: E402

st.set_page_config(page_title="K线拟合与相似形态", page_icon="📈", layout="wide")

PAGES = {
    "📈 趋势拟合": "单只或多只股票的 K 线趋势：回归通道、均线、支撑/阻力线、形态和突破判断",
    "🔁 相似区间": "在一只参照股的历史里找和目标股近期最像的区间，叠图并推演后续走势",
    "🔎 多股筛选": "在一批股票里找和目标股走势最像的股票/区间，汇总它们之后的走势分布",
    "🎯 情景推演": "按参照股某段指定行情（最乐观/最悲观）的路径推演目标股的目标价和时间",
}
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


# ---------- 通用小工具 ----------
def upload(label, key):
    """可选的 CSV 上传；返回临时文件路径或 None。"""
    f = st.file_uploader(label, type="csv", key=key,
                         help="Yahoo Finance → Historical Data → Download 导出的日线 CSV（Date,Open,High,Low,Close）")
    if f is None:
        return None
    path = os.path.join(tempfile.gettempdir(), f"saf_{key}.csv")
    with open(path, "wb") as fh:
        fh.write(f.getvalue())
    return path


def png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    return buf.getvalue()


def show_fig(fig, name):
    st.pyplot(fig, use_container_width=True)
    st.download_button("下载图片", png_bytes(fig), file_name=name, mime="image/png", key=f"dl_{name}")


def show_table(df, name, num_cols=()):
    fmt = {c: "{:+.1%}" for c in df.columns if str(c).startswith(("+", "max"))}
    fmt.update({c: "{:.2f}" for c in num_cols if c in df.columns})
    st.dataframe(df.style.format(fmt), use_container_width=True, hide_index=True)
    st.download_button("下载表格 CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=name,
                       mime="text/csv", key=f"dl_{name}")


def show_lines(lines):
    with st.expander("完整文字摘要"):
        st.code("\n".join(lines), language=None)


def run_safely(fn, *args, **kw):
    """执行分析；把脚本里的 SystemExit/网络错误转成页面上的提示。"""
    try:
        with st.spinner("正在取数和计算……（多股筛选首次约 15–30 秒）"):
            return fn(*args, **kw)
    except SystemExit as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if any(x in msg for x in ("403", "Connection", "curl", "resolve")):
            st.error("连不上 Yahoo Finance。请检查网络（需放行 query1/query2.finance.yahoo.com、fc.yahoo.com、"
                     "guce.yahoo.com），或改用上传 CSV。\n\n" + msg[:300])
        else:
            st.exception(e)
    return None


def kpi(col, label, value, note=None):
    """指标卡：说明文字用小字显示，避免 st.metric 的涨跌箭头造成误读。"""
    col.metric(label, value)
    if note:
        col.caption(note)


def horizon_metrics(summary, last, target):
    cols = st.columns(len(summary))
    for c, (h, v) in zip(cols, summary.items()):
        extra = f"  上涨占比 {v['up']:.0%}"
        c.metric(f"{h} 个交易日后 · 中位参考价", f"{v['price']:.2f}", f"{v['median']:+.1%}", help=extra)
    st.caption(f"以 {target} 最新收盘 {last:.2f} 为基准。" + "  ".join(
        f"{h}日上涨占比 {v['up']:.0%}" for h, v in summary.items()))


NUM = ["corr", "amp", "p0", "p1"]


# ---------- 各功能页面 ----------
def page_trend():
    c1, c2 = st.columns([3, 1])
    tickers = c1.text_input("股票代码（多个用逗号分隔）", "NIO, SECZ").upper()
    period = c2.selectbox("时间范围", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2)
    csv = upload("或上传一只股票的 CSV（可选，仅对第一个代码生效）", "trend_csv")
    if st.button("开始分析", type="primary", key="go_trend"):
        out = []
        for i, t in enumerate([x.strip() for x in tickers.split(",") if x.strip()]):
            res = run_safely(kline_fit.run, t, period, csv if i == 0 else None)
            if res:
                out.append((t, res))
        st.session_state["trend"] = out
    for t, res in st.session_state.get("trend", []):
        st.subheader(t)
        r = res["result"]
        c = st.columns(4)
        c[0].metric("最新收盘", f"{r['last']:.2f}")
        tag = "区间涨幅" if r["n"] < 126 else "年化"
        kpi(c[1], f"近 {r['short_win']} 日 {tag}", f"{(r['short']['chg'] if r['n'] < 126 else r['short']['ann']):+.1%}", f"R² {r['short']['r2']:.2f}")
        kpi(c[2], f"整体 {tag}", f"{(r['long']['chg'] if r['n'] < 126 else r['long']['ann']):+.1%}", f"R² {r['long']['r2']:.2f}")
        c[3].metric("整体通道位置", f"{r['long']['z']:+.2f}σ")
        st.markdown("\n".join(f"- {x}" for x in res["lines"][1:]))
        show_fig(res["fig"], f"{t}_fit.png")
        st.divider()


def analog_params(prefix, default_window=60):
    with st.expander("高级参数"):
        c1, c2, c3 = st.columns(3)
        window = c1.number_input("比较窗口（目标最近 N 个交易日）", 15, 500, default_window, key=f"{prefix}_w")
        horizon = c2.number_input("推演天数", 5, 120, 40, key=f"{prefix}_h")
        amp = c3.slider("振幅比范围", 0.2, 3.0, (0.6, 1.6), 0.05, key=f"{prefix}_amp",
                        help="参照区间的振幅 ÷ 目标的振幅。太宽会让小波动区间凭形状入选")
        scales = st.multiselect("允许的时间伸缩倍数", SCALES, [0.5, 0.75, 1.0, 1.5, 2.0, 3.0], key=f"{prefix}_sc",
                                help="参照股用几倍的天数走完同一形态；只要同速就只选 1")
    return window, horizon, amp, scales or [1.0]


def page_analog():
    c1, c2, c3 = st.columns([2, 2, 1])
    target = c1.text_input("目标股票", "SECZ").upper().strip()
    ref = c2.text_input("参照股票（全部历史）", "NIO").upper().strip()
    top = c3.number_input("取前 N 段", 1, 10, 5)
    window, horizon, amp, scales = analog_params("an")
    u1, u2 = st.columns(2)
    with u1:
        tcsv = upload("目标股 CSV（可选）", "an_t")
    with u2:
        rcsv = upload("参照股 CSV（可选）", "an_r")
    if st.button("开始匹配", type="primary", key="go_an"):
        a = SimpleNamespace(target=target, ref=ref, window=window, horizon=horizon, top=top,
                            scales=",".join(f"{x:g}" for x in scales), amp_min=amp[0], amp_max=amp[1],
                            target_csv=tcsv, ref_csv=rcsv)
        r = run_safely(analog.run, a)
        st.session_state["analog"] = (a, r) if r else None
    res = st.session_state.get("analog")
    if res:
        a, r = res
        horizon_metrics(r["summary"], r["last"], a.target)
        show_fig(r["fig"], f"{a.target}_vs_{a.ref}_analog.png")
        st.markdown("**相似区间明细**（后续涨跌已换算到目标股的交易日）")
        show_table(r["table"], f"{a.target}_vs_{a.ref}_analog.csv", NUM)
        show_lines(r["lines"])


def page_screen():
    c1, c2, c3 = st.columns([2, 1, 1])
    target = c1.text_input("目标股票", "SECZ", key="sc_t").upper().strip()
    since = c2.date_input("候选股历史起点", date(2012, 1, 1), key="sc_since")
    top = c3.number_input("汇总前 N 段", 3, 30, 10, key="sc_top")
    universe = st.text_area("候选股票池（逗号分隔，可改）", screen.DEFAULT_UNIVERSE.replace(",", ", "), height=110)
    c4, c5 = st.columns(2)
    per = c4.number_input("每只股票最多取几段", 1, 5, 2)
    min_corr = c5.slider("入选最低相关系数", 0.5, 0.99, 0.75, 0.01)
    window, horizon, amp, scales = analog_params("sc")
    tcsv = upload("目标股 CSV（可选）", "sc_csv")
    if st.button("开始筛选", type="primary", key="go_sc"):
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
        show_fig(r["fig"], f"{a.target}_screen.png")
        st.markdown("**汇总用的相似区间**")
        show_table(r["table"], f"{a.target}_screen_top.csv", NUM)
        st.markdown("**每只股票的最佳匹配（按相关系数排序）**")
        show_table(r["all_best"], f"{a.target}_screen_all.csv", NUM)
        st.info("注意：窗口太短（几乎单边的急涨）时相关系数普遍很高、区分度差；默认候选池有幸存者偏差，结果可能偏乐观。")
        show_lines(r["lines"])


def page_scenario():
    c1, c2, c3 = st.columns([2, 2, 2])
    target = c1.text_input("目标股票", "SECZ", key="sn_t").upper().strip()
    ref = c2.text_input("参照股票", "NIO", key="sn_r").upper().strip()
    direction = c3.radio("情景", ["up", "down"], horizontal=True,
                         format_func=lambda x: "乐观（到区间最高）" if x == "up" else "悲观（到区间最低）")
    d1, d2 = st.columns(2)
    ps = d1.date_input("参照行情起点", date(2020, 6, 1), min_value=date(1990, 1, 1))
    pe = d2.date_input("参照行情终点附近", date(2021, 1, 11), min_value=date(1990, 1, 1),
                       help="在起点到这一天之间自动取最高价（乐观）或最低价（悲观）作为终点")
    with st.expander("高级参数"):
        e1, e2, e3 = st.columns(3)
        af = e1.date_input("锚点搜索起点（可选）", None, key="sn_af")
        at = e2.date_input("锚点搜索终点（可选）", None, key="sn_at", help="默认为起点到极值点的前半段")
        window = e3.number_input("比较窗口（目标最近 N 日）", 15, 500, 60, key="sn_w")
        e4, e5 = st.columns(2)
        sc = e4.slider("时间伸缩倍数范围", 0.25, 4.0, (0.5, 3.0), 0.25, key="sn_sc")
        amp = e5.slider("振幅比范围", 0.2, 3.0, (0.8, 1.25), 0.05, key="sn_amp")
    u1, u2 = st.columns(2)
    with u1:
        tcsv = upload("目标股 CSV（可选）", "sn_tc")
    with u2:
        rcsv = upload("参照股 CSV（可选）", "sn_rc")
    if st.button("开始推演", type="primary", key="go_sn"):
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
            st.warning(f"拟合度低（相关 {k['corr']:.2f} < 0.6）：{a.target} 当前形态和这段 {a.ref} 行情并不像，"
                       "下面只是“如果照搬这段路径”的假设。")
        c = st.columns(4)
        kpi(c[0], "情景目标价", f"{k['target']:.2f}", f"{k['mult']:.1f} 倍")
        kpi(c[1], "预计到达", str(k["target_date"]), f"约 {k['days']} 个交易日后")
        kpi(c[2], "锚点", str(k["anchor"]), f"相关 {k['corr']:.2f} · 时间 {k['speed']:g}x")
        kpi(c[3], "途中最大" + ("回撤" if a.direction == "up" else "反弹"), f"{k['worst']:+.0%}", f"{k['worst_from']:.2f} → {k['worst_to']:.2f}")
        show_fig(r["fig"], f"{a.target}_{a.ref}_scenario_{a.direction}.png")
        st.markdown("**锚点敏感性**：对齐到不同日期时的目标价，应把它当作一个区间来看")
        show_table(r["sensitivity"], f"{a.target}_{a.ref}_scenario_sensitivity.csv")
        show_lines(r["lines"])


# ---------- 布局 ----------
with st.sidebar:
    st.title("📈 K线拟合与相似形态")
    page = st.radio("功能", list(PAGES), label_visibility="collapsed")
    st.caption(PAGES[page])
    st.divider()
    st.caption("数据：Yahoo Finance 日线（yfinance），也可上传 CSV。")
    st.caption("基于历史走势的技术面情景推演，不是预测，也不构成投资建议。")

st.header(page)
{"📈 趋势拟合": page_trend, "🔁 相似区间": page_analog,
 "🔎 多股筛选": page_screen, "🎯 情景推演": page_scenario}[page]()
