"""多股票相似形态筛选：在一篮子候选股票的历史里找与目标股近期走势最像的股票/区间，
汇总这些区间之后的走势，给出目标股的情景分布。

用法:
  python screen.py SECZ                                   # 默认候选池（见 DEFAULT_UNIVERSE）
  python screen.py SECZ --universe COIN,HOOD,CRCL,NIO    # 自定义候选池
  python screen.py SECZ --since 2015-01-01 --per-ticker 2 --top 10
主要参数:
  --window 目标比较窗口(默认60)  --horizon 推演天数(默认40)
  --per-ticker 每只股票最多取几段(默认2，避免一只股票霸榜)  --top 汇总前N段(默认10)
  --min-corr 入选的最低相关系数(默认0.75)  --scales / --amp-min / --amp-max 同 analog.py
输出: <outdir>/<T>_screen.png（叠图+后续扇形）、<T>_screen.csv（全部股票的最佳匹配）+ 终端表格
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from analog import match, path
from common import fmt_log_axis, load, out_path, setup_fonts

# 候选池：近年大幅波动的成长/题材股、新上市/SPAC、加密与金融科技、电动车、AI/量子
DEFAULT_UNIVERSE = (
    "COIN,HOOD,MSTR,MARA,RIOT,CLSK,HUT,CIFR,IREN,WULF,BTBT,BKKT,GLXY,CRCL,BLSH,FIGR,GEMI,"
    "SOFI,AFRM,UPST,NU,PYPL,XYZ,"
    "TSLA,NIO,XPEV,LI,RIVN,LCID,"
    "NVDA,AMD,SMCI,PLTR,ARM,CRWV,ALAB,APP,"
    "IONQ,RGTI,QBTS,OKLO,SMR,NNE,ASTS,RKLB,LUNR,JOBY,ACHR,"
    "RDDT,CART,RBRK,TEM,DJT,GME,AMC,CVNA,DKNG,SNOW,ABNB,CPNG,U,ROKU,ZM,MRNA,BNTX"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--universe", default=DEFAULT_UNIVERSE)
    ap.add_argument("--since", default="2012-01-01")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--per-ticker", type=int, default=2)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-corr", type=float, default=0.75)
    ap.add_argument("--scales", default="0.5,0.75,1,1.5,2,3")
    ap.add_argument("--amp-min", type=float, default=0.6)
    ap.add_argument("--amp-max", type=float, default=1.6)
    ap.add_argument("--target-csv")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    res = run(a)
    print("\n".join(res["lines"]))
    csv_out = out_path(a.outdir, f"{a.target}_screen.csv")
    res["all_best"].to_csv(csv_out, index=False)
    out = out_path(a.outdir, f"{a.target}_screen.png")
    res["fig"].savefig(out, dpi=130)
    print(f"\n图: {out}\n表: {csv_out}")


def run(a):
    """返回 {"lines", "fig", "table": 汇总用的区间, "all_best": 各股票最佳匹配, "summary": 各期限统计}"""
    setup_fonts()
    lines = []

    def say(*x):
        lines.extend(" ".join(str(v) for v in x).split("\n"))

    tgt = load(a.target, "2y", csv=a.target_csv)["Close"].iloc[-a.window:]
    L, H = len(tgt), a.horizon
    t_log = np.log(tgt.to_numpy())
    last = tgt.iloc[-1]
    scales = [float(x) for x in a.scales.split(",")]
    tickers = [x.strip().upper() for x in a.universe.split(",") if x.strip() and x.strip().upper() != a.target.upper()]

    raw = yf.download(tickers, start=a.since, interval="1d", auto_adjust=False, progress=False, group_by="column")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    say(f"{a.target}: {tgt.index[0].date()} → {tgt.index[-1].date()} ({L} 日)  收盘 {last:.2f}")
    say(f"候选池 {len(tickers)} 只，取到数据 {int(closes.notna().any().sum())} 只，历史自 {a.since}\n")

    segs, best_rows = [], []
    for tk in tickers:
        if tk not in closes:
            continue
        s = closes[tk].dropna()
        if len(s) < L * 2 + H:
            continue
        rv = s.to_numpy()
        picks = match(t_log, np.log(rv), H, a.per_ticker, scales, a.amp_min, a.amp_max)
        for rank, (corr, amp, st, W, m) in enumerate(picks):
            p = path(rv, st, W, m, L, H) / rv[st + W - 1]
            row = dict(ticker=tk, start=s.index[st].date(), end=s.index[st + W - 1].date(), days=W,
                       speed=f"{m:g}x", corr=corr, amp=amp, p0=rv[st], p1=rv[st + W - 1],
                       **{f"+{h}d": p[L - 1 + h] - 1 for h in (5, 10, 20, H)},
                       maxup=p[L - 1:].max() - 1, maxdd=p[L - 1:].min() - 1)
            segs.append((row, p))
            if rank == 0:
                best_rows.append(row)

    if not segs:
        raise SystemExit("没有找到满足振幅条件的区间：放宽 --amp-min/--amp-max 或扩大候选池")
    allbest = pd.DataFrame(best_rows).sort_values("corr", ascending=False)

    segs.sort(key=lambda x: -x[0]["corr"])
    chosen = [x for x in segs if x[0]["corr"] >= a.min_corr][:a.top] or segs[:a.top]
    df = pd.DataFrame([r for r, _ in chosen])
    pd.set_option("display.width", 230)
    fmt = {c: "{:+.1%}".format for c in df.columns if c.startswith(("+", "max"))}
    fm = {**fmt, "corr": "{:.2f}".format, "amp": "{:.2f}".format, "p0": "{:.2f}".format, "p1": "{:.2f}".format}
    say(f"== 各股票最佳匹配（前 15，共 {len(allbest)} 只有合格区间）")
    say(allbest.head(15)[["ticker", "start", "end", "speed", "corr", "amp"]].to_string(
        index=False, formatters={"corr": "{:.2f}".format, "amp": "{:.2f}".format}))
    say(f"\n== 汇总用的前 {len(df)} 段（相关 ≥ {a.min_corr}，每只股票最多 {a.per_ticker} 段）")
    say(df.to_string(index=False, formatters=fm))
    say(f"\n(后续收益已按时间伸缩换算到 {a.target} 的交易日)")
    summary = {}
    for h in (5, 10, 20, H):
        v = df[f"+{h}d"]
        summary[h] = dict(median=v.median(), q1=v.quantile(.25), q3=v.quantile(.75), up=(v > 0).mean(),
                          price=last * (1 + v.median()))
        say(f"+{h}日: 中位 {v.median():+.1%}  四分位 [{v.quantile(.25):+.1%}, {v.quantile(.75):+.1%}]  "
              f"上涨占比 {(v > 0).mean():.0%}  → {a.target} 中位参考价 {last * (1 + v.median()):.2f}")

    # ---------- 作图 ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1, 1]})
    xt, xf, xa = np.arange(-L + 1, 1), np.arange(0, H + 1), np.arange(-L + 1, H + 1)
    top_n = min(5, len(chosen))
    cols = plt.cm.tab10(np.arange(top_n))
    ax1.plot(xt, tgt.to_numpy() / last, color="#c2185b", lw=3, label=f"{a.target} 实际", zorder=5)
    for (r, p), c in zip(chosen[:top_n], cols):
        ax1.plot(xt, p[:L], color=c, lw=1.4, label=f"{r['ticker']} {r['start']}→{r['end']} ({r['speed']}, r={r['corr']:.2f})")
        ax1.plot(xf, p[L - 1:], color=c, lw=1.4, ls="--")
    ax1.axvline(0, color="gray", ls=":")
    ax1.axhline(1, color="gray", lw=0.8)
    fmt_log_axis(ax1, n=10, pct=True)
    ax1.set_title(f"与 {a.target} 走势最相似的 {top_n} 段（不同股票，末端对齐；虚线为其后走势）")
    ax1.set_ylabel(f"相对 {a.target} 今日收盘")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.2)

    P = np.array([p[L - 1:] for _, p in chosen])
    ax2.plot(xt, tgt.to_numpy() / last, color="#c2185b", lw=3, label=a.target)
    for _, p in chosen:
        ax2.plot(xa, p, color="#3b82c4", lw=0.8, alpha=0.5)
    ax2.fill_between(xf, np.quantile(P, .25, 0), np.quantile(P, .75, 0), color="#3b82c4", alpha=0.25, label="25%–75% 区间")
    ax2.fill_between(xf, P.min(0), P.max(0), color="#3b82c4", alpha=0.08, label="最小–最大")
    ax2.plot(xf, np.median(P, 0), color="black", lw=2.2, ls="--", label="中位路径")
    ax2.axvline(0, color="gray", ls=":")
    ax2.axhline(1, color="gray", lw=0.8)
    fmt_log_axis(ax2, n=10, pct=True)
    ax2.set_title(f"前 {len(chosen)} 段相似区间的后续走势分布（情景推演，非预测）")
    ax2.set_ylabel(f"相对 {a.target} 今日收盘")
    ax2.set_xlabel("交易日（0 = 最新一日；参照区间已按时间伸缩对齐）")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    return {"lines": lines, "fig": fig, "table": df, "all_best": allbest, "summary": summary, "last": last}


if __name__ == "__main__":
    main()
