"""单只股票 K 线趋势拟合：回归通道 + 近期回归 + 三次多项式 + 均线 + 支撑/阻力线 + 形态/突破判断。

用法:
  python kline_fit.py NIO SECZ [--period 2y] [--outdir out]
  python kline_fit.py NIO --csv nio.csv
输出: <outdir>/<TICKER>_fit.png + 终端趋势摘要
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from common import fmt_log_axis, load, out_path, setup_fonts


def linfit(x, y):
    k, b = np.polyfit(x, y, 1)
    fit = k * x + b
    ss_tot = np.sum((y - y.mean()) ** 2)
    return k, b, 1 - np.sum((y - fit) ** 2) / ss_tot if ss_tot else 0.0, np.std(y - fit)


def pivot_line(x, y, idx, n_last=4):
    idx = idx[-n_last:]
    if len(idx) < 2:
        return None
    k, b = np.polyfit(x[idx], y[idx], 1)
    return k, b, idx


def analyze(ticker, df):
    close = df["Close"].to_numpy(float)
    n = len(close)
    x = np.arange(n, dtype=float)
    logc = np.log(close)
    short_win = min(60, max(10, n // 2))  # 新股历史短，近期窗口按长度缩
    r = {"ticker": ticker, "n": n, "start": df.index[0].date(), "end": df.index[-1].date(),
         "last": close[-1], "short_win": short_win}

    k, b, r2, sd = linfit(x, logc)
    r["long"] = dict(k=k, b=b, r2=r2, sd=sd, ann=np.exp(k * 252) - 1, chg=close[-1] / close[0] - 1,
                     z=(logc[-1] - (k * x[-1] + b)) / sd)
    xs, ys = x[-short_win:], logc[-short_win:]
    k2, b2, r22, sd2 = linfit(xs, ys)
    r["short"] = dict(k=k2, b=b2, r2=r22, sd=sd2, ann=np.exp(k2 * 252) - 1,
                      chg=close[-1] / close[-short_win] - 1, z=(ys[-1] - (k2 * xs[-1] + b2)) / sd2)

    p3 = np.polyfit(x / n, logc, 3)
    r["poly"] = dict(coef=p3, slope_end=np.polyval(np.polyder(p3, 1), 1.0),
                     curv_end=np.polyval(np.polyder(p3, 2), 1.0))

    s = df["Close"]
    ma = {w: s.rolling(w).mean().to_numpy() for w in (5, 10, 20, 60, 120) if w <= n // 2}
    r["ma"] = {w: v[-1] for w, v in ma.items()}

    order = max(2, min(5, n // 30))
    hi_idx = argrelextrema(df["High"].to_numpy(), np.greater_equal, order=order)[0]
    lo_idx = argrelextrema(df["Low"].to_numpy(), np.less_equal, order=order)[0]
    r["res_line"] = pivot_line(x, df["High"].to_numpy(), hi_idx[hi_idx < n - order])
    r["sup_line"] = pivot_line(x, df["Low"].to_numpy(), lo_idx[lo_idx < n - order])
    return r, ma


def verdict(r):
    L, S, P = r["long"], r["short"], r["poly"]
    short = r["n"] < 126  # 不足半年：年化外推会失真，改报区间涨幅

    def rate(d):
        return f"区间涨幅 {d['chg']:+.1%}" if short else f"年化 {d['ann']:+.1%}"

    def dirn(a):
        return "上升" if a > 0.05 else ("下降" if a < -0.05 else "横盘")

    lines = [
        f"== {r['ticker']}  {r['start']} → {r['end']}  ({r['n']} 根K线)  最新收盘 {r['last']:.2f}",
        f"整体趋势(对数线性回归): {dirn(L['ann'])}  {rate(L)}  R²={L['r2']:.2f}  通道位置 z={L['z']:+.2f}σ",
        f"近 {r['short_win']} 日趋势: {dirn(S['ann'])}  {rate(S)}  R²={S['r2']:.2f}  z={S['z']:+.2f}σ",
        f"三次多项式: 末端" + ("向上" if P["slope_end"] > 0 else "向下")
        + ("且在加速" if P["slope_end"] * P["curv_end"] > 0 else "但在放缓/可能拐头"),
    ]
    if L["r2"] < 0.2:
        lines.append("注: 整体回归 R² 很低，说明整段没有单一方向（震荡或 V 形），以近期趋势为准")
    ma = r["ma"]
    ws = sorted(ma)[-3:]
    if len(ws) >= 2 and all(np.isfinite([ma[w] for w in ws])):
        v = [ma[w] for w in ws]
        order = "多头排列" if all(a > b for a, b in zip(v, v[1:])) else (
            "空头排列" if all(a < b for a, b in zip(v, v[1:])) else "均线缠绕")
        lines.append("均线 " + " ".join(f"MA{w}={ma[w]:.2f}" for w in ws) + f" → {order}")
    rl, sl = r["res_line"], r["sup_line"]
    if rl and sl:
        xe = r["n"] - 1
        res_now, sup_now = rl[0] * xe + rl[1], sl[0] * xe + sl[1]
        lines.append(f"阻力线当前≈{res_now:.2f}  支撑线当前≈{sup_now:.2f}")
        pat = ("收敛三角形（方向待突破）" if rl[0] < 0 < sl[0] else "上升通道" if rl[0] > 0 and sl[0] > 0
               else "下降通道" if rl[0] < 0 and sl[0] < 0 else "扩散形态（波动放大）")
        lines.append(f"形态: {pat}")
        if r["last"] > res_now:
            lines.append(f"信号: 收盘已站上阻力线（向上突破）")
        elif r["last"] < sup_now:
            lines.append(f"信号: 收盘已跌破支撑线（向下破位）")
    if L["z"] > 2:
        lines.append("提示: 价格在整体通道 +2σ 之外，短期偏热")
    elif L["z"] < -2:
        lines.append("提示: 价格在整体通道 -2σ 之外，超跌")
    return "\n".join(lines)


def plot(ticker, df, r, ma, out):
    n = len(df)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(14, 7))
    up = df["Close"] >= df["Open"]
    col = np.where(up, "#d6453d", "#2a9d6a")  # 红涨绿跌
    ax.vlines(x, df["Low"], df["High"], color=col, lw=0.7)
    ax.bar(x, (df["Close"] - df["Open"]).abs().clip(lower=1e-6), bottom=np.minimum(df["Open"], df["Close"]),
           color=col, width=0.7)
    for w, c in zip(sorted(ma)[-3:], ("#e9a23b", "#3b82c4", "#8e5cc4")):
        ax.plot(x, ma[w], color=c, lw=1, label=f"MA{w}")
    tag = (lambda d: f"涨幅{d['chg']:+.0%}") if n < 126 else (lambda d: f"年化{d['ann']:+.0%}")
    L = r["long"]
    mid = L["k"] * x + L["b"]
    ax.plot(x, np.exp(mid), "k--", lw=1.2, label=f"整体回归 {tag(L)} R²={L['r2']:.2f}")
    ax.fill_between(x, np.exp(mid - 2 * L["sd"]), np.exp(mid + 2 * L["sd"]), color="gray", alpha=0.1, label="±2σ 通道")
    ax.plot(x, np.exp(np.polyval(r["poly"]["coef"], x / n)), color="#555", lw=1, alpha=0.7, label="三次多项式")
    S = r["short"]
    xs = x[-r["short_win"]:]
    ax.plot(xs, np.exp(S["k"] * xs + S["b"]), color="#c2185b", lw=2, label=f"近{r['short_win']}日 {tag(S)}")
    for line, c, name in ((r["res_line"], "#d6453d", "阻力线"), (r["sup_line"], "#2a9d6a", "支撑线")):
        if line:
            k, b, idx = line
            xx = np.arange(idx[0], n)
            ax.plot(xx, k * xx + b, color=c, lw=1.5, ls=":", label=name)
            ax.scatter(idx, k * idx + b, color=c, s=18, zorder=5)
    ticks = np.linspace(0, n - 1, 8).astype(int)
    ax.set_xticks(ticks, [df.index[i].strftime("%Y-%m-%d") for i in ticks])
    fmt_log_axis(ax)
    ax.set_title(f"{ticker} 日K 趋势拟合" + ("（上市不足半年）" if n < 126 else ""))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--csv", help="本地 CSV（仅配合单个 ticker）")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    setup_fonts()
    for t in a.tickers:
        df = load(t, a.period, csv=a.csv)
        if len(df) < 20:
            print(f"{t}: 数据不足 ({len(df)} 行)", file=sys.stderr)
            continue
        r, ma = analyze(t, df)
        print(verdict(r), "\n")
        out = out_path(a.outdir, f"{t}_fit.png")
        plot(t, df, r, ma, out)
        print("图:", out, "\n")


if __name__ == "__main__":
    main()
