"""历史相似区间匹配：在参照股全部历史中找与目标股近期走势最像的区间，叠图并给出后续情景推演。

用法:
  python analog.py SECZ NIO                       # 目标 SECZ 最近 60 日（不足则全部）vs NIO 全历史
  python analog.py TSLA TSLA --window 40          # 自身历史类比（自动排除与当前窗口重叠的区间）
  python analog.py SECZ NIO --target-csv secz.csv --ref-csv nio.csv
主要参数:
  --window 目标比较窗口天数(默认60)   --horizon 推演天数(默认40)   --top 取前N段(默认5)
  --scales 时间伸缩倍数(默认 0.5,0.75,1,1.5,2,3；只要同速就填 1)
  --amp-min/--amp-max 振幅比过滤(默认 0.6~1.6，防止小波动区间凭形状入选)
输出: <outdir>/<T>_vs_<R>_analog.png、<T>_vs_<R>_analog.csv + 终端表格
"""
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from common import fmt_log_axis, load, log_ticks, out_path, resample, setup_fonts


def match(t_log, r_log, horizon, top, scales, amp_lo, amp_hi, exclude_from=None):
    """形状 = 标准化对数价格的相关系数；振幅比在 [amp_lo, amp_hi] 内才算幅度可比。
    允许时间伸缩：参照股用 m 倍天数走完同一形态，重采样到目标长度再比较。
    返回 [(corr, amp, start, W, m)]，互不重叠。exclude_from: 参照序列中该下标之后的窗口不参与（自身类比用）。
    """
    L = len(t_log)
    tz = (t_log - t_log.mean()) / t_log.std()
    trng = t_log.max() - t_log.min()
    limit = len(r_log) if exclude_from is None else exclude_from
    cands = []
    for m in scales:
        W, F = int(round(L * m)), int(round(horizon * m))
        for s in range(0, len(r_log) - W - F + 1):
            if s + W + F > limit:
                break
            w = r_log[s:s + W]
            amp = (w.max() - w.min()) / trng
            if not amp_lo <= amp <= amp_hi:
                continue
            if W != L:
                w = resample(w, L)
            cands.append((float(np.mean(tz * (w - w.mean()) / w.std())), amp, s, W, m))
    cands.sort(reverse=True)
    picked = []
    for c in cands:
        s, W = c[2], c[3]
        if all(s + W <= p[2] or p[2] + p[3] <= s for p in picked):
            picked.append(c)
        if len(picked) == top:
            break
    return picked


def path(rv, s, W, m, L, H):
    """参照股区间 + 其后走势，按时间伸缩重采样到目标的 L + H 个交易日（对数空间插值）。"""
    F = int(round(H * m))
    return np.exp(np.concatenate([resample(np.log(rv[s:s + W]), L),
                                  resample(np.log(rv[s + W - 1:s + W + F]), H + 1)[1:]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("ref")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--scales", default="0.5,0.75,1,1.5,2,3")
    ap.add_argument("--amp-min", type=float, default=0.6)
    ap.add_argument("--amp-max", type=float, default=1.6)
    ap.add_argument("--target-csv")
    ap.add_argument("--ref-csv")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    res = run(a)
    print("\n".join(res["lines"]))
    csv_out = out_path(a.outdir, f"{a.target}_vs_{a.ref}_analog.csv")
    res["table"].to_csv(csv_out, index=False)
    out = out_path(a.outdir, f"{a.target}_vs_{a.ref}_analog.png")
    res["fig"].savefig(out, dpi=130)
    print(f"\n图: {out}\n表: {csv_out}")


def run(a):
    """a: 带 target/ref/window/horizon/top/scales/amp_min/amp_max/target_csv/ref_csv 属性的对象。
    返回 {"lines": 摘要文字行, "table": 相似区间 DataFrame, "fig": matplotlib Figure, "summary": 各期限中位数}"""
    setup_fonts()
    lines = []

    def say(*x):
        lines.extend(" ".join(str(v) for v in x).split("\n"))

    tgt_all = load(a.target, "2y", csv=a.target_csv)["Close"]
    tgt = tgt_all.iloc[-a.window:]
    ref = load(a.ref, "max", csv=a.ref_csv)["Close"]
    L, H = len(tgt), a.horizon
    rv = ref.to_numpy()
    exclude = None
    if a.target.upper() == a.ref.upper():  # 自身类比：不能用到当前窗口及之后
        exclude = int(np.searchsorted(ref.index, tgt.index[0]))
    scales = [float(x) for x in a.scales.split(",")]
    picks = match(np.log(tgt.to_numpy()), np.log(rv), H, a.top, scales, a.amp_min, a.amp_max, exclude)
    if not picks:
        raise SystemExit("没有找到满足振幅条件的相似区间：可放宽 --amp-min/--amp-max 或增加 --scales")
    last = tgt.iloc[-1]

    say(f"{a.target}: {tgt.index[0].date()} → {tgt.index[-1].date()} ({L} 日)  收盘 {last:.2f}")
    say(f"{a.ref} 可用历史: {ref.index[0].date()} → {ref.index[-1].date()}\n")
    rows, paths = [], []
    for i, (corr, amp, s, W, m) in enumerate(picks, 1):
        p = path(rv, s, W, m, L, H) / rv[s + W - 1]
        paths.append(p)
        fwd = {h: p[L - 1 + h] - 1 for h in (5, 10, 20, H)}
        rows.append(dict(rank=i, start=ref.index[s].date(), end=ref.index[s + W - 1].date(),
                         days=W, speed=f"{m:g}x", corr=corr, amp=amp, p0=rv[s], p1=rv[s + W - 1],
                         **{f"+{h}d": v for h, v in fwd.items()},
                         maxup=p[L - 1:].max() - 1, maxdd=p[L - 1:].min() - 1))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    fmt = {c: "{:+.1%}".format for c in df.columns if c.startswith(("+", "max"))}
    say(df.to_string(index=False, formatters={**fmt, "corr": "{:.2f}".format, "amp": "{:.2f}".format,
                                                "p0": "{:.2f}".format, "p1": "{:.2f}".format}))
    say(f"(days = {a.ref} 实际用的交易日数；speed = 时间伸缩倍数；amp = 振幅比；后续收益已换算到 {a.target} 的交易日)")
    for h in (5, 10, 20, H):
        v = df[f"+{h}d"]
        say(f"+{h}日: 中位 {v.median():+.1%}  上涨占比 {(v > 0).mean():.0%}  → {a.target} 参考价 {last * (1 + v.median()):.2f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [1.1, 1]})
    xt, xf = np.arange(-L + 1, 1), np.arange(0, H + 1)

    # 图1：目标 vs 最佳匹配（参照股按末端对齐缩放；右轴为参照股当时真实价格）
    corr, amp, s, W, m = picks[0]
    e = s + W - 1
    k = last / rv[e]
    p0 = paths[0] * rv[e]
    ax1.plot(xt, tgt.to_numpy(), color="#c2185b", lw=2.4,
             label=f"{a.target} 实际 ({tgt.index[0]:%Y-%m-%d} → {tgt.index[-1]:%Y-%m-%d})")
    ax1.plot(xt, p0[:L] * k, color="#3b82c4", lw=1.8,
             label=f"{a.ref} 最相似区间 ({ref.index[s]:%Y-%m-%d} → {ref.index[e]:%Y-%m-%d}, {W}日, {m:g}x)  相关 {corr:.2f}")
    ax1.plot(xf, p0[L - 1:] * k, color="#3b82c4", lw=1.8, ls="--", label=f"{a.ref} 其后 {H} 日 → {a.target} 推演")
    ax1.axvline(0, color="gray", lw=1, ls=":")
    fmt_log_axis(ax1)
    lo, hi = ax1.get_ylim()
    sec = ax1.secondary_yaxis("right", functions=(lambda y: y / k, lambda y: y * k))
    sec.set_yticks(log_ticks(lo / k, hi / k))
    sec.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    sec.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    sec.set_ylabel(f"{a.ref} 当时真实价格")
    ax1.set_ylabel(f"{a.target} 价格（{a.ref} 已按比例对齐）")
    ax1.set_xlabel(f"{a.target} 交易日（0 = 最新一日；{a.ref} 区间已按时间伸缩对齐）")
    ax1.set_title(f"{a.target} 与 {a.ref} 历史最相似区间叠图（末端对齐）")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.2)

    # 图2：前 N 段的后续推演扇形
    ax2.plot(xt, tgt.to_numpy() / last, color="#c2185b", lw=2.4, label=a.target)
    cols = plt.cm.Blues(np.linspace(0.9, 0.4, len(picks)))
    for (corr, amp, s, W, m), p, c in zip(picks, paths, cols):
        ax2.plot(np.arange(-L + 1, H + 1), p, color=c, lw=1,
                 label=f"{a.ref} {ref.index[s]:%Y-%m-%d}→{ref.index[s + W - 1]:%Y-%m-%d} ({m:g}x, r={corr:.2f})")
    P = np.array([p[L - 1:] for p in paths])
    ax2.fill_between(xf, P.min(0), P.max(0), color="#3b82c4", alpha=0.12)
    ax2.plot(xf, np.median(P, 0), color="black", lw=2, ls="--", label="后续中位路径")
    ax2.axhline(1, color="gray", lw=0.8)
    ax2.axvline(0, color="gray", lw=1, ls=":")
    fmt_log_axis(ax2, n=10, pct=True)
    ax2.set_ylabel(f"相对 {a.target} 今日收盘")
    ax2.set_xlabel("交易日（0 = 最新一日）")
    ax2.set_title(f"前 {len(picks)} 个相似区间的后续走势（情景推演，非预测）")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    summary = {h: dict(median=df[f"+{h}d"].median(), up=(df[f"+{h}d"] > 0).mean(),
                       price=last * (1 + df[f"+{h}d"].median())) for h in (5, 10, 20, H)}
    return {"lines": lines, "table": df, "fig": fig, "summary": summary, "last": last}


if __name__ == "__main__":
    main()
