"""指定情景推演：用参照股某段历史行情（如 NIO 2020-06 → 2021-01-11 最高点）的路径外推目标股。

步骤:
 1. 在 [anchor-from, anchor-to] 内逐日找“锚点”，使参照股截至锚点的走势（允许时间伸缩 m 倍）
    与目标股最近 window 日形状最像（相关系数）且振幅相当 → 目标股今天 ≈ 参照股锚点日。
 2. 参照股 锚点→极值点 的路径按 价格比例 k = 目标今收/参照锚点收盘、时间比例 1/m 映射到目标股未来。

用法:
  python scenario.py SECZ NIO --path-start 2020-06-01 --path-end 2021-01-11          # 乐观(默认 up，取区间最高价)
  python scenario.py SECZ NIO --path-start 2021-01-11 --path-end 2022-05-12 --direction down   # 悲观(取最低价)
可选: --anchor-from/--anchor-to（默认 path-start 到路径前半段）  --window 60  --scales 0.5:3:0.25
      --amp-min 0.8 --amp-max 1.25  --target-csv/--ref-csv  --outdir
输出: <outdir>/<T>_<R>_scenario_<direction>.png + 终端摘要（锚点、目标价、时间、途中最大回撤/反弹、倍数节点）
"""
import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from common import fmt_log_axis, load, log_ticks, out_path, resample, setup_fonts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("ref")
    ap.add_argument("--path-start", required=True)
    ap.add_argument("--path-end", required=True, help="路径终点附近日期；在 [path-start, path-end] 内取最高/最低价作为终点")
    ap.add_argument("--direction", choices=["up", "down"], default="up")
    ap.add_argument("--anchor-from")
    ap.add_argument("--anchor-to")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--scales", default="0.5:3:0.25")
    ap.add_argument("--amp-min", type=float, default=0.8)
    ap.add_argument("--amp-max", type=float, default=1.25)
    ap.add_argument("--target-csv")
    ap.add_argument("--ref-csv")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    res = run(a)
    print("\n".join(res["lines"]))
    out = out_path(a.outdir, f"{a.target}_{a.ref}_scenario_{a.direction}.png")
    res["fig"].savefig(out, dpi=130)
    print("\n图:", out)


def run(a):
    """返回 {"lines", "fig", "key": 关键数字 dict, "sensitivity": 锚点敏感性 DataFrame}"""
    setup_fonts()
    lines = []

    def say(*x):
        lines.extend(" ".join(str(v) for v in x).split("\n"))
    up = a.direction == "up"

    tgt = load(a.target, "2y", csv=a.target_csv).iloc[-a.window:]
    ref = load(a.ref, "max", csv=a.ref_csv)
    ref = ref[ref.index <= pd.Timestamp(a.path_end) + pd.Timedelta(days=10)]
    ps, pe = pd.Timestamp(a.path_start), pd.Timestamp(a.path_end)
    span = ref.loc[ps:pe + pd.Timedelta(days=5)]
    ext_day = span["High"].idxmax() if up else span["Low"].idxmin()
    ext_px = span.loc[ext_day, "High" if up else "Low"]
    af = pd.Timestamp(a.anchor_from) if a.anchor_from else ps
    at = pd.Timestamp(a.anchor_to) if a.anchor_to else ps + (ext_day - ps) / 2

    t = np.log(tgt["Close"].to_numpy())
    L = len(t)
    tz = (t - t.mean()) / t.std()
    trng = t.max() - t.min()
    r = np.log(ref["Close"].to_numpy())
    idx = ref.index
    lo_s, hi_s, st_s = (float(x) for x in a.scales.split(":"))

    best = None
    for m in np.arange(lo_s, hi_s + 1e-9, st_s):
        W = int(round(L * m))
        for e in range(W - 1, len(r)):
            if not af <= idx[e] <= at:
                continue
            w = resample(r[e - W + 1:e + 1], L)
            amp = (w.max() - w.min()) / trng
            if not a.amp_min <= amp <= a.amp_max:
                continue
            corr = float(np.mean(tz * (w - w.mean()) / w.std()))
            if best is None or corr > best[0]:
                best = (corr, m, e, W, amp)
    if best is None:
        raise SystemExit("锚点区间内没有振幅可比的窗口：放宽 --amp-min/--amp-max 或调整 --anchor-from/--anchor-to")
    corr, m, e, W, amp = best
    anchor = idx[e]
    last = tgt["Close"].iloc[-1]
    k = last / ref["Close"].iloc[e]
    p_end = idx.get_loc(ext_day)
    if p_end <= e:
        raise SystemExit("锚点落在极值点之后，请调整 --anchor-to")
    today = tgt.index[-1]

    fut = pd.bdate_range(today, periods=int(np.ceil((p_end - e) / m)) + 5)
    all_days = tgt.index.append(fut[1:])
    all_num = mdates.date2num(all_days)

    def to_num(j):  # 参照股第 j 日 → 目标股时间轴（小数位置插值，避免锯齿）
        return np.interp((L - 1) + (np.asarray(j) - e) / m, np.arange(len(all_days)), all_num)

    def to_date(j):
        return pd.Timestamp(mdates.num2date(float(to_num(j))).date())

    seg = ref["Close"].iloc[e:p_end + 1]
    if up:
        run_ = seg.cummax(); dd = seg / run_ - 1; worst = dd.idxmin(); top = seg.loc[:worst].idxmax()
    else:
        run_ = seg.cummin(); dd = seg / run_ - 1; worst = dd.idxmax(); top = seg.loc[:worst].idxmin()
    target_px = ext_px * k

    say(f"情景: {a.ref} {ps.date()} → {ext_day.date()} {'最高' if up else '最低'} {ext_px:.2f}（{'乐观' if up else '悲观'}）")
    say(f"锚点: {a.ref} {anchor.date()} 收盘 {ref['Close'].iloc[e]:.2f}  ↔  {a.target} {today.date()} 收盘 {last:.2f}")
    say(f"拟合: 相关 {corr:.3f}  振幅比 {amp:.2f}  时间伸缩 {m:g}x（{a.ref} 用 {W} 日走完 {a.target} {L} 日的形态）  价格比例 k={k:.3f}")
    if corr < 0.6:
        say(f"⚠ 拟合度低（相关 {corr:.2f} < 0.6）：{a.target} 当前形态与这段 {a.ref} 行情并不像，"
              f"下面的推演只是“如果照搬这段路径”的假设，不要当作相似性结论")
    say(f"{a.ref} 锚点→终点: {p_end - e} 个交易日  {ext_px / ref['Close'].iloc[e] - 1:+.0%}")
    say(f"{a.target} 情景目标: {target_px:.2f}（{target_px / last:.1f} 倍）  时间 ≈ {to_date(p_end).date()}（约 {int(round((p_end - e) / m))} 个交易日后）")
    say(f"途中最大{'回撤' if up else '反弹'} {dd.loc[worst]:+.0%}: {a.ref} {top.date()}→{worst.date()}  ↔ "
          f"{a.target} 约 {to_date(idx.get_loc(top)).date()} {run_[worst] * k:.2f} → {to_date(idx.get_loc(worst)).date()} {seg[worst] * k:.2f}")
    for mult in ((2, 3, 4, 6, 8) if up else (0.8, 0.6, 0.5, 0.3)):
        hit = seg[seg * k >= last * mult] if up else seg[seg * k <= last * mult]
        if len(hit):
            say(f"  {a.target} 到 {last * mult:.2f}（{mult:g} 倍）≈ {to_date(idx.get_loc(hit.index[0])).date()}  对应 {a.ref} {hit.index[0].date()}")

    # 其它锚点的敏感性：同一时间段内相关度次优、但锚点不同的候选
    say("\n锚点敏感性（相关度前列、日期不同的锚点 → 目标价）:")
    cands = []
    for mm in np.arange(lo_s, hi_s + 1e-9, st_s):
        WW = int(round(L * mm))
        for ee in range(WW - 1, len(r)):
            if not af <= idx[ee] <= at:
                continue
            w = resample(r[ee - WW + 1:ee + 1], L)
            cands.append((float(np.mean(tz * (w - w.mean()) / w.std())), mm, idx[ee], (w.max() - w.min()) / trng,
                          ext_px / ref["Close"].iloc[ee] * last))
    cands.sort(key=lambda c: -c[0])
    seen, sens = [], []
    for c in cands:
        if all(abs((c[2] - s).days) > 10 for s in seen):
            seen.append(c[2])
            sens.append(dict(anchor=c[2].date(), corr=round(c[0], 2), amp_ratio=round(float(c[3]), 2),
                             speed=f"{c[1]:g}x", target=round(float(c[4]), 2)))
            say(f"  锚点 {c[2].date()}  相关 {c[0]:.2f}  振幅比 {c[3]:.2f}  时间 {c[1]:g}x  → 目标 {c[4]:.1f}")
        if len(seen) == 4:
            break

    sens = pd.DataFrame(sens)

    # ---------- 作图 ----------
    js = np.arange(e - W + 1, p_end + 1)
    proj = ref["Close"].to_numpy()[js] * k
    xs = to_num(js)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(tgt.index, tgt["Close"], color="#c2185b", lw=2.6,
            label=f"{a.target} 实际收盘 ({tgt.index[0]:%Y-%m-%d} → {today:%Y-%m-%d})")
    h = js <= e
    ax.plot(xs[h], proj[h], color="#3b82c4", lw=1.6, alpha=0.8,
            label=f"{a.ref} {idx[e - W + 1]:%Y-%m-%d} → {anchor:%Y-%m-%d}（拟合段，相关 {corr:.2f}）")
    f = js >= e
    ax.plot(xs[f], proj[f], color="#3b82c4", lw=2, ls="--",
            label=f"{a.ref} {anchor:%Y-%m-%d} → {ext_day:%Y-%m-%d} 路径 → {a.target} {'乐观' if up else '悲观'}推演")
    pd_num = to_num(p_end)
    ax.scatter([pd_num], [target_px], color="#e65100", s=60, zorder=5)
    ax.annotate(f"情景目标 ≈ {target_px:.4g}\n({a.ref} {'最高' if up else '最低'} {ext_px:.2f}, {ext_day:%Y-%m-%d})",
                (pd_num, target_px), xytext=(-170, -10 if up else 20), textcoords="offset points", color="#e65100",
                arrowprops=dict(arrowstyle="->", color="#e65100"))
    ax.annotate(f"途中{'回撤' if up else '反弹'} {dd.loc[worst]:+.0%}\n({a.ref} {top:%m/%d}→{worst:%m/%d})",
                (to_num(idx.get_loc(worst)), seg[worst] * k), xytext=(10, -45 if up else 35),
                textcoords="offset points", color="#555", arrowprops=dict(arrowstyle="->", color="#999"))
    ax.axvline(mdates.date2num(today), color="gray", ls=":", lw=1)
    fmt_log_axis(ax)
    lo, hi = ax.get_ylim()
    sec = ax.secondary_yaxis("right", functions=(lambda y: y / k, lambda y: y * k))
    sec.set_yticks(log_ticks(lo / k, hi / k))
    sec.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    sec.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    sec.set_ylabel(f"{a.ref} 当时真实价格")
    ax.set_ylabel(f"{a.target} 价格")
    ax.set_title(f"{'乐观' if up else '悲观'}情景：用 {a.ref} {ps:%Y-%m} → {ext_day:%Y-%m-%d} 路径拟合 {a.target}"
                 f"（锚点 {a.ref} {anchor:%Y-%m-%d} ↔ {a.target} {today:%Y-%m-%d}，时间 {m:g}x）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    key = dict(anchor=anchor.date(), anchor_px=float(ref["Close"].iloc[e]), corr=corr, amp=amp, speed=m,
               k=k, last=float(last), target=float(target_px), mult=float(target_px / last),
               target_date=to_date(p_end).date(), days=int(round((p_end - e) / m)), ext_day=ext_day.date(),
               ext_px=float(ext_px), worst=float(dd.loc[worst]), worst_from=float(run_[worst] * k),
               worst_to=float(seg[worst] * k))
    to_d = lambda x: [pd.Timestamp(v.date()) for v in mdates.num2date(np.atleast_1d(x))]  # noqa: E731
    plot = dict(target=a.target, ref=a.ref, up=up, tgt_dates=list(tgt.index), tgt=tgt["Close"].to_numpy(),
                dates=to_d(xs), proj=proj, is_hist=h, ref_dates=[x.date() for x in idx[js]],
                ref_px=ref["Close"].to_numpy()[js], today=today, target_date=to_d(pd_num)[0],
                target_px=float(target_px), ext_day=ext_day.date(), ext_px=float(ext_px),
                worst_date=to_d(to_num(idx.get_loc(worst)))[0], worst_px=float(seg[worst] * k),
                worst_pct=float(dd.loc[worst]), top_date=top.date(), bottom_date=worst.date(),
                fit_start=idx[e - W + 1].date(), anchor=anchor.date(), corr=corr, speed=m)
    return {"lines": lines, "fig": fig, "key": key, "sensitivity": sens, "plot": plot}


if __name__ == "__main__":
    main()
