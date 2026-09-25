"""共用：取数（yfinance 或本地 CSV）、中文字体、对数轴刻度。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]


def load(ticker, period=None, start=None, end=None, csv=None):
    """返回按日期升序的 OHLC(V) DataFrame（未复权收盘，便于和历史最高价等原始报价对照）。

    csv: 本地文件路径，需含 Date,Open,High,Low,Close 列（Yahoo / 券商导出格式均可）。
    """
    if csv:
        df = pd.read_csv(csv)
        df.columns = [c.strip().title() for c in df.columns]
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df["Close"] = df["Adj Close"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        for c in ("Open", "High", "Low", "Close"):
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[$,]", "", regex=True))
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
    else:
        import yfinance as yf
        kw = dict(start=start, end=end) if (start or end) else dict(period=period or "2y")
        df = yf.download(ticker, interval="1d", auto_adjust=False, progress=False, **kw)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    df = df[[c for c in COLS if c in df.columns]].dropna(subset=["Close"])
    if df.empty:
        raise SystemExit(f"{ticker}: 没取到数据（网络被拦截？代码有误？可改用 --csv）")
    return df


def setup_fonts():
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for font in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC",
                 "Microsoft YaHei", "SimHei", "Arial Unicode MS"):
        if font in names:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False


def log_ticks(lo, hi, n=9):
    base = np.array([1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9])
    c = np.concatenate([base * 10.0 ** e for e in range(-3, 5)])
    c = c[(c >= lo) & (c <= hi)]
    return c[np.linspace(0, len(c) - 1, min(n, len(c))).round().astype(int)] if len(c) else c


def fmt_log_axis(ax, n=9, pct=False):
    """对数纵轴：整齐刻度 + 普通数字（pct=True 时显示为相对 1 的百分比）。"""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_yticks(log_ticks(lo, hi, n))
    f = (lambda v, _: f"{v - 1:+.0%}") if pct else (lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def resample(y, n):
    return np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)


def out_path(outdir, name):
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, name)
