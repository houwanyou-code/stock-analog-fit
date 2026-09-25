# stock-analog-fit

A Claude skill for stock price history. It does four things:

| Script | What it does |
|---|---|
| `scripts/kline_fit.py` | Daily candlestick chart with trend fitting: log-price regression channel (±2σ), recent-window regression, cubic polynomial, moving averages, support/resistance lines, pattern and breakout detection |
| `scripts/analog.py` | Finds the windows in a reference ticker's history that look most like the target's recent path (shape correlation + amplitude filter + 0.5–3× time scaling), overlays them on one chart, and projects forward paths |
| `scripts/screen.py` | Screens a basket of tickers (default ~65 volatile growth, crypto, fintech, EV, AI/quantum, IPO and meme names) for the windows most similar to the target, overlays the top matches, and summarizes the spread of what happened next |
| `scripts/scenario.py` | Maps a chosen historical leg of a reference ticker (e.g. NIO Jun 2020 → Jan 2021 high) onto the target: target price, timing, drawdowns along the way, and anchor sensitivity |

Usage instructions for Claude are in [`SKILL.md`](SKILL.md) (in Chinese).

## Install

**claude.ai**: download this repo as a ZIP (Code → Download ZIP), then upload it under Settings → Capabilities → Skills → Upload skill.

**Claude Code (all projects)**:

```bash
git clone https://github.com/houwanyou-code/stock-analog-fit ~/.claude/skills/stock-analog-fit
```

**Claude Code (one project)**: clone into `<project>/.claude/skills/stock-analog-fit`.

Python dependencies:

```bash
python3 -m pip install -r requirements.txt   # or inside a venv on macOS
```

## Web UI

All four features are available in a browser UI (`app.py`, built with Streamlit):

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

Pick a feature in the sidebar, fill in tickers and parameters (less common options are under "Advanced settings"), and click the button. The UI is in English. Charts are interactive (Plotly): drag to zoom, double-click to reset, hover for values (analog charts also show the reference stock's real date and price at each point), and use the camera icon to save a PNG. Tables can be downloaded as CSV, and every page accepts CSV uploads instead of live Yahoo data.

**Design.** Colors follow a validated data-viz palette (checked for color-blind separation and contrast in both light and dark mode): the target stock is always the blue accent, reference stocks take the next palette slots in a fixed order, and context (whole-period trend, moving averages, individual paths) is gray. Rising candles are hollow and falling candles filled, so direction never relies on color alone. The app follows the viewer's light/dark setting, and every chart has a **Table** tab with the same numbers.

**Deploy for free (a link you can open on your phone):** on [share.streamlit.io](https://share.streamlit.io), create an app from this repo with main file `app.py`.

## Run the scripts directly

```bash
cd scripts
python3 kline_fit.py NIO SECZ --period 2y
python3 analog.py SECZ NIO                     # SECZ's last 60 days vs NIO's full history
python3 screen.py SECZ                         # most similar stocks across a default basket
python3 scenario.py SECZ NIO --path-start 2020-06-01 --path-end 2021-01-11          # bull case
python3 scenario.py SECZ NIO --path-start 2021-01-11 --path-end 2022-05-15 --direction down
```

Data comes from Yahoo Finance via `yfinance`. If Yahoo is unreachable, pass exported CSVs with `--csv`, `--target-csv` or `--ref-csv` (columns `Date,Open,High,Low,Close`).

Charts and projections are based only on historical price patterns. They are scenarios, not forecasts or investment advice.
