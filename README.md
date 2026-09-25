# stock-analog-fit

A Claude skill for stock price history. It does three things:

| Script | What it does |
|---|---|
| `scripts/kline_fit.py` | Daily candlestick chart with trend fitting: log-price regression channel (±2σ), recent-window regression, cubic polynomial, moving averages, support/resistance lines, pattern and breakout detection |
| `scripts/analog.py` | Finds the windows in a reference ticker's history that look most like the target's recent path (shape correlation + amplitude filter + 0.5–3× time scaling), overlays them on one chart, and projects forward paths |
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

## Run the scripts directly

```bash
cd scripts
python3 kline_fit.py NIO SECZ --period 2y
python3 analog.py SECZ NIO                     # SECZ's last 60 days vs NIO's full history
python3 scenario.py SECZ NIO --path-start 2020-06-01 --path-end 2021-01-11          # bull case
python3 scenario.py SECZ NIO --path-start 2021-01-11 --path-end 2022-05-15 --direction down
```

Data comes from Yahoo Finance via `yfinance`. If Yahoo is unreachable, pass exported CSVs with `--csv`, `--target-csv` or `--ref-csv` (columns `Date,Open,High,Low,Close`).

Charts and projections are based only on historical price patterns. They are scenarios, not forecasts or investment advice.
