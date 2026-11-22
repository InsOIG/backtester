Disclaimer : Research purpose only, Backtest results are shit, don't use this in real env

# EMA / VWAP Backtesters (Alpaca)

Two Python backtesters for EMA-VWAP crossover strategies on US stocks, using Alpaca market data. Both trade **long ITM options by default** (calls on long signals, puts on short signals) and can also run on plain shares.

| File | What it is |
|---|---|
| `ema_vwap_backtest.py` | Original strategy: EMA crosses VWAP, with volume, trend and VWAP-distance filters. ATR stop, fixed R-multiple target, flat by end of day. Includes a train/test optimizer. |
| `ema_vwap_pine_port.py` | Python port of a Pine v5 "EMA + VWAP Strategy": EMA 9/21, VWAP-slope filter, 4 stop types, reversals, positions can be held overnight. Includes a train/test optimizer. |

## Setup

```bash
pip install -r requirements.txt        # alpaca-py, pandas, numpy, tqdm
export APCA_API_KEY_ID="your_key_id"
export APCA_API_SECRET_KEY="your_secret"
```

Free paper-account keys from alpaca.markets are enough. The default `--feed iex` is the free data feed. Use `--demo` to run on synthetic data without keys.

## Quick start

```bash
# Original strategy
python ema_vwap_backtest.py --symbol SPY --days 365                   # ITM options (default)
python ema_vwap_backtest.py --symbol SPY --instrument shares          # plain shares
python ema_vwap_backtest.py --symbol QQQ --optimize                   # train/test grid search

# Pine port
python ema_vwap_pine_port.py --symbol SPY --days 365
python ema_vwap_pine_port.py --symbol SPY --sl-type Swing --rr 2.5
python ema_vwap_pine_port.py --symbol SPY --instrument shares         # Pine-style sizing

# No API keys needed
python ema_vwap_backtest.py --demo
python ema_vwap_pine_port.py --demo
```

## Common options

| Option | Meaning |
|---|---|
| `--symbol`, `--timeframe`, `--days` | Underlying, bar size in minutes, history in calendar days |
| `--instrument options\|shares` | Trade ITM options (default) or shares |
| `--opt-source alpaca\|model` | Real Alpaca option bars, or Black-Scholes (`--iv`). Demo mode always uses the model |
| `--itm-pct`, `--min-dte`, `--strike-step` | How far ITM, target days to expiry, strike spacing |
| `--capital` | Start capital (25,000 original, 10,000 Pine port) |
| `--risk-pct`, `--max-alloc-pct` | Options sizing: % of equity risked per trade, max % spent on premium |
| `--sizing risk\|premium`, `--min-one` | Sizing mode, and take 1 contract when the budget rounds to 0 |
| `--slippage`, `--commission` | Share costs; `--opt-slip` and `--opt-commission` for options |
| `--optimize`, `--train-frac` | Tune on the first part of the data, validate on the rest |
| `--save-trades file.csv`, `--quiet` | Export the trade log, turn off progress bars and logging |

Strategy-specific options: `--ema`, `--rr`, `--stop-atr`, `--min-vol-ratio`, `--no-trend` (original); `--ema-fast`, `--ema-slow`, `--sl-type ATR|Swing|SlowEMA|Fixed`, `--vwap-flat`, `--extended-hours` (Pine port). Run either script with `--help` for the full list.

## Output

Progress bars and API logs, sample data tables, win rate, average win/loss, expectancy (in R), profit factor, start and end capital, max drawdown, and a sizing-diagnostics block that explains skipped trades.

## Things to know

- **Sizing:** options trade in whole contracts. If the risk budget or premium cap is too small for one contract, the trade is skipped. The diagnostics block suggests fixes (`--risk-pct`, `--min-one`, `--sizing premium`, more capital).
- **Option data:** Alpaca historical option bars start around February 2024 and exist only when a contract traded. Unpriced trades are skipped and counted.
- **Model mode:** Black-Scholes uses a constant implied volatility, so it ignores IV changes and real spreads.
- **Pine port vs TradingView:** results can differ because of data feed and session differences. Compare the first few trades to check. Trades whose stop lands on the wrong side of the entry are skipped.
- **Overfitting:** only trust an optimized setup if the TEST results stay close to TRAIN. Aim for 100+ trades before drawing conclusions.

## Disclaimer

For research and education only. Not financial advice or a recommendation to trade. Backtests do not guarantee future results, and options can lose most or all of the premium.
