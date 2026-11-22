#!/usr/bin/env python3
"""
EMA / VWAP crossover backtester (US stocks, intraday) using Alpaca market data.

Setup:
    pip install alpaca-py pandas numpy
    export APCA_API_KEY_ID="your_key"
    export APCA_API_SECRET_KEY="your_secret"

Run:
    python ema_vwap_backtest.py --symbol SPY --timeframe 5 --days 365     # ITM options (default)
    python ema_vwap_backtest.py --symbol SPY --instrument shares          # plain shares
    python ema_vwap_backtest.py --symbol QQQ --optimize          # train/test grid search
    python ema_vwap_backtest.py --demo --optimize                # synthetic data, no API keys

Design choices aimed at improving expectancy (not guaranteed, always validate out of sample):
  * Daily-reset VWAP computed on regular trading hours only
  * Cross must happen with volume confirmation (volume / 20-bar average)
  * Trend filter: trade only in the direction of a slower EMA and its slope
  * Overextension filter: skip signals where price is already far from VWAP
  * Time window: skips the open and the late session
  * ATR-based stop and fixed R-multiple target (reward:risk set by --rr)
  * Entry on the NEXT bar's open (no look-ahead), slippage and commission included
  * If stop and target are both touched in the same bar, the stop is assumed hit first
  * Mandatory flat at end of day (no overnight risk)
"""

import argparse
import itertools
import os
import math
import time
from datetime import timedelta
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # minimal fallback so the script still runs without tqdm
    class tqdm:  # noqa: N801
        def __init__(self, iterable=None, total=None, desc="", disable=False, **kw):
            self.iterable, self.desc, self.disable, self.n = iterable, desc, disable, 0
            self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)

        def __iter__(self):
            for x in self.iterable:
                yield x
                self.update(1)
            self.close()

        def update(self, n=1):
            self.n += n
            if not self.disable:
                print(f"\r{self.desc}: {self.n}/{self.total}", end="", flush=True)

        def set_postfix(self, *a, **kw):
            pass

        def close(self):
            if not self.disable:
                print()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

        @staticmethod
        def write(msg):
            print(msg)

NY = "America/New_York"
VERBOSE = True


def log(msg=""):
    """Print without breaking active progress bars."""
    if VERBOSE:
        tqdm.write(str(msg))


def sleep_with_bar(seconds, desc="Waiting"):
    """Sleep while showing a countdown progress bar."""
    if seconds <= 0:
        return
    steps = max(1, int(seconds * 10))
    for _ in tqdm(range(steps), desc=f"{desc} ({seconds:.1f}s)", leave=False, disable=not VERBOSE,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}"):
        time.sleep(seconds / steps)


def show_sample(df, title, cols=None, n=5):
    if not VERBOSE or df is None or len(df) == 0:
        return
    d = df[cols] if cols else df
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        log(f"\n--- {title} ({len(df):,} rows; first {n} and last 3) ---")
        log(d.head(n).to_string())
        log("   ...")
        log(d.tail(3).to_string())
        log()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_alpaca_bars(symbol, minutes, days, feed="iex", chunk_days=30, pause=0.35, max_retries=5):
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")

    log(f"[API] Authenticating with key {key[:4]}...{key[-4:]} (secret hidden)")
    client = StockHistoricalDataClient(key, secret)

    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)  # clear of free-tier recency limit
    start = end - pd.Timedelta(days=days)
    windows, cur = [], start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end)
        windows.append((cur, nxt))
        cur = nxt
    log(f"[API] Symbol={symbol} timeframe={minutes}min feed={feed.upper()} "
        f"range={start:%Y-%m-%d} -> {end:%Y-%m-%d}")
    log(f"[API] Splitting into {len(windows)} request(s) of up to {chunk_days} days, "
        f"{pause:.2f}s pause between requests")

    frames, total_bars, t_all = [], 0, time.perf_counter()
    with tqdm(total=len(windows), desc=f"Downloading {symbol}", unit="req", disable=not VERBOSE) as bar:
        for k, (ws, we) in enumerate(windows, 1):
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                start=ws.to_pydatetime(), end=we.to_pydatetime(),
                feed=DataFeed.SIP if feed.lower() == "sip" else DataFeed.IEX,
                adjustment="split",
            )
            df = None
            for attempt in range(1, max_retries + 1):
                t_req = time.perf_counter()
                try:
                    log(f"[API] GET /v2/stocks/bars  request {k}/{len(windows)}  "
                        f"{ws:%Y-%m-%d} -> {we:%Y-%m-%d}  (attempt {attempt}/{max_retries})")
                    df = client.get_stock_bars(req).df
                    break
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    low = msg.lower()
                    if "subscription" in low or "forbidden" in low or "403" in low:
                        raise SystemExit(f"Access denied: {msg}\nTip: free accounts must use --feed iex.")
                    if "unauthorized" in low or "401" in low:
                        raise SystemExit(f"Authentication failed: {msg}\nCheck your API key and secret.")
                    wait = 60 if ("429" in msg or "rate" in low or "too many" in low) else min(2 ** attempt, 30)
                    log(f"[API] ! {type(exc).__name__}: {msg[:120]}")
                    log(f"[API] ! Retrying in {wait}s ({'rate limited' if wait == 60 else 'backoff'})")
                    sleep_with_bar(wait, "Retry wait")
            if df is None:
                raise SystemExit(f"Request {k} failed after {max_retries} attempts.")

            n = len(df)
            total_bars += n
            if n:
                frames.append(df.reset_index(level=0, drop=True))
            log(f"[API] <- {n:,} bars in {time.perf_counter() - t_req:.2f}s (running total {total_bars:,})")
            bar.set_postfix(bars=f"{total_bars:,}")
            bar.update(1)
            if k < len(windows):
                sleep_with_bar(pause, "Rate-limit pause")

    if not frames:
        raise SystemExit("No data returned. Check symbol, dates, feed, and API permissions.")
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    log(f"[API] Done: {len(out):,} bars from {len(windows)} request(s) in {time.perf_counter() - t_all:.1f}s")
    return out


def synthetic_bars(days=250, minutes=5, seed=7):
    """Random-walk bars for testing the code path without API access."""
    rng = np.random.default_rng(seed)
    rows, price = [], 400.0
    for d in pd.bdate_range("2024-01-02", periods=days, tz=NY):
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=30),
                            d + pd.Timedelta(hours=15, minutes=55), freq=f"{minutes}min")
        drift = rng.normal(0, 0.0004)  # slight daily trend component
        for t in idx:
            o = price
            c = o * (1 + drift / 10 + rng.normal(0, 0.0007))
            h = max(o, c) * (1 + abs(rng.normal(0, 0.0003)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.0003)))
            rows.append((t, o, h, l, c, rng.integers(5_000, 50_000)))
            price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.set_index("timestamp")


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def prepare(df, ema_len=9, trend_len=50, atr_len=14, vol_len=20):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert(NY) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(NY)
    df = df.sort_index()

    # Regular trading hours only (bar start 09:30 .. 15:55)
    mins = df.index.hour * 60 + df.index.minute
    df = df[(mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 55)].copy()

    df["date"] = df.index.date
    df["min"] = df.index.hour * 60 + df.index.minute

    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).groupby(df["date"]).cumsum() / df["volume"].groupby(df["date"]).cumsum()

    df["ema"] = df["close"].ewm(span=ema_len, adjust=False).mean()
    df["trend"] = df["close"].ewm(span=trend_len, adjust=False).mean()
    df["trend_slope"] = df["trend"] - df["trend"].shift(10)

    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=atr_len, adjust=False).mean()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(vol_len).mean()
    return df.dropna()


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    ema_len: int = 9
    rr: float = 2.0                 # target = rr * risk
    stop_atr: float = 1.5           # stop distance in ATRs
    min_vol_ratio: float = 1.2      # volume confirmation (0 disables)
    use_trend: bool = True          # trade only with slow EMA + slope
    max_vwap_dist_atr: float = 1.5  # skip if close is already this far from VWAP
    start_min: int = 9 * 60 + 45    # no entries before 09:45
    end_min: int = 15 * 60          # no entries after 15:00
    max_trades_day: int = 2
    allow_short: bool = True
    slip_per_share: float = 0.01    # $ per share, each side on market orders
    comm_per_share: float = 0.0


def backtest(df, p: Params, progress=False, desc="Backtesting"):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    ema, vwap = df["ema"].to_numpy(), df["vwap"].to_numpy()
    trend, slope = df["trend"].to_numpy(), df["trend_slope"].to_numpy()
    atr, volr, mins = df["atr"].to_numpy(), df["vol_ratio"].to_numpy(), df["min"].to_numpy()
    dates = df["date"].to_numpy()
    times = df.index

    day_starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(df)]
    trades = []

    day_iter = list(zip(day_starts[:-1], day_starts[1:]))
    if progress:
        day_iter = tqdm(day_iter, desc=desc, unit="day", disable=not VERBOSE)
    for s, e in day_iter:
        if progress and hasattr(day_iter, "set_postfix"):
            day_iter.set_postfix(trades=len(trades))
        n_today, i = 0, s + 1
        while i < e - 1 and n_today < p.max_trades_day:
            if not (p.start_min <= mins[i] <= p.end_min):
                i += 1
                continue

            cross_up = ema[i - 1] <= vwap[i - 1] and ema[i] > vwap[i]
            cross_dn = ema[i - 1] >= vwap[i - 1] and ema[i] < vwap[i]
            side = 0
            if cross_up:
                side = 1
            elif cross_dn and p.allow_short:
                side = -1
            if side == 0:
                i += 1
                continue

            # Filters
            if p.min_vol_ratio and volr[i] < p.min_vol_ratio:
                i += 1; continue
            if p.use_trend and not ((side == 1 and c[i] > trend[i] and slope[i] > 0) or
                                    (side == -1 and c[i] < trend[i] and slope[i] < 0)):
                i += 1; continue
            if abs(c[i] - vwap[i]) > p.max_vwap_dist_atr * atr[i]:
                i += 1; continue

            # Entry on next bar open
            ent_i = i + 1
            entry = o[ent_i] + side * p.slip_per_share
            risk = p.stop_atr * atr[i]
            if risk <= 0:
                i += 1; continue
            stop = entry - side * risk
            target = entry + side * p.rr * risk

            exit_px, exit_i, reason = None, e - 1, "eod"
            raw_exit = c[e - 1]
            for j in range(ent_i, e):
                hit_stop = l[j] <= stop if side == 1 else h[j] >= stop
                hit_tgt = h[j] >= target if side == 1 else l[j] <= target
                if hit_stop:                       # stop first if both touched
                    exit_px, exit_i, reason = stop - side * p.slip_per_share, j, "stop"
                    raw_exit = stop
                    break
                if hit_tgt:
                    exit_px, exit_i, reason = target, j, "target"
                    raw_exit = target
                    break
            if exit_px is None:
                exit_px = c[e - 1] - side * p.slip_per_share

            pnl = side * (exit_px - entry) - 2 * p.comm_per_share
            trades.append({
                "entry_time": times[ent_i], "exit_time": times[exit_i],
                "side": "long" if side == 1 else "short",
                "entry": entry, "exit": exit_px, "risk": risk,
                "u_entry": o[ent_i], "u_exit": raw_exit,
                "pnl_per_share": pnl, "R": pnl / risk, "reason": reason,
            })
            n_today += 1
            i = exit_i + 1

    return pd.DataFrame(trades)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(tr):
    if tr is None or tr.empty:
        return dict(trades=0, win_rate=np.nan, avg_win_R=np.nan, avg_loss_R=np.nan,
                    expectancy_R=np.nan, profit_factor=np.nan, total_R=0.0, max_dd_R=0.0)
    r = tr["R"].to_numpy()
    wins, losses = r[r > 0], r[r <= 0]
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(eq) - eq).max()
    gl = -losses.sum()
    return dict(
        trades=len(r),
        win_rate=len(wins) / len(r),
        avg_win_R=wins.mean() if len(wins) else 0.0,
        avg_loss_R=losses.mean() if len(losses) else 0.0,
        expectancy_R=r.mean(),                      # = win% * avg win - loss% * avg loss
        profit_factor=(wins.sum() / gl) if gl > 0 else np.inf,
        total_R=r.sum(),
        max_dd_R=dd,
    )


def print_metrics(title, m):
    print(f"\n=== {title} ===")
    print(f"Trades          : {m['trades']}")
    if m["trades"] == 0:
        return
    print(f"Win rate        : {m['win_rate']*100:.1f}%")
    print(f"Avg win / loss  : {m['avg_win_R']:.2f}R / {m['avg_loss_R']:.2f}R")
    print(f"Expectancy      : {m['expectancy_R']:+.3f}R per trade")
    print(f"Profit factor   : {m['profit_factor']:.2f}")
    print(f"Total / Max DD  : {m['total_R']:+.1f}R / {m['max_dd_R']:.1f}R")


# --------------------------------------------------------------------------- #
# Capital simulation (fixed-fractional position sizing)
# --------------------------------------------------------------------------- #
def apply_capital(tr, capital, risk_pct, max_leverage):
    """Size each trade so a stop-out loses ~risk_pct of current equity, capped by leverage."""
    if tr is None or tr.empty:
        return tr
    tr = tr.copy()
    equity, shares_l, pnl_l, eq_l = capital, [], [], []
    for row in tr.itertuples():
        by_risk = (equity * risk_pct / 100) / row.risk
        by_lev = (equity * max_leverage) / row.entry
        shares = int(max(0, min(by_risk, by_lev)))
        pnl = shares * row.pnl_per_share
        equity += pnl
        shares_l.append(shares); pnl_l.append(pnl); eq_l.append(equity)
    tr["shares"], tr["pnl_usd"], tr["equity"] = shares_l, pnl_l, eq_l
    return tr


def print_capital(tr, capital, sizing_note, size_col="shares", unit="shares"):
    if tr is None or tr.empty:
        print(f"\n=== Capital ===\nStart capital: ${capital:,.2f} | no trades, end capital: ${capital:,.2f}")
        return
    eq = np.r_[capital, tr["equity"].to_numpy()]
    peak = np.maximum.accumulate(eq)
    dd_usd, dd_pct = (peak - eq).max(), ((peak - eq) / peak).max() * 100
    end = eq[-1]
    span_days = max((tr["exit_time"].iloc[-1] - tr["entry_time"].iloc[0]).days, 1)
    print("\n=== Capital ===")
    print(f"Start capital   : ${capital:,.2f}")
    print(f"End capital     : ${end:,.2f}")
    print(f"Net profit      : ${end - capital:+,.2f} ({(end / capital - 1) * 100:+.2f}%)")
    print(f"Max drawdown    : ${dd_usd:,.2f} ({dd_pct:.2f}%)")
    print(f"Sizing          : {sizing_note}")
    print(f"Avg position    : {tr[size_col].mean():.1f} {unit}")
    print(f"Period          : {span_days} days between first entry and last exit")


# --------------------------------------------------------------------------- #
# ITM options layer
#   The underlying (SPY etc.) generates the signal, stop and target exactly as before.
#   Each signal is then expressed as a long ITM option:
#     long signal  -> buy an ITM CALL      short signal -> buy an ITM PUT
#   and the option's entry/exit prices come from Alpaca option bars or a Black-Scholes model.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptParams:
    source: str = "alpaca"        # "alpaca" (real option bars) or "model" (Black-Scholes)
    itm_pct: float = 1.0          # strike this % in the money vs the underlying
    min_dte: int = 7              # target days to expiry (rolled to next weekday)
    strike_step: float = 1.0      # strike spacing ($1 for SPY/QQQ/IWM; use 2.5 or 5 for many stocks)
    assumed_delta: float = 0.75   # used only for position sizing (estimated loss at the stop)
    max_alloc_pct: float = 25.0   # max premium outlay per trade, % of equity
    slip: float = 0.03            # $ per share (x100 per contract) each side, half-spread proxy
    comm: float = 0.65            # $ per contract each side (Alpaca is lower; this is conservative)
    iv: float = 0.18              # implied vol for model mode (constant, so no vega effects)
    rate: float = 0.04
    max_gap_bars: int = 2         # accept an option print up to this many bars after the signal
    pause: float = 0.35
    sizing: str = "risk"          # "risk" (risk budget) or "premium" (spend max_alloc_pct on premium)
    min_one: bool = False         # take 1 contract when the budget rounds down to 0
    max_risk_pct: float = 3.0     # ceiling on est. risk of a --min-one override, % of equity


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, r, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def next_weekday(d):
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def occ_symbol(root, expiry, is_call, strike):
    return f"{root}{expiry:%y%m%d}{'C' if is_call else 'P'}{int(round(strike * 1000)):08d}"


def itm_strike(spot, is_call, itm_pct, step):
    if is_call:
        return math.floor(spot * (1 - itm_pct / 100) / step) * step
    return math.ceil(spot * (1 + itm_pct / 100) / step) * step


def _years(expiry, ts, r=0.0):
    expiry_ts = pd.Timestamp(expiry).tz_localize(NY) + pd.Timedelta(hours=16)
    return max((expiry_ts - ts).total_seconds(), 0) / (365 * 86400)


def model_prices(symbol, t, o: OptParams, minutes):
    is_call = t["side"] == "long"
    K = itm_strike(t["u_entry"], is_call, o.itm_pct, o.strike_step)
    expiry = next_weekday(t["entry_time"].date() + timedelta(days=o.min_dte))
    p0 = bs_price(t["u_entry"], K, _years(expiry, t["entry_time"]), o.rate, o.iv, is_call)
    exit_ts = t["exit_time"] + pd.Timedelta(minutes=minutes if t["reason"] == "eod" else minutes / 2)
    p1 = bs_price(t["u_exit"], K, _years(expiry, exit_ts), o.rate, o.iv, is_call)
    return dict(sym=occ_symbol(symbol, expiry, is_call, K), expiry=expiry, strike=K, entry=p0, exit=p1)


def _option_client():
    from alpaca.data.historical import OptionHistoricalDataClient
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")
    log("[API] Connecting to Alpaca OPTIONS market data client")
    return OptionHistoricalDataClient(key, secret)


def fetch_option_day(client, symbols, day, minutes, max_retries=5):
    """One request: intraday bars for several candidate contracts on one day."""
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    start = pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)
    end = start + pd.Timedelta(hours=6, minutes=30)
    req = OptionBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                            start=start.to_pydatetime(), end=end.to_pydatetime())
    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            log(f"[API] GET /v1beta1/options/bars  {day}  {len(symbols)} candidate contract(s)  "
                f"(attempt {attempt}/{max_retries})")
            df = client.get_option_bars(req).df
            log(f"[API] <- {len(df):,} option bars in {time.perf_counter() - t0:.2f}s")
            return df
        except Exception as exc:  # noqa: BLE001
            msg, low = str(exc), str(exc).lower()
            if "401" in low or "unauthorized" in low or "403" in low or "forbidden" in low:
                raise SystemExit(f"Options data access denied: {msg}")
            if "400" in low or "invalid" in low or "no data" in low:
                log(f"[API] ! {msg[:100]} (treating as no data)")
                return None
            wait = 60 if ("429" in msg or "rate" in low or "too many" in low) else min(2 ** attempt, 30)
            log(f"[API] ! {type(exc).__name__}: {msg[:100]} -> retrying in {wait}s")
            sleep_with_bar(wait, "Retry wait")
    return None


def _price_from_bars(bars, t, o: OptParams, minutes):
    """Entry = open of first option bar at/after the signal; exit = bar close (EOD) or mid-bar (stop/target)."""
    idx = bars.index.searchsorted(t["entry_time"], side="left")
    if idx >= len(bars):
        return None
    if (bars.index[idx] - t["entry_time"]).total_seconds() > o.max_gap_bars * minutes * 60:
        return None
    pos = bars.index.searchsorted(t["exit_time"], side="right") - 1
    if pos < idx:
        return None
    bar = bars.iloc[pos]
    if t["reason"] != "eod" and bars.index[pos] == t["exit_time"]:
        exit_px = (bar["open"] + bar["close"]) / 2
    else:
        exit_px = bar["close"]
    return dict(entry=float(bars.iloc[idx]["open"]), exit=float(exit_px))


def alpaca_prices(client, symbol, t, o: OptParams, minutes):
    is_call = t["side"] == "long"
    K = itm_strike(t["u_entry"], is_call, o.itm_pct, o.strike_step)
    strikes = [K, K - o.strike_step if is_call else K + o.strike_step]  # target strike, then one step deeper ITM
    expiries, d = [], next_weekday(t["entry_time"].date() + timedelta(days=o.min_dte))
    while len(expiries) < 3:
        expiries.append(d)
        d = next_weekday(d + timedelta(days=1))
    cands = [(occ_symbol(symbol, ex, is_call, k), ex, k) for ex in expiries for k in strikes]
    df = fetch_option_day(client, [c[0] for c in cands], t["entry_time"].date(), minutes)
    if df is None or len(df) == 0:
        return None
    df = df.reset_index()
    for sym, ex, k in cands:  # earliest expiry first, target strike before deeper ITM
        part = df[df["symbol"] == sym]
        if part.empty:
            continue
        bars = part.set_index("timestamp").sort_index()
        bars.index = bars.index.tz_convert(NY)
        px = _price_from_bars(bars, t, o, minutes)
        if px:
            return dict(sym=sym, expiry=ex, strike=k, **px)
    return None


def apply_options(tr, symbol, capital, risk_pct, o: OptParams, minutes):
    """Convert underlying trades into long-ITM-option trades.

    Sizing modes (o.sizing):
      risk    : contracts = floor(equity x risk% / est. loss at stop), capped by the premium limit (default)
      premium : contracts = floor(equity x max_alloc% / premium); ignores the risk budget
    With min_one=True a trade that rounds to 0 contracts still takes 1 contract if one contract is affordable
    and its estimated loss at the stop is at most max_risk_pct of equity.
    """
    if tr is None or tr.empty:
        return tr
    client = _option_client() if o.source == "alpaca" else None
    equity, rows = capital, []
    cnt = dict(skip_no_data=0, skip_risk=0, skip_alloc=0, oversized=0)
    prem_list, risk_list = [], []
    bar = tqdm(tr.to_dict("records"), desc=f"Pricing options ({o.source})", unit="trade", disable=not VERBOSE)
    for t in bar:
        q = model_prices(symbol, t, o, minutes) if o.source == "model" else alpaca_prices(client, symbol, t, o, minutes)
        if o.source == "alpaca":
            sleep_with_bar(o.pause, "Rate-limit pause")
        stamp = f"{t['entry_time']:%Y-%m-%d %H:%M}"
        if q is None:
            cnt["skip_no_data"] += 1
            log(f"[OPT] skip {stamp} {t['side']}: no usable option prices")
            continue
        paid = q["entry"] + o.slip
        recv = max(q["exit"] - o.slip, 0.0)
        prem_pc = paid * 100                                   # cost of 1 contract
        risk_pc = o.assumed_delta * t["risk"] * 100            # est. loss of 1 contract if the stop is hit
        prem_list.append(prem_pc)
        risk_list.append(risk_pc)
        n_alloc = (equity * o.max_alloc_pct / 100) / prem_pc
        if o.sizing == "premium":
            n = int(max(0, n_alloc))
        else:
            n = int(max(0, min((equity * risk_pct / 100) / risk_pc, n_alloc)))
            if n < 1 and o.min_one and n_alloc >= 1 and risk_pc <= equity * o.max_risk_pct / 100:
                n = 1
                cnt["oversized"] += 1
        if n < 1:
            if n_alloc < 1:
                cnt["skip_alloc"] += 1
                log(f"[OPT] skip {stamp}: 1 contract costs ${prem_pc:,.0f}, above the {o.max_alloc_pct}% premium cap "
                    f"(${equity * o.max_alloc_pct / 100:,.0f})")
            else:
                cnt["skip_risk"] += 1
                log(f"[OPT] skip {stamp}: 1 contract risks ~${risk_pc:,.0f} at the stop, above the "
                    f"{risk_pct}% risk budget (${equity * risk_pct / 100:,.0f})")
            continue
        pnl = n * 100 * (recv - paid) - 2 * n * o.comm
        planned_risk = n * risk_pc
        equity += pnl
        extra = {"shares_pnl_usd": t["pnl_usd"]} if "pnl_usd" in t else {}
        rows.append({**t, **extra, "R_underlying": t["R"], "opt_symbol": q["sym"], "strike": q["strike"],
                     "expiry": q["expiry"], "past_expiry": q.get("past_expiry", False),
                     "opt_entry": paid, "opt_exit": recv, "contracts": n,
                     "premium_return_pct": (recv / paid - 1) * 100, "pnl_usd": pnl,
                     "equity": equity, "R": pnl / planned_risk})
        bar.set_postfix(priced=len(rows), skipped=sum(v for k, v in cnt.items() if k.startswith("skip")),
                        equity=f"${equity:,.0f}")
    out = pd.DataFrame(rows)
    out.attrs.update(cnt)
    out.attrs["signals"] = len(tr)
    out.attrs["skipped"] = cnt["skip_no_data"] + cnt["skip_risk"] + cnt["skip_alloc"]
    out.attrs["prem"], out.attrs["risk"] = prem_list, risk_list
    return out


def print_sizing_report(trades, capital, risk_pct, o: OptParams):
    a = trades.attrs
    nd, sr, sa, ov = (a.get(k, 0) for k in ("skip_no_data", "skip_risk", "skip_alloc", "oversized"))
    prem, rk = a.get("prem") or [], a.get("risk") or []
    if not (nd or sr or sa or ov or prem):
        return
    print("\n=== Sizing diagnostics ===")
    print(f"Skipped        : no option data {nd} | risk budget too small for 1 contract {sr} | "
          f"1 contract above premium cap {sa}")
    if ov:
        print(f"Overrides      : {ov} trades took 1 contract although it risks more than {risk_pct}% "
              f"(allowed up to {o.max_risk_pct}%)")
    if prem:
        mp, mr = float(np.median(prem)), float(np.median(rk))
        print(f"Typical 1 lot  : premium ${mp:,.0f} | est. loss at stop ${mr:,.0f} "
              f"({mr / capital * 100:.1f}% of ${capital:,.0f})")
        print(f"Capital needed : ~${mr / (risk_pct / 100):,.0f} for 1 contract at {risk_pct}% risk | "
              f"~${mp / (o.max_alloc_pct / 100):,.0f} for 1 contract under the {o.max_alloc_pct}% premium cap")
    if sr or sa:
        print("Ways to size up:")
        if sr and prem:
            print(f"  --risk-pct {min(10.0, max(risk_pct, np.ceil(mr / capital * 1000) / 10)):.1f}   "
                  f"raise the risk budget so 1 contract fits (more risk per trade)")
            print(f"  --min-one --max-risk-pct 3   take 1 contract when the budget rounds to 0, up to a risk ceiling")
        if sa:
            print("  --max-alloc-pct 40           allow a bigger share of equity in premium per trade")
        print("  --sizing premium             size by premium spent instead of risk (e.g. with --max-alloc-pct 20)")
        print("  --capital <larger>           or test with more starting capital")


# --------------------------------------------------------------------------- #
# Optimization with out-of-sample validation
# --------------------------------------------------------------------------- #
def split_by_days(df, train_frac):
    days = np.array(sorted(set(df["date"])))
    cut = days[int(len(days) * train_frac)]
    return df[df["date"] < cut], df[df["date"] >= cut]


def optimize(raw, base: Params, train_frac=0.6, min_trades=30, top=5):
    grid = dict(
        ema_len=[9, 20],
        rr=[1.5, 2.0, 3.0],
        stop_atr=[1.0, 1.5, 2.0],
        min_vol_ratio=[0.0, 1.2, 1.5],
        use_trend=[True, False],
        max_vwap_dist_atr=[1.0, 1.5, 2.5],
    )
    results = []
    log("[OPT] Preparing indicators for each EMA length...")
    prepared = {n: prepare(raw, ema_len=n) for n in grid["ema_len"]}
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    ndays = prepared[grid["ema_len"][0]]["date"].nunique()
    log(f"[OPT] {len(combos)} combinations | {ndays} trading days | train fraction {train_frac:.0%} "
        f"| min {min_trades} trades in-sample")
    best, t0 = float("-inf"), time.perf_counter()
    bar = tqdm(combos, desc="Optimizing", unit="combo", disable=not VERBOSE)
    for vals in bar:
        p = replace(base, **dict(zip(keys, vals)))
        train, _ = split_by_days(prepared[p.ema_len], train_frac)
        m = metrics(backtest(train, p))
        if m["trades"] >= min_trades:
            results.append((m["expectancy_R"], p, m))
            best = max(best, m["expectancy_R"])
        bar.set_postfix(valid=len(results), best_exp=f"{best:+.3f}R" if results else "n/a")
    log(f"[OPT] Grid finished in {time.perf_counter() - t0:.1f}s")

    if not results:
        print("No parameter set reached the minimum trade count on the training data.")
        return

    results.sort(key=lambda x: x[0], reverse=True)
    print(f"\nTested {len(combos)} combos; "
          f"{len(results)} met the {min_trades}-trade minimum in-sample.")
    print(f"\nTop {top} by TRAIN expectancy, with their unseen TEST results:")
    for rank, (_, p, mtr) in enumerate(results[:top], 1):
        _, test = split_by_days(prepared[p.ema_len], train_frac)
        mte = metrics(backtest(test, p, progress=True, desc=f"Validating #{rank} on test data"))
        print(f"\n#{rank} ema={p.ema_len} rr={p.rr} stop_atr={p.stop_atr} vol>={p.min_vol_ratio} "
              f"trend={p.use_trend} vwap_dist<={p.max_vwap_dist_atr}")
        print(f"   TRAIN: n={mtr['trades']:4d} win={mtr['win_rate']*100:5.1f}% "
              f"exp={mtr['expectancy_R']:+.3f}R PF={mtr['profit_factor']:.2f}")
        print(f"   TEST : n={mte['trades']:4d} win={mte['win_rate']*100 if mte['trades'] else float('nan'):5.1f}% "
              f"exp={mte['expectancy_R'] if mte['trades'] else float('nan'):+.3f}R "
              f"PF={mte['profit_factor'] if mte['trades'] else float('nan'):.2f}")
    print("\nOnly trust a setup if TEST expectancy stays positive and close to TRAIN. "
          "A big drop means the parameters were overfit.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="EMA/VWAP crossover backtest on Alpaca data")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--timeframe", type=int, default=5, help="bar size in minutes")
    ap.add_argument("--days", type=int, default=365, help="calendar days of history")
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"], help="iex is free; sip needs a paid plan")
    ap.add_argument("--ema", type=int, default=9)
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--min-vol-ratio", type=float, default=1.2)
    ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--slippage", type=float, default=0.01, help="$ per share per side")
    ap.add_argument("--commission", type=float, default=0.0, help="$ per share per side")
    ap.add_argument("--optimize", action="store_true", help="grid search on train split, report on test split")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--capital", type=float, default=25000.0, help="starting capital in USD")
    ap.add_argument("--risk-pct", type=float, default=1.0, help="%% of equity risked per trade")
    ap.add_argument("--max-leverage", type=float, default=1.0, help="max position value / equity (1 = cash)")
    ap.add_argument("--instrument", default="options", choices=["options", "shares"],
                    help="trade ITM options (default) or plain shares")
    ap.add_argument("--opt-source", default=None, choices=["alpaca", "model"],
                    help="option prices: real Alpaca bars, or Black-Scholes model (default: alpaca; model in --demo)")
    ap.add_argument("--itm-pct", type=float, default=1.0, help="how far in the money the strike is, %% of price")
    ap.add_argument("--min-dte", type=int, default=7, help="target days to expiry")
    ap.add_argument("--strike-step", type=float, default=1.0, help="strike spacing for the symbol")
    ap.add_argument("--assumed-delta", type=float, default=0.75, help="delta used to size risk at the stop")
    ap.add_argument("--max-alloc-pct", type=float, default=25.0, help="max premium per trade, %% of equity")
    ap.add_argument("--opt-slip", type=float, default=0.03, help="$ per share per side (x100 per contract)")
    ap.add_argument("--opt-commission", type=float, default=0.65, help="$ per contract per side")
    ap.add_argument("--iv", type=float, default=0.18, help="implied vol for model pricing")
    ap.add_argument("--sizing", default="risk", choices=["risk", "premium"],
                    help="risk: size by risk budget; premium: spend --max-alloc-pct of equity on premium")
    ap.add_argument("--min-one", action="store_true",
                    help="take 1 contract when the risk budget rounds to 0 (see --max-risk-pct)")
    ap.add_argument("--max-risk-pct", type=float, default=3.0,
                    help="max est. risk (%% of equity) allowed for a --min-one override")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no API keys)")
    ap.add_argument("--quiet", action="store_true", help="turn off progress bars and API logging")
    ap.add_argument("--chunk-days", type=int, default=30, help="days of history per API request")
    ap.add_argument("--pause", type=float, default=0.35, help="seconds to wait between API requests")
    ap.add_argument("--save-trades", default=None, help="CSV path for the trade log")
    a = ap.parse_args()
    global VERBOSE
    VERBOSE = not a.quiet

    if a.demo:
        log("[DEMO] Generating synthetic bars (no API calls)")
        raw = synthetic_bars(minutes=a.timeframe)
    else:
        raw = fetch_alpaca_bars(a.symbol, a.timeframe, a.days, a.feed, a.chunk_days, a.pause)
    show_sample(raw, "Raw bars", cols=["open", "high", "low", "close", "volume"])
    base = Params(ema_len=a.ema, rr=a.rr, stop_atr=a.stop_atr, min_vol_ratio=a.min_vol_ratio,
                  use_trend=not a.no_trend, allow_short=not a.long_only,
                  slip_per_share=a.slippage, comm_per_share=a.commission)

    if a.optimize:
        optimize(raw, base, a.train_frac)
        return

    log("[PREP] Filtering to regular hours, computing VWAP, EMA, ATR, volume ratio...")
    df = prepare(raw, ema_len=base.ema_len)
    show_sample(df, "Prepared data with indicators",
                cols=["close", "volume", "vwap", "ema", "trend", "atr", "vol_ratio"])
    print(f"Bars: {len(df)} | Days: {df['date'].nunique()} | {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    trades = backtest(df, base, progress=True, desc="Backtesting")
    if a.instrument == "options":
        source = a.opt_source or ("model" if a.demo else "alpaca")
        if a.demo and source == "alpaca":
            log("[DEMO] No real option data in demo mode, switching to the Black-Scholes model")
            source = "model"
        o = OptParams(source=source, itm_pct=a.itm_pct, min_dte=a.min_dte, strike_step=a.strike_step,
                      assumed_delta=a.assumed_delta, max_alloc_pct=a.max_alloc_pct, slip=a.opt_slip,
                      comm=a.opt_commission, iv=a.iv, pause=a.pause,
                      sizing=a.sizing, min_one=a.min_one, max_risk_pct=a.max_risk_pct)
        log(f"[OPT] {len(trades)} underlying signals -> pricing as ITM {'calls/puts'} ({o.itm_pct}% ITM, "
            f"~{o.min_dte} DTE, source={o.source})")
        trades = apply_options(trades, a.symbol, a.capital, a.risk_pct, o, a.timeframe)
        show_sample(trades, "Sample option trades",
                    cols=["entry_time", "side", "opt_symbol", "opt_entry", "opt_exit", "contracts",
                          "R", "pnl_usd", "equity", "reason"])
        print(f"\nSignals: {trades.attrs.get('signals', 0)} | priced: {len(trades)} | "
              f"skipped (no option data / sizing): {trades.attrs.get('skipped', 0)}")
        print_sizing_report(trades, a.capital, a.risk_pct, o)
        print_metrics("ITM options results (R = P&L / planned risk at the underlying stop)", metrics(trades))
        if not trades.empty:
            print(f"Avg return on premium: {trades['premium_return_pct'].mean():+.1f}% per trade")
        print_capital(trades, a.capital,
                      f"sizing={o.sizing}, risk {a.risk_pct}% of equity (delta {o.assumed_delta}), "
                      f"max premium {o.max_alloc_pct}% of equity, {o.source} prices",
                      size_col="contracts", unit="contracts")
    else:
        trades = apply_capital(trades, a.capital, a.risk_pct, a.max_leverage)
        show_sample(trades, "Sample trades",
                    cols=["entry_time", "side", "entry", "exit", "R", "shares", "pnl_usd", "equity", "reason"])
        print_metrics("Full sample", metrics(trades))
        print_capital(trades, a.capital,
                      f"risk {a.risk_pct}% of equity per trade, max leverage {a.max_leverage}x")
    if not trades.empty:
        print("\nExit reasons:", trades["reason"].value_counts().to_dict())
        print("By side:\n", trades.groupby("side")["R"].agg(["count", "mean"]).round(3))
    if a.save_trades and not trades.empty:
        trades.to_csv(a.save_trades, index=False)
        print(f"\nTrade log saved to {a.save_trades}")


if __name__ == "__main__":
    main()
