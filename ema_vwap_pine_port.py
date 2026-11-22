#!/usr/bin/env python3
"""
Python port of the TradingView Pine Script "EMA + VWAP Strategy" (Pine v5), running on Alpaca data.

Pine logic reproduced
  * Long : EMA(fast) crosses over VWAP, EMA(fast) > EMA(slow), VWAP slope > 0 and not "flat", weekday
  * Short: EMA(fast) crosses under VWAP, EMA(fast) < EMA(slow), VWAP slope < 0 and not "flat", weekday
  * VWAP = ta.vwap(close): CLOSE-weighted by volume (not hlc3), resets every trading day
  * VWAP slope = % change of VWAP over `vwap_lookback` bars; |slope| < flat threshold counts as flat
  * Stop types: ATR | Swing (lowest low / highest high) | SlowEMA | Fixed %; target = rr x risk
  * Stop and target are fixed from the SIGNAL bar's close, and the order fills at the NEXT bar's open
  * Opposite signal reverses the position; a repeat signal in the same direction re-anchors the exit levels
  * ta.ema is seeded with an SMA and ta.atr is Wilder's RMA, exactly as in Pine
  * Same-bar stop/target ambiguity uses TradingView's broker-emulator rule: if the high is closer to the
    open than the low, price is assumed to go O->H->L->C, otherwise O->L->H->C. Gaps fill at the open.
  * No end-of-day exit: like the Pine script, positions can be held overnight
  * Share mode (--instrument shares) matches Pine's defaults: 10,000 capital, 100% of equity per trade,
    no commission, no slippage
  * Options mode (default): the same signals, stops, targets and reversals, expressed as long ITM calls (long)
    and puts (short), priced from Alpaca option bars (--opt-source alpaca) or Black-Scholes (--opt-source model)

Known differences from TradingView (see the notes printed in the README of the chat reply)
  * Trades whose stop lands on the wrong side of the entry (e.g. SlowEMA stop above a long entry) are skipped
  * A position still open on the last bar is closed at the last close and marked "open_at_end"
  * Data differs: Alpaca IEX bars are not identical to TradingView's feed

Setup:
    pip install -r requirements.txt
    export APCA_API_KEY_ID="..." ; export APCA_API_SECRET_KEY="..."

Run:
    python ema_vwap_pine_port.py --symbol SPY --timeframe 5 --days 365     # ITM options (default)
    python ema_vwap_pine_port.py --symbol SPY --instrument shares          # plain shares, Pine-style sizing
    python ema_vwap_pine_port.py --symbol SPY --sl-type Swing --rr 2.5
    python ema_vwap_pine_port.py --symbol QQQ --optimize
    python ema_vwap_pine_port.py --demo                  # synthetic data, no API keys
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
# Pine-style indicators
# --------------------------------------------------------------------------- #
def pine_ema(x, n):
    """ta.ema: alpha = 2/(n+1), seeded with the SMA of the first n values (NaN before that)."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    a = 2.0 / (n + 1)
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def pine_rma(x, n):
    """ta.rma (Wilder): alpha = 1/n, seeded with the SMA of the first n values."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


@dataclass(frozen=True)
class Params:
    ema_fast: int = 9
    ema_slow: int = 21
    vwap_flat_pct: float = 0.01     # |VWAP slope| below this % counts as flat
    vwap_lookback: int = 5
    weekdays_only: bool = True
    sl_type: str = "ATR"            # ATR | Swing | SlowEMA | Fixed
    atr_len: int = 14
    atr_mult: float = 1.5
    swing_lookback: int = 10
    fixed_sl_pct: float = 1.0
    rr: float = 2.0
    allow_short: bool = True
    reanchor: bool = True           # repeat same-direction signal moves the stop/target (Pine strategy.exit)
    capital: float = 10000.0
    qty_pct: float = 100.0          # % of equity per trade (Pine default_qty_value)
    whole_shares: bool = False      # Pine allows fractional quantities
    slip: float = 0.0               # $ per share on market/stop fills (Pine default: none)
    comm: float = 0.0               # $ per share per side (Pine default: none)


def prepare(raw, p: Params, extended=False):
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert(NY) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(NY)
    df = df.sort_index()
    if not extended:  # regular trading hours: bar starts 09:30 .. 15:55
        mins = df.index.hour * 60 + df.index.minute
        df = df[(mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 55)].copy()

    df["date"] = df.index.date
    df["dow"] = df.index.dayofweek                      # Mon=0 .. Sun=6
    pv = (df["close"] * df["volume"]).groupby(df["date"]).cumsum()
    vv = df["volume"].groupby(df["date"]).cumsum()
    df["vwap"] = pv / vv                                # ta.vwap(close)

    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    df["ema_fast"] = pine_ema(c, p.ema_fast)
    df["ema_slow"] = pine_ema(c, p.ema_slow)
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.fmax.reduce(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]))  # first bar: high-low
    df["atr"] = pine_rma(tr, p.atr_len)
    df["swing_low"] = df["low"].rolling(p.swing_lookback).min()
    df["swing_high"] = df["high"].rolling(p.swing_lookback).max()
    prev_v = df["vwap"].shift(p.vwap_lookback)         # reaches into the previous day, as in Pine
    df["vwap_slope"] = (df["vwap"] - prev_v) / prev_v * 100
    return df


def signals(df, p: Params):
    f, v, s, slope = df["ema_fast"], df["vwap"], df["ema_slow"], df["vwap_slope"]
    cross_up = (f.shift(1) <= v.shift(1)) & (f > v)     # ta.crossover(emaFast, vwap)
    cross_dn = (f.shift(1) >= v.shift(1)) & (f < v)     # ta.crossunder(emaFast, vwap)
    flat = slope.abs() < p.vwap_flat_pct
    allow = (df["dow"] < 5) if p.weekdays_only else pd.Series(True, index=df.index)
    long_ = cross_up & (f > s) & (slope > 0) & ~flat & allow
    short_ = cross_dn & (f < s) & (slope < 0) & ~flat & allow
    return long_.to_numpy(), short_.to_numpy()


# --------------------------------------------------------------------------- #
# Strategy (bar-by-bar, mirrors Pine's order processing)
# --------------------------------------------------------------------------- #
def backtest(df, p: Params, progress=False, desc="Backtesting"):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr, sw_lo, sw_hi, ema_s = (df[k].to_numpy(float) for k in ("atr", "swing_low", "swing_high", "ema_slow"))
    long_sig, short_sig = signals(df, p)
    times, n = df.index, len(df)

    equity = p.capital
    pos, qty, entry, stop, tgt, risk_ps, entry_i = 0, 0.0, 0.0, np.nan, np.nan, 0.0, 0
    u_entry = 0.0
    pending, trades, skipped = None, [], 0

    def levels(side, i):
        price = c[i]
        if p.sl_type == "ATR":
            sl = price - side * atr[i] * p.atr_mult
        elif p.sl_type == "Swing":
            sl = sw_lo[i] if side == 1 else sw_hi[i]
        elif p.sl_type == "SlowEMA":
            sl = ema_s[i]
        else:
            sl = price * (1 - side * p.fixed_sl_pct / 100)
        risk = side * (price - sl)
        if not np.isfinite(risk) or risk <= 0:
            return None                                  # stop on the wrong side of price / not available yet
        return sl, price + side * risk * p.rr, risk

    def close_pos(j, px, reason, slip=True):
        nonlocal equity, pos
        fill = px - pos * p.slip if slip else px
        pnl_ps = pos * (fill - entry) - 2 * p.comm
        equity += pnl_ps * qty
        trades.append({"entry_time": times[entry_i], "exit_time": times[j],
                       "side": "long" if pos == 1 else "short", "entry": entry, "exit": fill,
                       "qty": qty, "risk": risk_ps, "u_entry": u_entry, "u_exit": px, "pnl_per_share": pnl_ps, "R": pnl_ps / risk_ps,
                       "pnl_usd": pnl_ps * qty, "equity": equity, "reason": reason})
        pos = 0

    step = max(1, n // 200)
    bar = tqdm(total=n, desc=desc, unit="bar", disable=not (progress and VERBOSE))
    for i in range(n):
        if progress and i % step == 0:
            bar.update(step)
            bar.set_postfix(trades=len(trades), equity=f"${equity:,.0f}")

        # 1) order placed on the previous bar's close fills at this bar's open
        if pending is not None:
            side_n, sl_n, tp_n, ref, risk_n = pending
            pending = None
            if pos != 0:
                close_pos(i, o[i], "reverse")
            q = equity * p.qty_pct / 100 / ref
            if p.whole_shares:
                q = math.floor(q)
            if q > 0 and equity > 0:
                pos, qty, entry, u_entry = side_n, q, o[i] + side_n * p.slip, o[i]
                stop, tgt, risk_ps, entry_i = sl_n, tp_n, risk_n, i

        # 2) stop / target inside this bar (also applies on the entry bar)
        if pos != 0:
            ex = None
            high_first = abs(h[i] - o[i]) < abs(l[i] - o[i])   # TradingView broker-emulator path rule
            if pos == 1:
                if o[i] <= stop:
                    ex = (o[i], "stop")
                elif o[i] >= tgt:
                    ex = (o[i], "target")
                else:
                    hit_s, hit_t = l[i] <= stop, h[i] >= tgt
                    if hit_s and hit_t:
                        ex = (tgt, "target") if high_first else (stop, "stop")
                    elif hit_s:
                        ex = (stop, "stop")
                    elif hit_t:
                        ex = (tgt, "target")
            else:
                if o[i] >= stop:
                    ex = (o[i], "stop")
                elif o[i] <= tgt:
                    ex = (o[i], "target")
                else:
                    hit_s, hit_t = h[i] >= stop, l[i] <= tgt
                    if hit_s and hit_t:
                        ex = (stop, "stop") if high_first else (tgt, "target")
                    elif hit_s:
                        ex = (stop, "stop")
                    elif hit_t:
                        ex = (tgt, "target")
            if ex:
                close_pos(i, ex[0], ex[1], slip=(ex[1] == "stop"))

        # 3) signals evaluated on this bar's close
        side = 1 if long_sig[i] else (-1 if (short_sig[i] and p.allow_short) else 0)
        if side:
            lv = levels(side, i)
            if lv is None:
                skipped += 1
            else:
                sl, tp, rk = lv
                if pos == side:
                    if p.reanchor:                       # strategy.exit() again -> modifies the live exit order
                        stop, tgt = sl, tp
                else:
                    pending = (side, sl, tp, c[i], rk)   # opposite or flat -> market order at next open
    bar.close()

    if pos != 0:
        close_pos(n - 1, c[n - 1], "open_at_end", slip=False)
    out = pd.DataFrame(trades)
    out.attrs["skipped_invalid_sl"] = skipped
    return out


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
# ITM options layer (default instrument)
#   The Pine logic generates entries, stops, targets and reversals on the underlying, exactly as before.
#   Each underlying trade is then expressed as a long ITM option:
#     long trade -> buy an ITM CALL      short trade -> buy an ITM PUT
#   Entry/exit option prices come from Alpaca option bars or a Black-Scholes model.
#   Because this strategy can hold overnight, bars are fetched across the whole holding period.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptParams:
    source: str = "alpaca"        # "alpaca" (real option bars) or "model" (Black-Scholes)
    itm_pct: float = 1.0          # strike this % in the money vs the underlying
    min_dte: int = 14             # target days to expiry (rolled to next weekday)
    strike_step: float = 1.0      # strike spacing ($1 for SPY/QQQ/IWM; use 2.5 or 5 for many stocks)
    assumed_delta: float = 0.75   # used only for position sizing (estimated loss at the stop)
    max_alloc_pct: float = 25.0   # max premium outlay per trade, % of equity
    slip: float = 0.03            # $ per share (x100 per contract) each side, half-spread proxy
    comm: float = 0.65            # $ per contract each side (Alpaca is lower; this is conservative)
    iv: float = 0.18              # implied vol for model mode (constant, so no vega effects)
    rate: float = 0.04
    max_gap_bars: int = 2         # accept an option print up to this many bars after the signal
    max_stale_min: int = 60       # exit price must come from a print within this many minutes
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


def _expiry_close(expiry):
    return pd.Timestamp(expiry).tz_localize(NY) + pd.Timedelta(hours=16)


def model_prices(symbol, t, o: OptParams, minutes):
    is_call = t["side"] == "long"
    K = itm_strike(t["u_entry"], is_call, o.itm_pct, o.strike_step)
    expiry = next_weekday(t["entry_time"].date() + timedelta(days=o.min_dte))
    p0 = bs_price(t["u_entry"], K, _years(expiry, t["entry_time"]), o.rate, o.iv, is_call)
    lag = {"reverse": 0, "stop": minutes / 2, "target": minutes / 2}.get(t["reason"], minutes)
    exit_ts = t["exit_time"] + pd.Timedelta(minutes=lag)
    p1 = bs_price(t["u_exit"], K, _years(expiry, exit_ts), o.rate, o.iv, is_call)
    return dict(sym=occ_symbol(symbol, expiry, is_call, K), expiry=expiry, strike=K, entry=p0, exit=p1,
                past_expiry=bool(exit_ts > _expiry_close(expiry)))


def _option_client():
    from alpaca.data.historical import OptionHistoricalDataClient
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")
    log("[API] Connecting to Alpaca OPTIONS market data client")
    return OptionHistoricalDataClient(key, secret)


def fetch_option_days(client, symbols, d0, d1, minutes, max_retries=5):
    """One request: intraday bars for several candidate contracts from day d0 to day d1."""
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    start = pd.Timestamp(d0).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)
    end = pd.Timestamp(d1).tz_localize(NY) + pd.Timedelta(hours=16)
    req = OptionBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                            start=start.to_pydatetime(), end=end.to_pydatetime())
    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            log(f"[API] GET /v1beta1/options/bars  {d0} -> {d1}  {len(symbols)} candidate contract(s)  "
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


def _price_from_bars(bars, t, o: OptParams, minutes, past_expiry):
    """Entry = open of the first option bar at/after the entry; exit depends on how the underlying trade ended:
       reverse -> open of the exit bar | stop/target -> mid-bar | open_at_end -> close of the exit bar."""
    idx = bars.index.searchsorted(t["entry_time"], side="left")
    if idx >= len(bars):
        return None
    if (bars.index[idx] - t["entry_time"]).total_seconds() > o.max_gap_bars * minutes * 60:
        return None
    pos = bars.index.searchsorted(t["exit_time"], side="right") - 1
    if pos < idx:
        return None
    bar, same_bar = bars.iloc[pos], bars.index[pos] == t["exit_time"]
    if not past_expiry and (t["exit_time"] - bars.index[pos]).total_seconds() > o.max_stale_min * 60:
        return None                                   # last print is too old to trust as an exit price
    if same_bar and t["reason"] == "reverse":
        exit_px = bar["open"]
    elif same_bar and t["reason"] in ("stop", "target"):
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
    last_day = min(t["exit_time"].date(), expiries[-1])
    df = fetch_option_days(client, [c[0] for c in cands], t["entry_time"].date(), last_day, minutes)
    if df is None or len(df) == 0:
        return None
    df = df.reset_index()
    for sym, ex, k in cands:  # earliest expiry first, target strike before deeper ITM
        part = df[df["symbol"] == sym]
        if part.empty:
            continue
        bars = part.set_index("timestamp").sort_index()
        bars.index = bars.index.tz_convert(NY)
        past = bool(t["exit_time"] > _expiry_close(ex))
        px = _price_from_bars(bars, t, o, minutes, past)
        if px:
            return dict(sym=sym, expiry=ex, strike=k, past_expiry=past, **px)
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


def optimize(raw, base: Params, extended=False, train_frac=0.6, min_trades=30, top=5):
    grid = dict(
        sl_type=["ATR", "Swing", "SlowEMA", "Fixed"],
        atr_mult=[1.0, 1.5, 2.0],
        rr=[1.5, 2.0, 3.0],
        ema_slow=[21, 50],
        vwap_flat_pct=[0.0, 0.01, 0.03],
    )
    log("[OPT] Preparing indicators for each slow-EMA length...")
    prepared = {n: prepare(raw, replace(base, ema_slow=n), extended) for n in grid["ema_slow"]}
    combos = {}
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals))
        if kw["sl_type"] != "ATR":
            kw["atr_mult"] = base.atr_mult              # multiplier is irrelevant for other stop types
        combos[replace(base, **kw)] = None
    combos = list(combos)
    ndays = prepared[grid["ema_slow"][0]]["date"].nunique()
    log(f"[OPT] {len(combos)} unique combinations | {ndays} trading days | train {train_frac:.0%} "
        f"| min {min_trades} trades in-sample")

    results, best, t0 = [], float("-inf"), time.perf_counter()
    bar = tqdm(combos, desc="Optimizing", unit="combo", disable=not VERBOSE)
    for p in bar:
        train, _ = split_by_days(prepared[p.ema_slow], train_frac)
        m = metrics(backtest(train, p))
        if m["trades"] >= min_trades:
            results.append((m["expectancy_R"], p, m))
            best = max(best, m["expectancy_R"])
        bar.set_postfix(valid=len(results), best_exp=f"{best:+.3f}R" if results else "n/a")
    log(f"[OPT] Grid finished in {time.perf_counter() - t0:.1f}s")

    if not results:
        print(f"No parameter set reached {min_trades} in-sample trades. Try --min-trades lower, more --days, "
              f"or a shorter --timeframe.")
        return
    results.sort(key=lambda x: x[0], reverse=True)
    print(f"\n{len(results)} of {len(combos)} combos met the {min_trades}-trade minimum in-sample.")
    print(f"Top {top} by TRAIN expectancy, with their unseen TEST results:")
    for rank, (_, p, mtr) in enumerate(results[:top], 1):
        _, test = split_by_days(prepared[p.ema_slow], train_frac)
        mte = metrics(backtest(test, p, progress=True, desc=f"Validating #{rank} on test data"))
        nan = float("nan")
        print(f"\n#{rank} sl={p.sl_type} atr_mult={p.atr_mult} rr={p.rr} slow_ema={p.ema_slow} "
              f"vwap_flat<{p.vwap_flat_pct}%")
        print(f"   TRAIN: n={mtr['trades']:4d} win={mtr['win_rate'] * 100:5.1f}% "
              f"exp={mtr['expectancy_R']:+.3f}R PF={mtr['profit_factor']:.2f}")
        print(f"   TEST : n={mte['trades']:4d} win={mte['win_rate'] * 100 if mte['trades'] else nan:5.1f}% "
              f"exp={mte['expectancy_R'] if mte['trades'] else nan:+.3f}R "
              f"PF={mte['profit_factor'] if mte['trades'] else nan:.2f}")
    print("\nOnly trust a setup if TEST expectancy stays positive and close to TRAIN. "
          "A big drop means the parameters were overfit.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Python port of the Pine 'EMA + VWAP Strategy' on Alpaca data")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--timeframe", type=int, default=5, help="bar size in minutes")
    ap.add_argument("--days", type=int, default=365, help="calendar days of history")
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"], help="iex is free; sip needs a paid plan")
    ap.add_argument("--extended-hours", action="store_true", help="keep pre/post-market bars (default: RTH only)")
    # Pine inputs
    ap.add_argument("--ema-fast", type=int, default=9)
    ap.add_argument("--ema-slow", type=int, default=21)
    ap.add_argument("--vwap-flat", type=float, default=0.01, help="VWAP flat threshold, %%")
    ap.add_argument("--vwap-lookback", type=int, default=5)
    ap.add_argument("--no-weekday-filter", action="store_true")
    ap.add_argument("--sl-type", default="ATR", choices=["ATR", "Swing", "SlowEMA", "Fixed"])
    ap.add_argument("--atr-len", type=int, default=14)
    ap.add_argument("--atr-mult", type=float, default=1.5)
    ap.add_argument("--swing-lookback", type=int, default=10)
    ap.add_argument("--fixed-sl", type=float, default=1.0, help="fixed stop loss, %%")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--no-reanchor", action="store_true", help="ignore repeat signals while already in a position")
    # account / costs
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--qty-pct", type=float, default=100.0, help="%% of equity per trade")
    ap.add_argument("--whole-shares", action="store_true")
    ap.add_argument("--slippage", type=float, default=0.0, help="$ per share on market/stop fills")
    ap.add_argument("--commission", type=float, default=0.0, help="$ per share per side")
    # instrument / options
    ap.add_argument("--instrument", default="options", choices=["options", "shares"],
                    help="trade ITM options (default) or plain shares")
    ap.add_argument("--opt-source", default=None, choices=["alpaca", "model"],
                    help="option prices: real Alpaca bars, or Black-Scholes model (default: alpaca; model in --demo)")
    ap.add_argument("--itm-pct", type=float, default=1.0, help="how far in the money the strike is, %% of price")
    ap.add_argument("--min-dte", type=int, default=14, help="target days to expiry (trades can hold overnight)")
    ap.add_argument("--strike-step", type=float, default=1.0, help="strike spacing for the symbol")
    ap.add_argument("--risk-pct", type=float, default=1.0, help="options: %% of equity risked per trade")
    ap.add_argument("--assumed-delta", type=float, default=0.75, help="delta used to size risk at the stop")
    ap.add_argument("--max-alloc-pct", type=float, default=25.0, help="options: max premium per trade, %% of equity")
    ap.add_argument("--opt-slip", type=float, default=0.03, help="$ per share per side (x100 per contract)")
    ap.add_argument("--opt-commission", type=float, default=0.65, help="$ per contract per side")
    ap.add_argument("--iv", type=float, default=0.18, help="implied vol for model pricing")
    ap.add_argument("--sizing", default="risk", choices=["risk", "premium"],
                    help="risk: size by risk budget; premium: spend --max-alloc-pct of equity on premium")
    ap.add_argument("--min-one", action="store_true",
                    help="take 1 contract when the risk budget rounds to 0 (see --max-risk-pct)")
    ap.add_argument("--max-risk-pct", type=float, default=3.0,
                    help="max est. risk (%% of equity) allowed for a --min-one override")
    # run control
    ap.add_argument("--optimize", action="store_true", help="grid search on train split, report on test split")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--min-trades", type=int, default=30, help="min in-sample trades for --optimize")
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

    p = Params(ema_fast=a.ema_fast, ema_slow=a.ema_slow, vwap_flat_pct=a.vwap_flat, vwap_lookback=a.vwap_lookback,
               weekdays_only=not a.no_weekday_filter, sl_type=a.sl_type, atr_len=a.atr_len, atr_mult=a.atr_mult,
               swing_lookback=a.swing_lookback, fixed_sl_pct=a.fixed_sl, rr=a.rr, allow_short=not a.long_only,
               reanchor=not a.no_reanchor, capital=a.capital, qty_pct=a.qty_pct, whole_shares=a.whole_shares,
               slip=a.slippage, comm=a.commission)

    if a.optimize:
        log("[OPT] Optimizing the underlying Pine signal (share-based R); option settings are not tuned.")
        optimize(raw, p, a.extended_hours, a.train_frac, a.min_trades)
        return

    log("[PREP] Computing VWAP(close), EMAs, ATR (Wilder), swing levels, VWAP slope...")
    df = prepare(raw, p, a.extended_hours)
    show_sample(df, "Prepared data with indicators",
                cols=["close", "volume", "vwap", "ema_fast", "ema_slow", "atr", "vwap_slope"])
    print(f"Bars: {len(df)} | Days: {df['date'].nunique()} | {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    trades = backtest(df, p, progress=True, desc="Backtesting")
    bh = df["close"].iloc[-1] / df["open"].iloc[0] - 1
    print(f"\nUnderlying signals/trades: {len(trades)} | skipped (invalid stop level): "
          f"{trades.attrs.get('skipped_invalid_sl', 0)}")

    if a.instrument == "options":
        source = a.opt_source or ("model" if a.demo else "alpaca")
        if a.demo and source == "alpaca":
            log("[DEMO] No real option data in demo mode, switching to the Black-Scholes model")
            source = "model"
        o = OptParams(source=source, itm_pct=a.itm_pct, min_dte=a.min_dte, strike_step=a.strike_step,
                      assumed_delta=a.assumed_delta, max_alloc_pct=a.max_alloc_pct, slip=a.opt_slip,
                      comm=a.opt_commission, iv=a.iv, pause=a.pause,
                      sizing=a.sizing, min_one=a.min_one, max_risk_pct=a.max_risk_pct)
        log(f"[OPT] Pricing {len(trades)} underlying trades as long ITM calls/puts ({o.itm_pct}% ITM, "
            f"~{o.min_dte} DTE, source={o.source})")
        n_signals = len(trades)
        trades = apply_options(trades, a.symbol, a.capital, a.risk_pct, o, a.timeframe)
        show_sample(trades, "Sample option trades",
                    cols=["entry_time", "exit_time", "side", "opt_symbol", "opt_entry", "opt_exit",
                          "contracts", "R", "pnl_usd", "equity", "reason"])
        print(f"Priced: {len(trades)} of {n_signals} | skipped (no option data / sizing): "
              f"{trades.attrs.get('skipped', 0)}")
        print_sizing_report(trades, a.capital, a.risk_pct, o)
        print_metrics("Pine strategy port as ITM options (R = P&L / planned risk at the underlying stop)",
                      metrics(trades))
        if not trades.empty:
            overnight = int((trades["exit_time"].dt.date != trades["entry_time"].dt.date).sum())
            print(f"Avg return on premium: {trades['premium_return_pct'].mean():+.1f}% per trade")
            print(f"Held overnight: {overnight} of {len(trades)} trades | "
                  f"held past expiry: {int(trades['past_expiry'].sum())}")
        print_capital(trades, a.capital,
                      f"sizing={o.sizing}, risk {a.risk_pct}% of equity (delta {o.assumed_delta}), "
                      f"max premium {o.max_alloc_pct}% of equity, {o.source} prices, ~{o.min_dte} DTE",
                      size_col="contracts", unit="contracts")
        print(f"Underlying buy & hold: {bh * 100:+.2f}% over the same bars")
    else:
        show_sample(trades, "Sample trades",
                    cols=["entry_time", "side", "entry", "exit", "qty", "R", "pnl_usd", "equity", "reason"])
        print_metrics("Pine strategy port (shares)", metrics(trades))
        print_capital(trades, a.capital,
                      f"{a.qty_pct}% of equity per trade ({'whole' if a.whole_shares else 'fractional'} shares), "
                      f"SL={a.sl_type}, RR={a.rr}, slippage ${a.slippage}/sh, commission ${a.commission}/sh",
                      size_col="qty", unit="shares")
        print(f"Buy & hold      : {bh * 100:+.2f}% over the same bars (${a.capital * (1 + bh):,.2f} end capital)")
    if not trades.empty:
        print("\nExit reasons:", trades["reason"].value_counts().to_dict())
        print("By side:\n", trades.groupby("side")["R"].agg(["count", "mean"]).round(3))
    if a.save_trades and not trades.empty:
        trades.to_csv(a.save_trades, index=False)
        print(f"\nTrade log saved to {a.save_trades}")


if __name__ == "__main__":
    main()
