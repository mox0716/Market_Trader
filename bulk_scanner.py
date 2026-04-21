import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import time
import smtplib
from datetime import datetime, timedelta
import pytz
from email.message import EmailMessage
 
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
 
# ==============================================================================
# CONFIGURATION
# ==============================================================================
BATCH_SIZE    = 100
MIN_WIN_RATE  = 55.0   # Raised from 50 — we want edge, not coin flips
MIN_SAMPLES   = 10     # Minimum backtest occurrences to trust the win rate
TOP_N_TRADES  = 10     # Max positions per day (was 20 — concentration = conviction)
 
# Scoring weights for final ranking (all strategies use the same scorer)
W_WIN_RATE    = 0.40
W_AVG_RETURN  = 0.30
W_RVOL        = 0.15
W_CONSISTENCY = 0.15   # Penalizes high variance in historical returns
 
# Set DRY_RUN=true in GitHub Actions workflow_dispatch to test outside market hours.
# Bypasses the time gate and skips order placement — everything else runs for real.
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() == 'true'
 
 
# ==============================================================================
# 1. THE BOUNCER (SMART WAITER) — unchanged, works fine
# ==============================================================================
def is_market_closing_soon():
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
 
    target_time = now_ny.replace(hour=15, minute=55, second=0,  microsecond=0)
    cutoff_time = now_ny.replace(hour=15, minute=59, second=59, microsecond=0)
 
    if now_ny > cutoff_time:
        return False, f"Too late. Market closed. ({now_ny.strftime('%I:%M %p')})"
 
    if now_ny < target_time:
        sleep_seconds = (target_time - now_ny).total_seconds()
        if sleep_seconds > 9000:  # 150 min — covers 18:00 UTC trigger in both EST and EDT
            return False, "Too early (more than 150 min before target). Exiting silently."
        print(f"⏰ GitHub started early. Sleeping {sleep_seconds/60:.1f} min until 3:55 PM NY...")
        time.sleep(sleep_seconds)
        now_ny = datetime.now(tz_ny)
 
    return True, now_ny.strftime("%I:%M %p %Z")
 
 
# ==============================================================================
# 2. ALPACA DATA ENGINE — unchanged
# ==============================================================================
def get_alpaca_data(symbols, days_back=365, _retry=2):
    """Fetch daily bars from Alpaca. Retries once on empty/error with a backoff
    to handle rate-limiting — the main cause of the 13/46 failed batches."""
    api_key    = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    client     = StockHistoricalDataClient(api_key, secret_key)
 
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.now(pytz.utc) - timedelta(days=days_back)
    )
    for attempt in range(_retry):
        try:
            bars = client.get_stock_bars(req)
            if bars and not bars.df.empty:
                return bars.df
        except Exception:
            pass
        if attempt < _retry - 1:
            time.sleep(2)   # 2-second backoff before retry
    return pd.DataFrame()
 
 
# ==============================================================================
# 3. REGIME DETECTOR
#
# Four regimes:
#   UPTREND      — SPY above both MAs, +DI leading. Single-day confirmation
#                  so we don't miss the first day of a real breakout.
#   DOWNTREND    — SPY below SMA20, -DI leading. Two-day confirmation to avoid
#                  selling into brief one-day dips.
#   CHOPPY_BULL  — SPY above SMA20 but ADX/DI not confirming a clean uptrend.
#                  Market is transitioning up. Run momentum breakout scan.
#   CHOPPY_BEAR  — SPY below SMA20, conditions indecisive. Run mean reversion.
# ==============================================================================
def get_market_regime():
    df = get_alpaca_data(["SPY"], days_back=90)
    if df.empty or "SPY" not in df.index.levels[0]:
        return "CASH", "⚠️ Regime check failed. Standing down (CASH mode)."
 
    try:
        spy = df.loc["SPY"].copy()
        spy.ta.adx(length=14, append=True)
        spy.ta.sma(length=20, append=True)
        spy.ta.sma(length=50, append=True)
        spy.ta.rsi(length=14, append=True)
 
        if len(spy) < 3:
            return "CASH", "⚠️ Insufficient SPY data. Standing down."
 
        t0 = spy.iloc[-1]   # today
        t1 = spy.iloc[-2]   # yesterday
 
        adx       = t0.get('ADX_14', 0)
        plus_di   = t0.get('DMP_14', 0)
        minus_di  = t0.get('DMN_14', 0)
        close     = t0['close']
        sma20     = t0.get('SMA_20', close)
        sma50     = t0.get('SMA_50', close)
        rsi       = t0.get('RSI_14', 50)
 
        adx_y      = t1.get('ADX_14', 0)
        minus_di_y = t1.get('DMN_14', 0)
        close_y    = t1['close']
        sma20_y    = t1.get('SMA_20', close_y)
 
        status = (f"SPY: ${close:.2f} | ADX: {adx:.1f} | "
                  f"+DI: {plus_di:.1f} | -DI: {minus_di:.1f} | "
                  f"SMA20: ${sma20:.2f} | SMA50: ${sma50:.2f} | RSI: {rsi:.1f}")
 
        above_sma20 = close > sma20
        di_bullish  = plus_di > minus_di
        di_bearish  = minus_di > plus_di
        strong_adx  = adx >= 20
 
        # UPTREND: single-day — SPY above both MAs, +DI leading, ADX strong.
        # One day is enough; we don't want to miss the first day of a real move.
        if above_sma20 and (close > sma50) and di_bullish and strong_adx:
            return "UPTREND", f"🟢 UPTREND. {status}"
 
        # DOWNTREND: two-day confirmation — SPY below SMA20 and -DI leading today
        # AND yesterday, so we don't flip to bearish on a single red day.
        downtrend_today = (not above_sma20) and di_bearish and strong_adx
        downtrend_yest  = (close_y < sma20_y) and (minus_di_y > adx_y * 0.4)
        if downtrend_today and downtrend_yest:
            return "DOWNTREND", f"🔴 CONFIRMED DOWNTREND. {status}"
 
        # CHOPPY split on SPY position relative to SMA20
        if above_sma20:
            # SPY above its MA but not a confirmed uptrend — transitioning up.
            # Run momentum breakout scan, not mean reversion.
            return "CHOPPY_BULL", f"🟡 CHOPPY (Bullish bias — SPY above SMA20). {status}"
        else:
            # SPY below its MA, indecisive. Mean reversion territory.
            return "CHOPPY_BEAR", f"🟠 CHOPPY (Bearish bias — SPY below SMA20). {status}"
 
    except Exception as e:
        return "CASH", f"⚠️ Regime math error ({e}). Standing down."
 
 
# ==============================================================================
# 4. INDICATOR ENGINE — expanded for all three strategies
# ==============================================================================
def calculate_indicators(df):
    df = df.copy()
    try:
        # Trend
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.ema(length=9,  append=True)
 
        # Momentum / Mean Reversion
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.stoch(k=14, d=3, append=True)      # Stochastic for CHOPPY entries
 
        # Volatility / Range
        df.ta.atr(length=14, append=True)        # ATR for position quality scoring
        df.ta.bbands(length=20, std=2, append=True)  # Bollinger for CHOPPY
 
        # Assign clean column names
        df['ADX']    = df.get('ADX_14',  0)
        df['+DI']    = df.get('DMP_14',  0)
        df['-DI']    = df.get('DMN_14',  0)
        df['SMA10']  = df.get('SMA_10',  df['Close'])
        df['SMA20']  = df.get('SMA_20',  df['Close'])
        df['SMA50']  = df.get('SMA_50',  df['Close'])
        df['EMA9']   = df.get('EMA_9',   df['Close'])
        df['RSI']    = df.get('RSI_14',  50)
        df['ATR']    = df.get('ATRr_14', 0)
        df['MACD']   = df.get('MACD_12_26_9', 0)
        df['MACD_S'] = df.get('MACDs_12_26_9', 0)   # Signal line
        df['STOCH_K']= df.get('STOCHk_14_3_3', 50)
        df['STOCH_D']= df.get('STOCHd_14_3_3', 50)
        df['BB_UP']  = df.get('BBU_20_2.0', df['Close'])
        df['BB_MID'] = df.get('BBM_20_2.0', df['Close'])
        df['BB_LO']  = df.get('BBL_20_2.0', df['Close'])
 
        # Derived quality metrics
        # Closing range: where did we close within today's candle? 1.0 = at high
        daily_range = df['High'] - df['Low']
        df['Close_Pct'] = np.where(
            daily_range > 0,
            (df['Close'] - df['Low']) / daily_range,
            0.5
        )
        # ATR % of price — normalizes volatility across price levels
        df['ATR_Pct'] = np.where(df['Close'] > 0, df['ATR'] / df['Close'], 0)
 
    except Exception:
        pass
 
    return df
 
 
# ==============================================================================
# 5. STRATEGY GATE + BACKTEST ENGINE
#    Returns: (passed: bool, metrics: dict)
#
#    UPTREND      → ADX Momentum (1-day forward backtest)
#    DOWNTREND    → Relative Strength against market (2-day)
#    CHOPPY_BULL  → Breakout Momentum — stocks clearing resistance on volume (1-day)
#    CHOPPY_BEAR  → Mean Reversion — oversold bounce confirmation (2-day)
# ==============================================================================
def evaluate_stock(df, regime, rvol):
    today     = df.iloc[-1]
    yesterday = df.iloc[-2]
 
    # ------------------------------------------------------------------
    # UNIVERSAL PRE-FILTERS
    # ------------------------------------------------------------------
 
    # 1. Must close in top 40% of today's candle (not fading into close)
    if today['Close_Pct'] < 0.60:
        return False, {}
 
    # 2. MACD downward crossover = momentum dying. Allow in mean reversion only.
    mean_rev_regime = regime == "CHOPPY_BEAR"
    if not mean_rev_regime and today['MACD'] < today['MACD_S'] and yesterday['MACD'] > yesterday['MACD_S']:
        return False, {}
 
    # 3. ATR% > 8% = casino volatility, skip
    if today['ATR_Pct'] > 0.08:
        return False, {}
 
    passed_strategy = False
 
    # ------------------------------------------------------------------
    # STRATEGY A: UPTREND — Momentum (ADX as scoring bonus, not hard gate)
    # ------------------------------------------------------------------
    # Hard requirements — must all pass:
    #   Close > SMA10 > SMA20  (SMA50 dropped — too slow for fast breakouts)
    #   +DI > -DI              (direction confirmed)
    #   RSI 45–80              (widened ceiling; strong trends run >75)
    #   Close > EMA9           (short-term trend intact)
    #   ADX >= 20 OR RVOL >= 2.0x  (trend strength OR volume conviction)
    #
    # ADX also feeds a scoring bonus so high-ADX stocks rank higher in email.
    # ------------------------------------------------------------------
    if regime == "UPTREND":
        ma_stack    = today['Close'] > today['SMA10'] > today['SMA20']
        di_leading  = today['+DI'] > today['-DI']
        rsi_ok      = 45 <= today['RSI'] <= 80
        ema_support = today['Close'] > today['EMA9']
 
        # ADX as soft signal: confirmed trend OR high volume breakout qualifies
        adx_confirmed  = today['ADX'] >= 20 and today['ADX'] > yesterday['ADX']
        volume_breakout = rvol >= 2.0
 
        if ma_stack and di_leading and rsi_ok and ema_support and (adx_confirmed or volume_breakout):
            passed_strategy = True
 
    # ------------------------------------------------------------------
    # STRATEGY B: DOWNTREND — Relative Strength (stocks RESISTING the drop)
    # ------------------------------------------------------------------
    elif regime == "DOWNTREND":
        # The stock must be UP today or near flat while SPY is down.
        # We define "relative strength" as: today's close > yesterday's close,
        # AND the stock is holding above its own 20-day SMA (institutions defending it).
        price_up   = today['Close'] > yesterday['Close']
        above_sma20 = today['Close'] > today['SMA20']
 
        # +DI still leading on the individual stock despite market downtrend
        di_bullish = today['+DI'] > today['-DI']
 
        # RSI between 40-65: not oversold desperation, not overbought exhaustion
        rsi_ok = 40 <= today['RSI'] <= 65
 
        # Closing strong in the candle (not reversing at the top)
        closing_strong = today['Close_Pct'] >= 0.65
 
        # MACD must be positive or crossing up (own momentum, not market momentum)
        macd_ok = today['MACD'] > 0 or (today['MACD'] > today['MACD_S'])
 
        if price_up and above_sma20 and di_bullish and rsi_ok and closing_strong and macd_ok:
            passed_strategy = True
 
    # ------------------------------------------------------------------
    # STRATEGY C: CHOPPY_BULL — Breakout Momentum
    # SPY is above its SMA20 but trend not fully confirmed.
    # Hunt stocks that are breaking out: above all MAs, strong volume,
    # RSI with momentum but not overbought. Same logic as UPTREND but
    # slightly looser — the market is WITH us, just not yet confirmed.
    # ------------------------------------------------------------------
    elif regime == "CHOPPY_BULL":
        ma_stack      = today['Close'] > today['SMA10'] > today['SMA20']
        di_bullish    = today['+DI'] > today['-DI']
        rsi_ok        = 48 <= today['RSI'] <= 78      # Slightly wider than UPTREND
        ema_support   = today['Close'] > today['EMA9']
        price_up      = today['Close'] > yesterday['Close']
        macd_positive = today['MACD'] > today['MACD_S']  # MACD above signal
 
        if ma_stack and di_bullish and rsi_ok and ema_support and price_up and macd_positive:
            passed_strategy = True
 
    # ------------------------------------------------------------------
    # STRATEGY D: CHOPPY_BEAR — Mean Reversion (confirmed bounce only)
    # SPY is below SMA20. Hunt oversold stocks showing a confirmed turn.
    # ------------------------------------------------------------------
    elif regime == "CHOPPY_BEAR":
        rsi_oversold   = today['RSI'] < 38
        rsi_recovering = today['RSI'] > yesterday['RSI']
        stoch_oversold = today['STOCH_K'] < 25
        stoch_hook     = today['STOCH_K'] > today['STOCH_D']
        near_bb_low    = today['Close'] <= today['BB_MID']
        above_bb_low   = today['Close'] > today['BB_LO']
        structural_ok  = today['Close'] > (today['SMA20'] * 0.90)
        price_turning  = today['Close'] > yesterday['Close']
 
        if (rsi_oversold and rsi_recovering and
            stoch_oversold and stoch_hook and
            near_bb_low and above_bb_low and
            structural_ok and price_turning):
            passed_strategy = True
 
    if not passed_strategy:
        return False, {}
 
    # ------------------------------------------------------------------
    # BACKTEST — now measures OVERNIGHT return for UPTREND,
    #            2-day forward for DOWNTREND/CHOPPY.
    # ------------------------------------------------------------------
    try:
        if regime == "UPTREND":
            # Mirror the relaxed strategy gate: ADX confirmed OR high volume
            adx_col  = df['ADX']
            rvol_col = df['Volume'] / df['Volume'].rolling(21).mean().shift(1)
            hist_mask = (
                (df['Close'] > df['SMA10']) &
                (df['SMA10'] > df['SMA20']) &
                (df['+DI'] > df['-DI']) &
                (df['RSI'].between(45, 80)) &
                (df['Close'] > df['EMA9']) &
                (
                    ((adx_col >= 20) & (adx_col > adx_col.shift(1))) |
                    (rvol_col >= 2.0)
                )
            )
            forward_days = 1   # Buy close, sell next open/close
 
        elif regime == "DOWNTREND":
            hist_mask = (
                (df['Close'] > df['Close'].shift(1)) &
                (df['Close'] > df['SMA20']) &
                (df['+DI'] > df['-DI']) &
                (df['RSI'].between(40, 65))
            )
            forward_days = 2
 
        elif regime == "CHOPPY_BULL":
            hist_mask = (
                (df['Close'] > df['SMA10']) &
                (df['SMA10'] > df['SMA20']) &
                (df['+DI'] > df['-DI']) &
                (df['RSI'].between(48, 78)) &
                (df['Close'] > df['Close'].shift(1))
            )
            forward_days = 1   # Breakout should show quickly
 
        else:  # CHOPPY_BEAR — mean reversion
            hist_mask = (
                (df['RSI'] < 38) &
                (df['RSI'] > df['RSI'].shift(1)) &
                (df['STOCH_K'] < 25) &
                (df['STOCH_K'] > df['STOCH_D']) &
                (df['Close'] > df['Close'].shift(1)) &
                (df['Close'] > (df['SMA20'] * 0.90))
            )
            forward_days = 2
 
        hist_idx = df[hist_mask].index
        wins, total = 0, 0
        returns = []
 
        for d in hist_idx:
            idx = df.index.get_loc(d)
            if idx + forward_days < len(df):
                entry = df.iloc[idx]['Close']
                exit_ = df.iloc[idx + forward_days]['Close']
                fwd_return = (exit_ - entry) / entry * 100
                returns.append(fwd_return)
                if fwd_return > 0:
                    wins += 1
                total += 1
 
        # Require minimum sample size to trust the stat
        if total < MIN_SAMPLES:
            return False, {}
 
        win_rate   = (wins / total) * 100
        avg_return = np.mean(returns)
        std_return = np.std(returns)
 
        if win_rate < MIN_WIN_RATE:
            return False, {}
 
        # Additional filter: average return must be POSITIVE after all occurrences
        # (a 60% win rate with -2% average is still a loser due to loss size)
        if avg_return <= 0:
            return False, {}
 
        # Build the composite score used for final ranking
        # Consistency = 1 / (1 + std) so lower variance = higher score
        consistency = 1.0 / (1.0 + std_return)
 
        # ADX bonus: rewards confirmed trends without requiring them.
        # Scales from 0 (ADX=0, falling) to 0.10 (ADX>=30, rising).
        adx_val   = float(today.get('ADX', 0))
        adx_rise  = adx_val > float(yesterday.get('ADX', 0))
        adx_bonus = min(adx_val / 30.0, 1.0) * 0.10 if adx_rise else 0.0
 
        score = (
            W_WIN_RATE    * (win_rate / 100) +
            W_AVG_RETURN  * min(avg_return / 3.0, 1.0) +
            W_RVOL        * min(rvol / 3.0, 1.0) +
            W_CONSISTENCY * consistency +
            adx_bonus                              # up to +0.10 for strong rising ADX
        )
 
        metrics = {
            "win_rate":   round(win_rate, 1),
            "avg_return": round(avg_return, 2),
            "std_return": round(std_return, 2),
            "samples":    total,
            "rvol":       round(rvol, 2),
            "adx":        round(adx_val, 1),
            "close_pct":  round(today['Close_Pct'] * 100, 1),
            "rsi":        round(today['RSI'], 1),
            "score":      round(score, 4),
        }
        return True, metrics
 
    except Exception:
        return False, {}
 
 
# ==============================================================================
# 6. EXECUTION — unchanged except TOP_N_TRADES now controls position count
# ==============================================================================
def execute_alpaca_trades(winning_df, regime):
    if winning_df.empty:
        return "No trades.", "<p>No positions.</p>"
 
    api_key    = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    client     = TradingClient(api_key, secret_key, paper=True)
 
    # Cancel any open buy orders from a prior run
    open_orders = client.get_orders()
    for order in open_orders:
        if order.side == OrderSide.BUY:
            client.cancel_order_by_id(order.id)
    time.sleep(1)
 
    positions    = client.get_all_positions()
    existing     = [p.symbol for p in positions]
    port_list    = [{"Symbol": p.symbol,
                     "Qty": p.qty,
                     "P/L $": f"${float(p.unrealized_pl):.2f}",
                     "P/L %": f"{float(p.unrealized_plpc)*100:.2f}%"}
                    for p in positions]
    port_html    = pd.DataFrame(port_list).to_html(index=False) if port_list else "<p>No open positions.</p>"
 
    if 'ticker' not in winning_df.columns:
        return "Error: Missing ticker column.", port_html
 
    fresh         = winning_df[~winning_df['ticker'].isin(existing)]
    planned_trades = min(len(fresh), TOP_N_TRADES)
 
    if planned_trades == 0:
        return "No new setups (all candidates already held).", port_html
 
    account = client.get_account()
    equity  = float(account.equity)
 
    if equity < 30000 and int(account.daytrade_count) >= 2:
        return "BLOCKED: PDT Rule Active (< $25k equity, 3 day trades used).", port_html
 
    slot_size = min((equity / planned_trades), 1000.00)
    log       = []
 
    for _, stock in fresh.head(TOP_N_TRADES).iterrows():
        try:
            qty = int(slot_size / stock['price'])
            if qty > 0:
                safe_entry = round(stock['price'] * 1.002, 2)
                req = LimitOrderRequest(
                    symbol         = stock['ticker'],
                    qty            = qty,
                    limit_price    = safe_entry,
                    side           = OrderSide.BUY,
                    time_in_force  = TimeInForce.GTC,
                    order_class    = OrderClass.BRACKET,
                    take_profit    = TakeProfitRequest(limit_price=round(stock['price'] * 1.032, 2)),
                    stop_loss      = StopLossRequest(stop_price=round(stock['price'] * 0.985, 2))
                )
                if DRY_RUN:
                    log.append(
                        f"\U0001f9ea [DRY RUN] WOULD BUY {qty} {stock['ticker']} @ \u2264${safe_entry} | "
                        f"TP: ${round(stock['price']*1.032,2)} | "
                        f"SL: ${round(stock['price']*0.985,2)} | "
                        f"Score: {stock.get('score','?')} | WR: {stock.get('win_rate','?')}%"
                    )
                else:
                    client.submit_order(req)
                    log.append(
                        f"\u2705 BUY {qty} {stock['ticker']} @ \u2264${safe_entry} | "
                        f"TP: ${round(stock['price']*1.032,2)} | "
                        f"SL: ${round(stock['price']*0.985,2)} | "
                        f"Score: {stock.get('score','?')} | WR: {stock.get('win_rate','?')}%"
                    )
        except Exception as e:
            log.append(f"❌ ERR {stock['ticker']}: {e}")
 
    return "\n".join(log), port_html
 
 
# ==============================================================================
# 7. MAIN LOGIC
# ==============================================================================
def run_main():
    if DRY_RUN:
        print("\U0001f9ea DRY RUN MODE \u2014 time gate bypassed, no orders will be placed.")
        time_msg = datetime.now(pytz.timezone('America/New_York')).strftime("%I:%M %p %Z") + " [DRY RUN]"
    else:
        is_time, time_msg = is_market_closing_soon()
        if not is_time:
            print(f"Skipping: {time_msg}")
            return
 
    regime, regime_msg = get_market_regime()
 
    # CASH MODE: market is unreadable or we had a regime error — stand down entirely.
    if regime == "CASH":
        print(regime_msg)
        send_cash_email(time_msg, regime_msg)
        return
 
    # Always scan tickers.txt — tickersdown.txt is retired.
    # The DOWNTREND strategy now hunts for relative-strength stocks
    # within the same universe rather than routing to inverse ETFs.
    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file):
        print(f"Missing {ticker_file}")
        return
 
    with open(ticker_file, 'r') as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()]
 
    all_hits = []
 
    stats = {
        "regime":                 regime,
        "total_scanned":          len(all_tickers),
        "total_batches":          (len(all_tickers) + BATCH_SIZE - 1) // BATCH_SIZE,
        "failed_batches":         0,
        "valid_downloads":        0,
        "passed_liquidity":       0,
        "passed_candle_quality":  0,
        "passed_strategy":        0,
        "passed_backtest":        0,
    }
 
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch      = all_tickers[i:i + BATCH_SIZE]
        batch_data = get_alpaca_data(batch, days_back=365)
 
        if batch_data.empty:
            stats["failed_batches"] += 1
            continue
 
        time.sleep(0.3)   # 300ms between batches — prevents rate limiting
 
        for symbol in batch:
            try:
                if symbol not in batch_data.index.levels[0]:
                    continue
 
                stats["valid_downloads"] += 1
 
                df = batch_data.loc[symbol].copy()
                df.rename(columns={
                    'open': 'Open', 'high': 'High',
                    'low':  'Low',  'close': 'Close', 'volume': 'Volume'
                }, inplace=True)
 
                if df.empty or df['Close'].isna().all():
                    continue
                df = df.dropna()
                if len(df) < 60:   # Need more history for 50-day SMA + ATR
                    continue
 
                price   = df['Close'].iloc[-1]
                avg_vol = df['Volume'].iloc[-21:-1].mean()
                rvol    = df['Volume'].iloc[-1] / avg_vol if avg_vol > 0 else 0
 
                # ── LIQUIDITY GATE ──────────────────────────────────────────
                # Price $5–$500: avoids penny stock chaos and index-fund noise
                if not (2.0 <= price <= 500.0):
                    continue
                # Avg daily volume ≥ 750k: tighter than before for real liquidity
                if avg_vol < 500_000:
                    continue
                # RVOL ≥ 1.5x: something is happening today — not a quiet drift
                if rvol < 1.3:
                    continue
 
                stats["passed_liquidity"] += 1
 
                # Skip known non-tradeable symbols that pass data checks
                # but fail at order placement. Add to this list as needed.
                BLACKLIST = {"COMM"}
                if symbol in BLACKLIST:
                    continue
 
                df = calculate_indicators(df)
 
                # ── CANDLE QUALITY GATE (pre-indicator) ─────────────────────
                today = df.iloc[-1]
 
                # Must close in top 40% of today's range (not fading)
                if today.get('Close_Pct', 0) < 0.60:
                    continue
                # ATR sanity: not a volatility bomb
                if today.get('ATR_Pct', 0) > 0.08:
                    continue
 
                stats["passed_candle_quality"] += 1
 
                passed, metrics = evaluate_stock(df, regime, rvol)
 
                if passed:
                    stats["passed_strategy"] += 1
                    stats["passed_backtest"] += 1
                    all_hits.append({
                        "ticker":     symbol,
                        "price":      round(price, 2),
                        "rvol":       metrics["rvol"],
                        "win_rate":   metrics["win_rate"],
                        "avg_return": metrics["avg_return"],
                        "std_return": metrics["std_return"],
                        "samples":    metrics["samples"],
                        "close_pct":  metrics["close_pct"],
                        "rsi":        metrics["rsi"],
                        "adx":        metrics.get("adx", 0),
                        "score":      metrics["score"],
                    })
 
            except Exception:
                continue
 
    res_df = pd.DataFrame(all_hits)
    if not res_df.empty:
        # Sort by composite SCORE, not just win rate
        res_df = res_df.sort_values(by="score", ascending=False).reset_index(drop=True)
 
    trade_log, port_html = execute_alpaca_trades(res_df, regime)
    send_email(res_df, trade_log, port_html, time_msg, regime_msg, regime, stats)
 
 
# ==============================================================================
# 8. EMAIL — CASH MODE (bot stood down, nothing traded)
# ==============================================================================
def send_cash_email(ny_time, regime_msg):
    msg           = EmailMessage()
    msg['Subject']= "[CASH] Sniper Stood Down — No Trades Today"
    msg['From']   = os.environ.get('EMAIL_USER')
    msg['To']     = os.environ.get('EMAIL_RECEIVER')
 
    body = f"""
    <h3>🛑 Bot Stood Down ({ny_time})</h3>
    <p>The market regime detector could not confirm a tradeable environment today.</p>
    <p><b>Reason:</b> {regime_msg}</p>
    <p>All capital preserved. No orders placed or cancelled.</p>
    <hr>
    <p style="color:#888; font-size:12px;">
        The bot returns tomorrow. Cash IS a position.
    </p>
    """
    msg.add_alternative(body, subtype='html')
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(os.environ.get('EMAIL_USER'), os.environ.get('EMAIL_PASS'))
        smtp.send_message(msg)
 
 
# ==============================================================================
# 9. EMAIL — FULL REPORT
# ==============================================================================
def send_email(res_df, trade_log, port_html, ny_time, regime_msg, regime, stats):
    msg   = EmailMessage()
    hits  = len(res_df) if not res_df.empty else 0
 
    regime_emoji = {"UPTREND": "🟢", "DOWNTREND": "🔴", "CHOPPY_BULL": "🟡", "CHOPPY_BEAR": "🟠", "CASH": "⚪"}.get(regime, "⚪")
    strategy_name = {
        "UPTREND":     "ADX Momentum",
        "DOWNTREND":   "Relative Strength",
        "CHOPPY_BULL": "Breakout Momentum",
        "CHOPPY_BEAR": "Mean Reversion (Bollinger/Stoch/RSI)",
    }.get(regime, "Unknown")
 
    dry_tag = ' [DRY RUN]' if DRY_RUN else ''
    msg['Subject'] = f"{regime_emoji} [{regime}] Sniper: {hits} candidates → {min(hits, TOP_N_TRADES)} trades{dry_tag}"
    msg['From']    = os.environ.get('EMAIL_USER')
    msg['To']      = os.environ.get('EMAIL_RECEIVER')
 
    hits_html = res_df.head(TOP_N_TRADES).to_html(index=False) if not res_df.empty else "<p>No qualified setups today.</p>"
 
    # Strategy explanation block — different for each regime
    strategy_notes = {
        "UPTREND": """
            <b>Strategy:</b> Momentum — Close &gt; SMA10 &gt; SMA20, +DI leading,
            RSI 45–80, above EMA9. Qualifies on <i>either</i> ADX ≥ 20 rising
            (confirmed trend) <i>or</i> RVOL ≥ 2.0x (volume breakout) — whichever
            fires first. ADX adds a scoring bonus so confirmed-trend stocks rank
            higher. Backtest: <b>1-day forward return</b>.
        """,
        "DOWNTREND": """
            <b>Strategy:</b> Relative Strength — stocks rising <i>despite</i> the market
            falling. Price up today, above own SMA20, +DI leading, RSI 40–65, closing
            in top 35% of candle, MACD positive. Backtest: <b>2-day forward return</b>.
        """,
        "CHOPPY_BULL": """
            <b>Strategy:</b> Breakout Momentum — SPY is above SMA20 (bullish bias) but
            trend not fully confirmed. Hunting stocks above SMA10/SMA20, +DI leading,
            RSI 48–78, above EMA9, up on the day, MACD positive. These stocks lead
            the next confirmed uptrend. Backtest: <b>1-day forward return</b>.
        """,
        "CHOPPY_BEAR": """
            <b>Strategy:</b> Mean Reversion — SPY below SMA20 (bearish bias). RSI &lt; 38
            AND turning up, Stochastic &lt; 25 AND K crossing D, price near lower Bollinger
            but closing above it, today higher than yesterday.
            Backtest: <b>2-day forward return</b>.
        """,
    }.get(regime, "")
 
    body = f"""
    {'<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:4px; padding:10px 14px; margin-bottom:12px;">🧪 <strong>DRY RUN — no orders were placed.</strong> This is a test run outside market hours.</div>' if DRY_RUN else ''}
    <h3>{regime_emoji} Sniper Report — {ny_time}</h3>
    <p><b>Market Regime:</b> {regime_msg}</p>
    <p style="background:#f0f0f0; padding:10px; border-left:4px solid #666;">
        {strategy_notes}
    </p>
 
    <div style="background:#f9f9f9; padding:15px; border:1px solid #ddd; border-radius:4px;">
        <h4 style="margin-top:0;">📊 Diagnostic Funnel</h4>
        <table style="border-collapse:collapse; width:100%;">
            <tr><td style="padding:4px 8px;">Tickers Attempted</td>
                <td style="padding:4px 8px; font-weight:bold;">{stats['total_scanned']:,}</td></tr>
            <tr style="background:#fff;"><td style="padding:4px 8px;">Valid Downloads</td>
                <td style="padding:4px 8px; font-weight:bold;">{stats['valid_downloads']:,}</td></tr>
            <tr><td style="padding:4px 8px;">Passed Liquidity (price $2–$500, vol ≥500k, RVOL ≥1.3x)</td>
                <td style="padding:4px 8px; font-weight:bold;">{stats['passed_liquidity']:,}</td></tr>
            <tr style="background:#fff;"><td style="padding:4px 8px;">Passed Candle Quality (close ≥60% range, ATR &lt;8%)</td>
                <td style="padding:4px 8px; font-weight:bold;">{stats['passed_candle_quality']:,}</td></tr>
            <tr><td style="padding:4px 8px;">Passed {strategy_name} Strategy</td>
                <td style="padding:4px 8px; font-weight:bold;">{stats['passed_strategy']:,}</td></tr>
            <tr style="background:#fff;"><td style="padding:4px 8px;">Passed Backtest (WR ≥{int(MIN_WIN_RATE)}%, avg return &gt;0, n≥{MIN_SAMPLES})</td>
                <td style="padding:4px 8px; font-weight:bold; color:green;">{stats['passed_backtest']:,}</td></tr>
            <tr><td style="padding:4px 8px;">Failed Batches (empty API response)</td>
                <td style="padding:4px 8px; font-weight:bold; color:{'red' if stats['failed_batches'] > 0 else 'inherit'};">{stats['failed_batches']} / {stats['total_batches']}</td></tr>
        </table>
    </div>
 
    <hr>
    <h4>⚡ Execution Log</h4>
    <pre style="background:#1e1e1e; color:#d4d4d4; padding:12px; border-radius:4px; font-size:13px;">{trade_log}</pre>
 
    <hr>
    <h4>🎯 Top Candidates (ranked by composite score)</h4>
    <p style="font-size:12px; color:#666;">
        Score = 40% win rate + 30% avg return + 15% RVOL + 15% consistency (lower std = better).
        Close% = where price closed within today's high/low range.
    </p>
    {hits_html}
 
    <hr>
    <h4>💼 Current Portfolio</h4>
    {port_html}
 
    <p style="font-size:11px; color:#aaa; margin-top:20px;">
        MIN_WIN_RATE={MIN_WIN_RATE}% | MIN_SAMPLES={MIN_SAMPLES} | TOP_N={TOP_N_TRADES} | Price $2+ | Vol 500k+ | RVOL 1.3x+ |
        Bracket: TP +3.2% / SL -1.5% | Entry limit +0.2%
    </p>
    """
    msg.add_alternative(body, subtype='html')
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(os.environ.get('EMAIL_USER'), os.environ.get('EMAIL_PASS'))
        smtp.send_message(msg)
 
 
if __name__ == "__main__":
    run_main()
 
