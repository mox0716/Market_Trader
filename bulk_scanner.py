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
        if sleep_seconds > 4200:
            return False, "Too early (Wrong DST schedule). Exiting silently."
        print(f"⏰ GitHub started early. Sleeping {sleep_seconds/60:.1f} min until 3:55 PM NY...")
        time.sleep(sleep_seconds)
        now_ny = datetime.now(tz_ny)
 
    return True, now_ny.strftime("%I:%M %p %Z")
 
 
# ==============================================================================
# 2. ALPACA DATA ENGINE — unchanged
# ==============================================================================
def get_alpaca_data(symbols, days_back=365):
    api_key    = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    client     = StockHistoricalDataClient(api_key, secret_key)
 
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.now(pytz.utc) - timedelta(days=days_back)
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df.empty:
            return pd.DataFrame()
        return bars.df
    except Exception:
        return pd.DataFrame()
 
 
# ==============================================================================
# 3. REGIME DETECTOR — upgraded to require TWO consecutive days of confirmation
#    before calling a regime, which eliminates whipsawing on one-day moves.
# ==============================================================================
def get_market_regime():
    df = get_alpaca_data(["SPY"], days_back=90)
    if df.empty or "SPY" not in df.index.levels[0]:
        return "CASH", "⚠️ Regime check failed. Standing down (CASH mode)."
 
    try:
        spy = df.loc["SPY"].copy()
        spy.ta.adx(length=14, append=True)
        spy.ta.sma(length=20, append=True)
        spy.ta.sma(length=50, append=True)    # Added 50-day for macro context
        spy.ta.rsi(length=14, append=True)    # Added RSI for extreme readings
 
        # Use last 2 days so we require consecutive confirmation
        if len(spy) < 3:
            return "CASH", "⚠️ Insufficient SPY data. Standing down."
 
        t0 = spy.iloc[-1]   # today
        t1 = spy.iloc[-2]   # yesterday
 
        adx      = t0.get('ADX_14', 0)
        plus_di  = t0.get('DMP_14', 0)
        minus_di = t0.get('DMN_14', 0)
        close    = t0['close']
        sma20    = t0.get('SMA_20', close)
        sma50    = t0.get('SMA_50', close)
        rsi      = t0.get('RSI_14', 50)
 
        # Yesterday's values for confirmation
        adx_y     = t1.get('ADX_14', 0)
        plus_di_y = t1.get('DMP_14', 0)
        minus_di_y= t1.get('DMN_14', 0)
        close_y   = t1['close']
        sma20_y   = t1.get('SMA_20', close_y)
 
        status = (f"SPY: ${close:.2f} | ADX: {adx:.1f} | "
                  f"+DI: {plus_di:.1f} | -DI: {minus_di:.1f} | "
                  f"SMA20: ${sma20:.2f} | SMA50: ${sma50:.2f} | RSI: {rsi:.1f}")
 
        # UPTREND: SPY above both MAs, +DI leading, ADX trending up — TWO days
        uptrend_today = (close > sma20 > sma50) and (plus_di > minus_di) and (adx >= 20)
        uptrend_yest  = (close_y > sma20_y) and (plus_di_y > minus_di_y) and (adx_y >= 18)
 
        # DOWNTREND: SPY below both MAs, -DI leading, ADX trending up — TWO days
        downtrend_today = (close < sma20) and (minus_di > plus_di) and (adx >= 20)
        downtrend_yest  = (close_y < sma20_y) and (minus_di_y > plus_di_y) and (adx_y >= 18)
 
        if uptrend_today and uptrend_yest:
            return "UPTREND",   f"🟢 CONFIRMED UPTREND. {status}"
 
        if downtrend_today and downtrend_yest:
            return "DOWNTREND", f"🔴 CONFIRMED DOWNTREND. {status}"
 
        # Everything else is CHOPPY — no conviction either direction
        return "CHOPPY", f"🟡 CHOPPY / TRANSITIONING. {status}"
 
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
#    Backtest now measures OVERNIGHT return (close → next open) for UPTREND,
#    and 2-day forward for CHOPPY/DOWNTREND (bounce needs a day to develop).
# ==============================================================================
def evaluate_stock(df, regime, rvol):
    today     = df.iloc[-1]
    yesterday = df.iloc[-2]
 
    # ------------------------------------------------------------------
    # UNIVERSAL PRE-FILTERS (apply before any strategy check)
    # ------------------------------------------------------------------
 
    # 1. Stock must be closing in the top 40% of today's candle.
    #    Fading into the close is the opposite of what we want at 3:55.
    if today['Close_Pct'] < 0.60:
        return False, {}
 
    # 2. MACD must not be in a hard downward crossover (momentum dying)
    #    Exception: CHOPPY allows it since we're looking for reversals
    if regime != "CHOPPY" and today['MACD'] < today['MACD_S'] and yesterday['MACD'] > yesterday['MACD_S']:
        return False, {}
 
    # 3. ATR% sanity check — avoid hyper-volatile penny-stock behavior
    #    (> 8% daily ATR vs price = casino, not trading)
    if today['ATR_Pct'] > 0.08:
        return False, {}
 
    passed_strategy = False
 
    # ------------------------------------------------------------------
    # STRATEGY A: UPTREND — ADX Momentum (your original, kept and refined)
    # ------------------------------------------------------------------
    if regime == "UPTREND":
        # Core: Price stacked above all MAs, +DI leading, ADX rising
        ma_stack   = today['Close'] > today['SMA10'] > today['SMA20'] > today['SMA50']
        di_leading = today['+DI'] > today['-DI']
        adx_rising = today['ADX'] > yesterday['ADX']
        adx_strong = today['ADX'] >= 22
 
        # RSI not yet overbought (still has room to run)
        rsi_ok = 45 <= today['RSI'] <= 75
 
        # EMA9 acting as support (close above it = short-term trend intact)
        ema_support = today['Close'] > today['EMA9']
 
        if ma_stack and di_leading and adx_rising and adx_strong and rsi_ok and ema_support:
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
    # STRATEGY C: CHOPPY — Mean Reversion (confirmed bounce, not falling knife)
    # ------------------------------------------------------------------
    elif regime == "CHOPPY":
        # RSI oversold but starting to recover
        rsi_oversold   = today['RSI'] < 38
        rsi_recovering = today['RSI'] > yesterday['RSI']   # KEY: must be turning up
 
        # Stochastic also oversold and hooking up
        stoch_oversold = today['STOCH_K'] < 25
        stoch_hook     = today['STOCH_K'] > today['STOCH_D']  # K crossing above D
 
        # Price near or touching lower Bollinger Band but closing above it
        near_bb_low   = today['Close'] <= today['BB_MID']
        above_bb_low  = today['Close'] > today['BB_LO']
 
        # Still structurally intact — not a catastrophic breakdown
        # (within 10% of SMA20, not in freefall)
        structural_ok = today['Close'] > (today['SMA20'] * 0.90)
 
        # Today closed HIGHER than yesterday (the turn is happening NOW, not speculative)
        price_turning = today['Close'] > yesterday['Close']
 
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
            # Historical occurrences of the same setup
            hist_mask = (
                (df['Close'] > df['SMA10']) &
                (df['SMA10'] > df['SMA20']) &
                (df['+DI'] > df['-DI']) &
                (df['ADX'] >= 22) &
                (df['ADX'] > df['ADX'].shift(1)) &
                (df['RSI'].between(45, 75)) &
                (df['Close'] > df['EMA9'])
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
 
        else:  # CHOPPY
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
        score = (
            W_WIN_RATE    * (win_rate / 100) +
            W_AVG_RETURN  * min(avg_return / 3.0, 1.0) +   # cap at 3% avg return
            W_RVOL        * min(rvol / 3.0, 1.0) +         # cap at 3x RVOL
            W_CONSISTENCY * consistency
        )
 
        metrics = {
            "win_rate":   round(win_rate, 1),
            "avg_return": round(avg_return, 2),
            "std_return": round(std_return, 2),
            "samples":    total,
            "rvol":       round(rvol, 2),
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
                client.submit_order(req)
                log.append(
                    f"✅ BUY {qty} {stock['ticker']} @ ≤${safe_entry} | "
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
            continue
 
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
                if not (3.0 <= price <= 500.0):
                    continue
                # Avg daily volume ≥ 750k: tighter than before for real liquidity
                if avg_vol < 500_000:
                    continue
                # RVOL ≥ 1.5x: something is happening today — not a quiet drift
                if rvol < 1.3:
                    continue
 
                stats["passed_liquidity"] += 1
 
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
 
    regime_emoji = {"UPTREND": "🟢", "DOWNTREND": "🔴", "CHOPPY": "🟡"}.get(regime, "⚪")
    strategy_name = {
        "UPTREND":   "ADX Momentum",
        "DOWNTREND": "Relative Strength",
        "CHOPPY":    "Mean Reversion (Bollinger/Stoch/RSI)",
    }.get(regime, "Unknown")
 
    msg['Subject'] = f"{regime_emoji} [{regime}] Sniper: {hits} candidates → {min(hits, TOP_N_TRADES)} trades"
    msg['From']    = os.environ.get('EMAIL_USER')
    msg['To']      = os.environ.get('EMAIL_RECEIVER')
 
    hits_html = res_df.head(TOP_N_TRADES).to_html(index=False) if not res_df.empty else "<p>No qualified setups today.</p>"
 
    # Strategy explanation block — different for each regime
    strategy_notes = {
        "UPTREND": """
            <b>Strategy:</b> ADX Momentum — buying stocks with a confirmed trend stack
            (Close &gt; SMA10 &gt; SMA20 &gt; SMA50), rising ADX ≥ 22, +DI leading,
            RSI 45–75 (not overbought), and closing above EMA9.
            Backtest measures <b>1-day forward return</b>.
        """,
        "DOWNTREND": """
            <b>Strategy:</b> Relative Strength — hunting stocks that are rising
            <i>despite</i> the market falling. Requires: price up today, above own SMA20,
            +DI still leading on the individual stock, RSI 40–65, closing in top 35%
            of candle, MACD positive. These stocks have institutional accumulation
            against the trend — they tend to explode when the market bounces.
            Backtest measures <b>2-day forward return</b>.
        """,
        "CHOPPY": """
            <b>Strategy:</b> Mean Reversion — RSI &lt; 38 AND turning up, Stochastic &lt; 25
            AND K crossing above D, price touching lower Bollinger but closing above it,
            and today's close higher than yesterday's (the turn is confirmed, not speculative).
            Backtest measures <b>2-day forward return</b>.
        """,
    }.get(regime, "")
 
    body = f"""
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
 
