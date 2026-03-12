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

# --- CONFIGURATION ---
BATCH_SIZE = 100       
MIN_WIN_RATE = 50.0    

# --- 1. THE BOUNCER (SMART WAITER) ---
def is_market_closing_soon():
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # Target execution time: Exactly 3:55:00 PM NY Time (2:55 PM CT)
    target_time = now_ny.replace(hour=15, minute=55, second=0, microsecond=0)
    cutoff_time = now_ny.replace(hour=15, minute=59, second=59, microsecond=0)

    if now_ny > cutoff_time:
        return False, f"Too late. Market closed. ({now_ny.strftime('%I:%M %p')})"

    if now_ny < target_time:
        sleep_seconds = (target_time - now_ny).total_seconds()
        if sleep_seconds > 4200:
            return False, "Too early (Wrong DST schedule). Exiting silently."
            
        print(f"⏰ GitHub started early. Sleeping for {sleep_seconds/60:.1f} minutes until exactly 3:55 PM NY Time...")
        time.sleep(sleep_seconds)
        now_ny = datetime.now(tz_ny)

    return True, now_ny.strftime("%I:%M %p %Z")

# --- 2. ALPACA DATA ENGINE ---
def get_alpaca_data(symbols, days_back=365):
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    client = StockHistoricalDataClient(api_key, secret_key)
    
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
    except Exception as e:
        return pd.DataFrame()

# --- 3. THE REGIME FILTER ---
def get_market_regime():
    df = get_alpaca_data(["SPY"], days_back=60)
    if df.empty or "SPY" not in df.index.levels[0]:
        return "CHOPPY", "⚠️ Regime Check Failed. Defaulting to CHOPPY."
    
    try:
        spy_df = df.loc["SPY"].copy()
        
        # Calculate ADX, +DI, -DI, and SMA20 for SPY
        spy_df.ta.adx(length=14, append=True)
        spy_df.ta.sma(length=20, append=True)
        
        # Extract latest values
        today = spy_df.iloc[-1]
        adx = today.get('ADX_14', 0)
        plus_di = today.get('DMP_14', 0)
        minus_di = today.get('DMN_14', 0)
        close = today['close']
        sma20 = today.get('SMA_20', close)
        
        status_msg = f"SPY: ${close:.2f} | ADX: {adx:.1f} | +DI: {plus_di:.1f} | -DI: {minus_di:.1f} | SMA20: ${sma20:.2f}"
        
        # 1. Choppy / Range-Bound
        if adx < 20:
            return "CHOPPY", f"🟡 MARKET CHOPPY (No Trend). {status_msg}"
            
        # 2. Trending Up
        elif close > sma20 and plus_di > minus_di:
            return "UPTREND", f"🟢 MARKET UPTREND. {status_msg}"
            
        # 3. Trending Down
        elif close < sma20 and minus_di > plus_di:
            return "DOWNTREND", f"🔴 MARKET DOWNTREND. {status_msg}"
            
        else:
            return "CHOPPY", f"🟡 MARKET TRANSITIONING. {status_msg}"
            
    except Exception as e:
        return "CHOPPY", f"⚠️ Regime Math Error. Defaulting to CHOPPY."

# --- 4. PRO INDICATORS ---
def calculate_indicators(df):
    df = df.copy()
    try:
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.rsi(length=14, append=True) # Added RSI for Mean Reversion
        
        df['ADX'] = df['ADX_14'] if 'ADX_14' in df.columns else 0
        df['+DI'] = df['DMP_14'] if 'DMP_14' in df.columns else 0
        df['-DI'] = df['DMN_14'] if 'DMN_14' in df.columns else 0
        df['SMA10'] = df['SMA_10'] if 'SMA_10' in df.columns else 0
        df['SMA20'] = df['SMA_20'] if 'SMA_20' in df.columns else 0
        df['RSI'] = df['RSI_14'] if 'RSI_14' in df.columns else 50
    except:
        return df
    return df

# --- 5. EXECUTION ---
def execute_alpaca_trades(winning_df):
    if winning_df.empty: return "No trades.", "<p>No positions.</p>"

    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    client = TradingClient(api_key, secret_key, paper=True) 
    
    open_orders = client.get_orders()
    for order in open_orders:
        if order.side == OrderSide.BUY:
            client.cancel_order_by_id(order.id)
    time.sleep(1)

    positions = client.get_all_positions()
    existing = [p.symbol for p in positions]
    port_list = [{"Symbol": p.symbol, "P/L": f"${float(p.unrealized_pl):.2f}"} for p in positions]
    port_html = pd.DataFrame(port_list).to_html(index=False) if port_list else "<p>No positions.</p>"

    if 'ticker' not in winning_df.columns: return "Error: Missing ticker col", port_html
    
    fresh = winning_df[~winning_df['ticker'].isin(existing)]
    
    planned_trades = min(len(fresh), 20)
    if planned_trades == 0: 
        return "No new setups.", port_html

    account = client.get_account()
    equity = float(account.equity)
    
    if equity < 30000 and int(account.daytrade_count) >= 2:
        return f"BLOCKED: PDT Active.", port_html
    
    slot_size = min((equity / planned_trades), 1000.00)
    log = []

    for _, stock in fresh.head(20).iterrows():
        try:
            qty = int(slot_size / stock['price'])
            if qty > 0:
                safe_entry_price = round(stock['price'] * 1.002, 2)
                req = LimitOrderRequest(
                    symbol=stock['ticker'], 
                    qty=qty, 
                    limit_price=safe_entry_price,         
                    side=OrderSide.BUY, 
                    time_in_force=TimeInForce.GTC, 
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(stock['price'] * 1.032, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stock['price'] * 0.985, 2))
                )
                client.submit_order(req)
                log.append(f"PLACED LIMIT BUY {qty} {stock['ticker']} @ up to ${safe_entry_price}")
        except Exception as e: log.append(f"Err {stock['ticker']}: {e}")
            
    return "\n".join(log), port_html

# --- 6. MAIN LOGIC ---
def run_main():
    is_time, time_msg = is_market_closing_soon()
    if not is_time:
        print(f"Skipping: {time_msg}")
        return

    # Evaluate the Market Regime
    regime, regime_msg = get_market_regime()
    
    # Route to the correct ticker file based on Regime
    ticker_file = "tickersdown.txt" if regime == "DOWNTREND" else "tickers.txt"
    
    if not os.path.exists(ticker_file): 
        print(f"Missing {ticker_file}")
        return
        
    with open(ticker_file, 'r') as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()]

    all_hits = []
    error_log = []
    
    stats = {
        "regime": regime,
        "total_scanned": len(all_tickers),
        "valid_downloads": 0,
        "passed_volume_filter": 0,
        "passed_regime_strategy": 0,
        "passed_backtest": 0
    }
    
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i+BATCH_SIZE]
        batch_data = get_alpaca_data(batch, days_back=365)
        
        if batch_data.empty: continue

        for symbol in batch:
            try:
                if symbol not in batch_data.index.levels[0]: continue
                stats["valid_downloads"] += 1
                
                df = batch_data.loc[symbol].copy()
                df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
                
                if df.empty or df['Close'].isna().all(): continue
                df = df.dropna()
                if len(df) < 50: continue
                
                price = df['Close'].iloc[-1]
                avg_vol = df['Volume'].iloc[-21:-1].mean()
                
                # Strict Liquidity & Momentum Filters
                if price < 1.0 or avg_vol < 500000: continue
                if (df['Volume'].iloc[-1] / avg_vol) < 1.3: continue
                
                stats["passed_volume_filter"] += 1
                
                df = calculate_indicators(df)
                today = df.iloc[-1]
                yesterday = df.iloc[-2]
                
                passed_strategy = False
                
                # STRATEGY 1: UPTREND or DOWNTREND (Momentum)
                # Note: We use upward momentum for DOWNTREND too, because tickersdown.txt should hold inverse ETFs.
                if regime in ["UPTREND", "DOWNTREND"]:
                    # ADX rising, +DI > -DI, Price above moving averages
                    if (today['Close'] > today['SMA10'] > today['SMA20']) and \
                       (today['+DI'] > today['-DI']) and \
                       (today['ADX'] > yesterday['ADX']):
                        passed_strategy = True
                        
                # STRATEGY 2: CHOPPY (Mean Reversion)
                elif regime == "CHOPPY":
                    # Deeply oversold (RSI < 35) but still fundamentally holding above a catastrophic crash
                    if today['RSI'] < 35 and today['Close'] > (today['SMA20'] * 0.85):
                        passed_strategy = True

                if passed_strategy:
                    stats["passed_regime_strategy"] += 1
                    
                    # Backtest logic matches the strategy triggered
                    if regime in ["UPTREND", "DOWNTREND"]:
                        hist = df[(df['Close'] > df['SMA10']) & (df['SMA10'] > df['SMA20']) & (df['+DI'] > df['-DI']) & (df['ADX'] > df['ADX'].shift(1))].index
                    else:
                        hist = df[(df['RSI'] < 35) & (df['Close'] > (df['SMA20'] * 0.85))].index
                        
                    wins, total = 0, 0
                    for d in hist:
                        idx = df.index.get_loc(d)
                        if idx + 3 < len(df):
                            if df.iloc[idx+3]['Close'] > df.iloc[idx]['Close']: wins += 1
                            total += 1
                    
                    if total > 0:
                        win_rate = (wins/total)*100
                        if win_rate >= MIN_WIN_RATE:
                            stats["passed_backtest"] += 1
                            all_hits.append({
                                "ticker": symbol, 
                                "win_rate": round(win_rate, 1), 
                                "price": round(price, 2)
                            })
            except: continue

    res_df = pd.DataFrame(all_hits)
    if not res_df.empty:
        res_df = res_df.sort_values(by="win_rate", ascending=False)

    trade_log, port_html = execute_alpaca_trades(res_df)
    send_email(res_df, trade_log, port_html, time_msg, regime_msg, regime, error_log, stats)

# --- EMAIL HELPER ---
def send_email(res_df, trade_log, port_html, ny_time, regime_msg, regime, error_log, stats):
    msg = EmailMessage()
    hits = len(res_df) if not res_df.empty else 0
    
    subject = f"[{regime}] Sniper Report: {hits} Hits"

    msg['Subject'] = subject
    msg['From'] = os.environ.get('EMAIL_USER')
    msg['To'] = os.environ.get('EMAIL_RECEIVER')
    
    hits_html = res_df.head(20).to_html(index=False) if not res_df.empty else "No hits found."

    body = f"""
    <h3>Run Complete ({ny_time})</h3>
    <p><b>Market Regime:</b> {regime_msg}</p>
    
    <div style="background: #f9f9f9; padding: 15px; border: 1px solid #ddd;">
        <h4 style="margin-top: 0;">Diagnostic Funnel (The Proof)</h4>
        <ul style="margin-bottom: 0;">
            <li><b>Attempted:</b> {stats['total_scanned']} tickers</li>
            <li><b>Valid Downloads:</b> {stats['valid_downloads']}</li>
            <li><b>Passed Volume/RVOL:</b> {stats['passed_volume_filter']}</li>
            <li><b>Passed {regime} Strategy:</b> {stats['passed_regime_strategy']}</li>
            <li><b>Passed Backtest:</b> {stats['passed_backtest']}</li>
        </ul>
    </div>
    
    <hr>
    <h4>Execution Log</h4>
    <pre>{trade_log}</pre>
    <hr>
    <h4>Scanner Results</h4>
    {hits_html}
    <hr>
    <h4>Current Portfolio</h4>
    {port_html}
    """
    msg.add_alternative(body, subtype='html')
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(os.environ.get('EMAIL_USER'), os.environ.get('EMAIL_PASS'))
        smtp.send_message(msg)

if __name__ == "__main__":
    run_main()
