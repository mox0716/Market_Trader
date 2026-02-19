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
# UPDATED: Imported LimitOrderRequest
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- CONFIGURATION ---
BATCH_SIZE = 100       
MIN_WIN_RATE = 50.0    
RISK_REWARD = 3.0      

# --- 1. THE BOUNCER ---
def is_market_closing_soon():
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # Target execution time: Exactly 3:45:00 PM NY Time
    target_time = now_ny.replace(hour=15, minute=45, second=0, microsecond=0)
    cutoff_time = now_ny.replace(hour=15, minute=59, second=59, microsecond=0)

    # 1. Too late? (After 4:00 PM)
    if now_ny > cutoff_time:
        return False, f"Too late. Market closed. ({now_ny.strftime('%I:%M %p')})"

    # 2. Too early? (Before 3:45 PM)
    if now_ny < target_time:
        sleep_seconds = (target_time - now_ny).total_seconds()
        
        # DST Collision Preventer
        if sleep_seconds > 3000:
            return False, "Too early (Wrong DST schedule). Exiting silently."
            
        print(f"⏰ GitHub started early. Sleeping for {sleep_seconds/60:.1f} minutes until exactly 3:45 PM NY Time...")
        time.sleep(sleep_seconds)
        now_ny = datetime.now(tz_ny)

    # 3. Exactly on time
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

# --- 3. MARKET TIDE ---
def get_market_tide():
    df = get_alpaca_data(["SPY", "QQQ"], days_back=40)
    if df.empty or "SPY" not in df.index.levels[0]:
        return True, "⚠️ Tide Check Failed. Assuming Neutral."
    try:
        spy_df = df.loc["SPY"]
        spy_close = spy_df['close'].iloc[-1]
        spy_sma20 = spy_df['close'].rolling(20).mean().iloc[-1]
        
        status_msg = f"SPY: ${spy_close:.2f} (SMA20: ${spy_sma20:.2f})"
        if spy_close > spy_sma20:
            return True, f"✅ MARKET HEALTHY. {status_msg}"
        return False, f"⛔ MARKET DOWN. {status_msg}"
    except Exception as e:
        return True, f"⚠️ Tide Check Error. Assuming Neutral."

# --- 4. PRO INDICATORS ---
def calculate_indicators(df):
    df = df.copy()
    try:
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        
        df['ADX'] = df['ADX_14'] if 'ADX_14' in df.columns else 0
        df['SMA10'] = df['SMA_10'] if 'SMA_10' in df.columns else 0
        df['SMA20'] = df['SMA_20'] if 'SMA_20' in df.columns else 0
    except:
        return df
    return df

# --- 5. EXECUTION ---
def execute_alpaca_trades(winning_df):
    if winning_df.empty: return "No trades.", "<p>No positions.</p>"

    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    client = TradingClient(api_key, secret_key, paper=True) 
    
    positions = client.get_all_positions()
    existing = [p.symbol for p in positions]
    port_list = [{"Symbol": p.symbol, "P/L": f"${float(p.unrealized_pl):.2f}"} for p in positions]
    port_html = pd.DataFrame(port_list).to_html(index=False) if port_list else "<p>No positions.</p>"

    if 'ticker' not in winning_df.columns: return "Error: Missing ticker col", port_html
    
    fresh = winning_df[~winning_df['ticker'].isin(existing)]
    if fresh.empty: return "No new setups.", port_html

    account = client.get_account()
    equity = float(account.equity)
    
    if equity < 30000 and int(account.daytrade_count) >= 2:
        return f"BLOCKED: PDT Active.", port_html
    
    slot_size = min((equity / len(fresh)), 5000.00)
    log = []

    for _, stock in fresh.head(20).iterrows():
        try:
            qty = int(slot_size / stock['price'])
            if qty > 0:
                # UPDATED: Limit Order Request with 4.5% Profit and 1.5% Stop
                req = LimitOrderRequest(
                    symbol=stock['ticker'], 
                    qty=qty, 
                    limit_price=round(stock['price'], 2), # Exact price from scanner
                    side=OrderSide.BUY, 
                    time_in_force=TimeInForce.DAY,        # Expires at 4:00 PM if unfilled
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(stock['price'] * 1.045, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stock['price'] * 0.985, 2))
                )
                client.submit_order(req)
                log.append(f"PLACED LIMIT BUY {qty} {stock['ticker']} @ ${stock['price']}")
        except Exception as e: log.append(f"Err {stock['ticker']}: {e}")
            
    return "\n".join(log), port_html

# --- 6. MAIN LOGIC ---
def run_main():
    is_time, time_msg = is_market_closing_soon()
    if not is_time:
        print(f"Skipping: {time_msg}")
        return

    tide_safe, tide_msg = get_market_tide()
    
    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file): return
    with open(ticker_file, 'r') as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()]

    all_hits = []
    error_log = []
    
    stats = {
        "total_scanned": len(all_tickers),
        "valid_downloads": 0,
        "passed_volume_filter": 0,
        "passed_trend_filter": 0,
        "passed_backtest": 0
    }
    
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i+BATCH_SIZE]
        batch_data = get_alpaca_data(batch, days_back=365)
        
        if batch_data.empty:
            error_log.append(f"Batch {i}: FAILED")
            continue

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
                
                # Volume/Price Filter 
                if price < 1.0 or avg_vol < 100000: continue
                
                stats["passed_volume_filter"] += 1
                
                df = calculate_indicators(df)
                today = df.iloc[-1]
                yesterday = df.iloc[-2]
                
                # Trend Filter
                if (today['Close'] > today['SMA10'] > today['SMA20']) and (today['ADX'] > yesterday['ADX']):
                    stats["passed_trend_filter"] += 1
                    
                    hist = df[(df['Close'] > df['SMA10']) & (df['SMA10'] > df['SMA20']) & (df['ADX'] > df['ADX'].shift(1))].index
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
    send_email(res_df, trade_log, port_html, time_msg, tide_msg, tide_safe, error_log, stats)

# --- EMAIL HELPER ---
def send_email(res_df, trade_log, port_html, ny_time, tide_msg, tide_safe, error_log, stats):
    msg = EmailMessage()
    hits = len(res_df) if not res_df.empty else 0
    
    subject = f"Sniper Report: {hits} Hits"
    if not tide_safe: subject = f"📉 Bear Scan: {hits} Hits"

    msg['Subject'] = subject
    msg['From'] = os.environ.get('EMAIL_USER')
    msg['To'] = os.environ.get('EMAIL_RECEIVER')
    
    hits_html = res_df.head(20).to_html(index=False) if not res_df.empty else "No hits found."

    body = f"""
    <h3>Run Complete ({ny_time})</h3>
    <p><b>Market Tide:</b> {tide_msg}</p>
    
    <div style="background: #f9f9f9; padding: 15px; border: 1px solid #ddd;">
        <h4 style="margin-top: 0;">Diagnostic Funnel (The Proof)</h4>
        <ul style="margin-bottom: 0;">
            <li><b>Attempted:</b> {stats['total_scanned']} tickers</li>
            <li><b>Valid Downloads:</b> {stats['valid_downloads']}</li>
            <li><b>Passed Volume Check:</b> {stats['passed_volume_filter']}</li>
            <li><b>Passed Trend Check:</b> {stats['passed_trend_filter']}</li>
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
