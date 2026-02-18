import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import time
import smtplib
import requests
import random
from datetime import datetime, timedelta
import pytz
from email.message import EmailMessage
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- CONFIGURATION ---
BATCH_SIZE = 50        # Smaller batches = Less blocking
MIN_WIN_RATE = 50.0    # 50% Win Rate Filter
RISK_REWARD = 3.0      # 3:1 Ratio

# --- 0. ANTI-BLOCK SESSION ---
def get_yfinance_session():
    session = requests.Session()
    # Rotating User Agents to evade detection
    agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
    ]
    session.headers.update({'User-Agent': random.choice(agents)})
    return session

# --- 1. THE BOUNCER ---
def is_market_closing_soon():
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # PRODUCTION CHECK
    if now_ny.hour == 15 and 40 <= now_ny.minute <= 59:
        return True, now_ny.strftime("%I:%M %p %Z")
    
    return False, now_ny.strftime("%I:%M %p %Z")

# --- 2. MARKET TIDE (VIA ALPACA) ---
# Switched to Alpaca Data to guarantee we get the Market Status
def get_market_tide():
    try:
        api_key = os.environ.get('ALPACA_API_KEY')
        secret_key = os.environ.get('ALPACA_SECRET_KEY')
        data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Get last 25 days of SPY
        req = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=datetime.now(pytz.utc) - timedelta(days=40)
        )
        bars = data_client.get_stock_bars(req)
        
        if not bars or "SPY" not in bars:
            return True, "⚠️ Tide Check Failed (Alpaca Empty). Assuming Neutral."
            
        spy_data = bars["SPY"]
        closes = [bar.close for bar in spy_data]
        
        if len(closes) < 20:
            return True, "⚠️ Tide Check Failed (Not enough data). Assuming Neutral."
            
        spy_close = closes[-1]
        spy_sma20 = sum(closes[-20:]) / 20
        
        status_msg = f"SPY: {spy_close:.2f} (SMA20: {spy_sma20:.2f})"
        
        # Logic: If SPY > SMA20, Market is Bullish. If not, Bearish/Neutral.
        if spy_close > spy_sma20:
            return True, f"✅ MARKET HEALTHY. {status_msg}"
        return False, f"⛔ MARKET DOWN. {status_msg}"
        
    except Exception as e:
        return True, f"⚠️ Tide Check Error ({e}). Assuming Neutral."

# --- 3. PRO INDICATORS ---
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

# --- 4. EXECUTION ---
def execute_alpaca_trades(winning_df):
    if winning_df.empty: return "No trades.", "<p>No positions.</p>"

    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    client = TradingClient(api_key, secret_key, paper=True) 
    
    client.cancel_orders()
    time.sleep(1)

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
        return f"BLOCKED: PDT Safeguard Active ({account.daytrade_count}/3).", port_html
    
    slot_size = min((equity / len(fresh)), 5000.00)
    log = []

    for _, stock in fresh.head(20).iterrows():
        try:
            qty = int(slot_size / stock['price'])
            if qty > 0:
                req = MarketOrderRequest(
                    symbol=stock['ticker'], qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(stock['price']*1.03, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stock['price']*0.99, 2))
                )
                client.submit_order(req)
                log.append(f"BOUGHT {qty} {stock['ticker']} @ {stock['price']}")
        except Exception as e: log.append(f"Err {stock['ticker']}: {e}")
            
    return "\n".join(log), port_html

# --- 5. MAIN LOGIC ---
def run_main():
    is_time, time_msg = is_market_closing_soon()
    if not is_time:
        print(f"Skipping: {time_msg}. Not in closing window.")
        return

    # ALPACA TIDE CHECK (No Yahoo)
    tide_safe, tide_msg = get_market_tide()
    print(f"Tide: {tide_msg}")
    
    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file): return
    with open(ticker_file, 'r') as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()]

    print(f"Starting Scan for {len(all_tickers)} tickers in batches of {BATCH_SIZE}...")
    session = get_yfinance_session()
    all_hits = []
    error_log = [] # NEW: Tracks failed batches
    
    # --- BATCH PROCESSING LOOP ---
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i+BATCH_SIZE]
        
        batch_data = pd.DataFrame()
        for attempt in range(2):
            try:
                # threads=False for Stealth
                batch_data = yf.download(batch, period="250d", group_by='ticker', threads=False, progress=False, auto_adjust=True, session=session)
                if not batch_data.empty:
                    break
                time.sleep(2)
            except Exception as e:
                time.sleep(2)

        if batch_data.empty:
            print(f"Batch {i} FAILED/BLOCKED.")
            error_log.append(f"Batch {i}-{i+BATCH_SIZE}: FAILED (Empty Data)")
            continue

        for symbol in batch:
            try:
                if len(batch) > 1:
                    if symbol not in batch_data.columns.levels[0]: continue
                    df = batch_data[symbol]
                else:
                    df = batch_data
                
                if df.empty or df['Close'].isna().all(): continue
                df = df.dropna()
                if len(df) < 50: continue
                
                price = df['Close'].iloc[-1]
                avg_vol = df['Volume'].iloc[-21:-1].mean()
                
                if price < 1.0 or avg_vol < 300000: continue
                if (df['Volume'].iloc[-1] / avg_vol) < 1.2: continue 
                
                df = calculate_indicators(df)
                today = df.iloc[-1]
                yesterday = df.iloc[-2]
                
                # Market Neutral Logic (Works for Bull & Bear tickers)
                if (today['Close'] > today['SMA10'] > today['SMA20']) and (today['ADX'] > yesterday['ADX']):
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
                            all_hits.append({
                                "ticker": symbol, 
                                "win_rate": round(win_rate, 1), 
                                "price": round(price, 2)
                            })
            except: continue
        
        time.sleep(2)

    res_df = pd.DataFrame(all_hits)
    if not res_df.empty:
        res_df = res_df.sort_values(by="win_rate", ascending=False)

    trade_log, port_html = execute_alpaca_trades(res_df)
    send_email(res_df, trade_log, port_html, time_msg, tide_msg, tide_safe, error_log)

# --- EMAIL HELPER ---
def send_email(res_df, trade_log, port_html, ny_time, tide_msg, tide_safe, error_log):
    msg = EmailMessage()
    hits = len(res_df) if not res_df.empty else 0
    
    subject = f"Sniper Report: {hits} Hits"
    if not tide_safe: subject = f"📉 Bear Scan: {hits} Hits"

    msg['Subject'] = subject
    msg['From'] = os.environ.get('EMAIL_USER')
    msg['To'] = os.environ.get('EMAIL_RECEIVER')
    
    hits_html = res_df.head(20).to_html(index=False) if not res_df.empty else "No hits found."
    
    # NEW: Error Log Display
    errors_html = ""
    if error_log:
        errors_html = f"<hr><h4 style='color:red'>Failed Batches ({len(error_log)})</h4><pre>{chr(10).join(error_log[:10])}</pre>"

    body = f"""
    <h3>Run Complete ({ny_time})</h3>
    <p><b>Market Tide (Alpaca):</b> {tide_msg}</p>
    <p><i>Note: Inverse ETFs active if market is down.</i></p>
    <hr>
    <h4>Execution Log</h4>
    <pre>{trade_log}</pre>
    <hr>
    <h4>Scanner Results</h4>
    {hits_html}
    {errors_html}
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
