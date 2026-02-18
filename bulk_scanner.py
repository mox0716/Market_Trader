import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import time
import smtplib
import requests
from datetime import datetime
import pytz
from email.message import EmailMessage
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# --- CONFIGURATION ---
BATCH_SIZE = 50        # Reduced batch size to stay under radar
MIN_WIN_RATE = 50.0    # 50% Win Rate Filter
RISK_REWARD = 3.0      # 3:1 Ratio

# --- 0. ANTI-BLOCK SESSION ---
def get_yfinance_session():
    session = requests.Session()
    # Updated User-Agent to look more recent
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    return session

# --- 1. THE BOUNCER ---
def is_market_closing_soon():
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # PRODUCTION CHECK (3:40 PM - 3:59 PM ET)
    if now_ny.hour == 15 and 40 <= now_ny.minute <= 59:
        return True, now_ny.strftime("%I:%M %p %Z")
    
    return False, now_ny.strftime("%I:%M %p %Z")

# --- 2. MARKET TIDE (With Retry) ---
def get_market_tide():
    session = get_yfinance_session()
    
    # Try twice to get Tide Data
    for attempt in range(2):
        try:
            # threads=False is key here
            data = yf.download(["SPY", "^VIX"], period="50d", group_by='ticker', threads=False, progress=False, auto_adjust=True, session=session)
            
            if data.empty:
                print(f"Tide Attempt {attempt+1} Empty. Waiting...")
                time.sleep(5)
                continue
            
            spy_close = data["SPY"]['Close'].iloc[-1]
            spy_sma20 = data["SPY"]['Close'].rolling(20).mean().iloc[-1]
            vix_close = data["^VIX"]['Close'].iloc[-1]
            
            status_msg = f"SPY: {spy_close:.2f} (SMA20: {spy_sma20:.2f}) | VIX: {vix_close:.2f}"
            
            if (spy_close > spy_sma20) and (vix_close < 32):
                return True, f"✅ MARKET HEALTHY. {status_msg}"
            return False, f"⛔ MARKET UNSAFE. {status_msg}"
                
        except Exception as e:
            print(f"Tide Error: {e}")
            time.sleep(2)
            
    return True, "⚠️ Tide Check Failed (Data Blocked). Assuming Safe."

# --- 3. PRO INDICATORS ---
def calculate_indicators(df):
    df = df.copy()
    try:
        # Explicit error handling for pandas_ta
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        
        # Mapping columns safely
        df['ADX'] = df['ADX_14'] if 'ADX_14' in df.columns else 0
        df['SMA10'] = df['SMA_10'] if 'SMA_10' in df.columns else 0
        df['SMA20'] = df['SMA_20'] if 'SMA_20' in df.columns else 0
    except Exception as e:
        # Print error to logs so we know if TA is failing
        print(f"TA Error: {e}")
        return df
    return df

# --- 4. EXECUTION ---
def execute_alpaca_trades(winning_df):
    if winning_df.empty: return "No trades (Empty Data).", "<p>No positions.</p>"

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

    tide_safe, tide_msg = get_market_tide()
    print(f"Tide: {tide_msg}")
    
    if not tide_safe:
        send_email(pd.DataFrame(), "Trade Blocked: Market Unsafe", "<p>No Trades.</p>", time_msg, tide_msg)
        return

    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file): return
    with open(ticker_file, 'r') as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()]

    print(f"Starting Scan for {len(all_tickers)} tickers in batches of {BATCH_SIZE}...")
    session = get_yfinance_session()
    all_hits = []
    
    # --- BATCH PROCESSING LOOP ---
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i+BATCH_SIZE]
        print(f"Processing Batch {i} to {i+len(batch)}...")
        
        # RETRY LOOP FOR EACH BATCH
        batch_data = pd.DataFrame()
        for attempt in range(2):
            try:
                # threads=False prevents blocking
                batch_data = yf.download(batch, period="250d", group_by='ticker', threads=False, progress=False, auto_adjust=True, session=session)
                if not batch_data.empty:
                    break
                print(f"Batch {i} Empty on attempt {attempt+1}...")
                time.sleep(3)
            except Exception as e:
                print(f"Batch {i} Download Error: {e}")
                time.sleep(3)

        if batch_data.empty:
            print(f"CRITICAL: Batch {i} FAILED/BLOCKED. Skipping.")
            continue

        # Process the valid data
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
        
        # Increase sleep to 2 seconds to act "Human"
        time.sleep(2)

    res_df = pd.DataFrame(all_hits)
    if not res_df.empty:
        res_df = res_df.sort_values(by="win_rate", ascending=False)

    trade_log, port_html = execute_alpaca_trades(res_df)
    send_email(res_df, trade_log, port_html, time_msg, tide_msg)

# --- EMAIL HELPER ---
def send_email(res_df, trade_log, port_html, ny_time, tide_msg):
    msg = EmailMessage()
    hits = len(res_df) if not res_df.empty else 0
    msg['Subject'] = f"Sniper Report: {hits} Hits Found"
    msg['From'] = os.environ.get('EMAIL_USER')
    msg['To'] = os.environ.get('EMAIL_RECEIVER')
    
    hits_html = res_df.head(20).to_html(index=False) if not res_df.empty else "No hits found."

    body = f"""
    <h3>Sniper Run Complete ({ny_time})</h3>
    <p><b>Market Tide:</b> {tide_msg}</p>
    <hr>
    <h4>Execution Log</h4>
    <pre>{trade_log}</pre>
    <hr>
    <h4>Scanner Results (Top 20)</h4>
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
