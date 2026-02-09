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
BATCH_SIZE = 100       # Download 100 stocks at a time
MIN_WIN_RATE = 50.0    # Minimum backtested win rate to accept
RISK_REWARD = 3.0      # 3:1 Ratio

# --- 0. ANTI-BLOCK SESSION ---
def get_yfinance_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- 1. THE BOUNCER (PRODUCTION MODE) ---
def is_market_closing_soon():
    """
    Returns True ONLY if NY Time is 3:40 PM - 3:59 PM.
    Prevents accidental trading at night or morning.
    """
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # PRODUCTION CHECK
    if now_ny.hour == 15 and 40 <= now_ny.minute <= 59:
        return True, now_ny.strftime("%I:%M %p %Z")
    
    # If not time, return False and the reason
    return False, now_ny.strftime("%I:%M %p %Z")

# --- 2. MARKET TIDE ---
def get_market_tide():
    session = get_yfinance_session()
    try:
        # Download SPY and VIX
        data = yf.download(["SPY", "^VIX"], period="50d", group_by='ticker', progress=False, auto_adjust=True, session=session)
        if data.empty: return True, "⚠️ Tide Check Failed (Empty). Assuming Safe."
        
        spy_close = data["SPY"]['Close'].iloc[-1]
        spy_sma20 = data["SPY"]['Close'].rolling(20).mean().iloc[-1]
        vix_close = data["^VIX"]['Close'].iloc[-1]
        
        status_msg = f"SPY: {spy_close:.2f} (SMA20: {spy_sma20:.2f}) | VIX: {vix_close:.2f}"
        
        # LOGIC: SPY > SMA20 AND VIX < 32
        if (spy_close > spy_sma20) and (vix_close < 32):
            return True, f"✅ MARKET HEALTHY. {status_msg}"
        return False, f"⛔ MARKET UNSAFE. {status_msg}"
            
    except Exception as e:
        return True, f"⚠️ Tide Check Error ({e}). Assuming Safe."

# --- 3. PRO INDICATORS ---
def calculate_indicators(df):
    df = df.copy()
    try:
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        
        # Normalize columns
        if 'ADX_14' in df.columns: df['ADX'] = df['ADX_14']
        else: df['ADX'] = 0
        if 'SMA_10' in df.columns: df['SMA10'] = df['SMA_10']
        else: df['SMA10'] = 0
        if 'SMA_20' in df.columns: df['SMA20'] = df['SMA_20']
        else: df['SMA20'] = 0
    except:
        return df
    return df

# --- 4. EXECUTION ---
def execute_alpaca_trades(winning_df):
    if winning_df.empty: return "No trades (Empty Data).", "<p>No positions.</p>"

    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    client = TradingClient(api_key, secret_key, paper=True) 
    
    # Cancel old orders first
    client.cancel_orders()
    time.sleep(1)

    # Portfolio Check
    positions = client.get_all_positions()
    existing = [p.symbol for p in positions]
    
    port_list = [{"Symbol": p.symbol, "P/L": f"${float(p.unrealized_pl):.2f}"} for p in positions]
    port_html = pd.DataFrame(port_list).to_html(index=False) if port_list else "<p>No positions.</p>"

    if 'ticker' not in winning_df.columns: return "Error: Missing ticker col", port_html
    
    fresh = winning_df[~winning_df['ticker'].isin(existing)]
    if fresh.empty: return "No new setups.", port_html

    # EXECUTION LOOP
    account = client.get_account()
    equity = float(account.equity)
    
    # PDT Check
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
    # 1. TIME CHECK
    is_time, time_msg = is_market_closing_soon()
    if not is_time:
        # Logs "Skipping: 09:00 PM EST..." to GitHub console
        print(f"Skipping: Time is {time_msg}. Not in closing window (3:40-3:59 PM).")
        return

    # 2. TIDE CHECK
    tide_safe, tide_msg = get_market_tide()
    print(f"Tide: {tide_msg}")
    
    if not tide_safe:
        # If unsafe, sends email WARNING and exits (saves time)
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
        
        try:
            data = yf.download(batch, period="250d", group_by='ticker', threads=True, progress=False, auto_adjust=True, session=session)
            
            for symbol in batch:
                try:
                    # Robust Data Access
                    if len(batch) > 1:
                        if symbol not in data.columns.levels[0]: continue
                        df = data[symbol]
                    else:
                        df = data
                    
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
                    
                    # Strategy: Uptrend + Momentum + Backtest
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
            time.sleep(0.5) # Throttle
        except Exception as e:
            print(f"Batch {i} Error: {e}")
            continue

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
