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

# --- 0. ANTI-BLOCK SESSION ---
def get_yfinance_session():
    """Creates a session that looks like a real Chrome browser to bypass blocks."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- 1. THE BOUNCER (Time Gate) ---
def is_market_closing_soon():
    """Checks if current NY time is strictly between 3:40 PM and 3:59 PM."""
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # 3:40 PM - 3:59 PM window
    if now_ny.hour == 15 and 40 <= now_ny.minute <= 59:
        return True, now_ny.strftime("%I:%M %p %Z")
        
    return False, now_ny.strftime("%I:%M %p %Z")

# --- 2. MARKET TIDE (SPY + VIX) ---
def get_market_tide():
    session = get_yfinance_session()
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Download SPY and VIX with browser session
            data = yf.download(["SPY", "^VIX"], period="50d", group_by='ticker', progress=False, auto_adjust=True, session=session)
            
            if data.empty: raise ValueError("Empty Data")

            # 1. SPY Check (Must be above SMA 20)
            spy_df = data["SPY"]
            spy_close = spy_df['Close'].iloc[-1]
            spy_sma20 = spy_df['Close'].rolling(20).mean().iloc[-1]
            spy_ok = spy_close > spy_sma20
            
            # 2. VIX Check (Must be below 32)
            vix_df = data["^VIX"]
            vix_close = vix_df['Close'].iloc[-1]
            vix_ok = vix_close < 32
            
            status_msg = f"SPY: {spy_close:.2f} (SMA20: {spy_sma20:.2f}) | VIX: {vix_close:.2f}"
            
            if spy_ok and vix_ok:
                return True, f"✅ MARKET HEALTHY. {status_msg}"
            else:
                return False, f"⛔ MARKET UNSAFE. {status_msg}"
                
        except Exception as e:
            print(f"Tide Check Retry {attempt+1}/{max_retries}: {e}")
            time.sleep(5)
            
    return True, "⚠️ Tide Check Failed (Rate Limited). Assuming Safe."

# --- 3. PRO INDICATORS ---
def calculate_indicators(df):
    df = df.copy()
    # Use pandas_ta for robust calculations
    try:
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        
        # Normalize column names if needed
        if 'ADX_14' in df.columns: df['ADX'] = df['ADX_14']
        else: df['ADX'] = 0
            
        if 'SMA_10' in df.columns: df['SMA10'] = df['SMA_10']
        else: df['SMA10'] = 0
            
        if 'SMA_20' in df.columns: df['SMA20'] = df['SMA_20']
        else: df['SMA20'] = 0
    except:
        return df # Return original if TA fails
        
    return df

# --- 4. EXECUTION ENGINE ---
def execute_alpaca_trades(winning_df):
    # CRASH FIX: If dataframe is empty, exit immediately
    if winning_df.empty:
        return "No trades to execute (Empty Data).", "<p>No open positions.</p>"

    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    
    client = TradingClient(api_key, secret_key, paper=True) 
    client.cancel_orders()
    time.sleep(1)

    account = client.get_account()
    equity = float(account.equity)
    
    if equity < 30000:
        if int(account.daytrade_count) >= 2:
            return f"BLOCKED: PDT Safeguard Active. Trades used: {account.daytrade_count}/3", ""

    positions = client.get_all_positions()
    existing_tickers = [p.symbol for p in positions]
    
    portfolio_list = []
    for p in positions:
        portfolio_list.append({
            "Symbol": p.symbol, "Qty": p.qty, "Price": round(float(p.current_price), 2),
            "P/L $": round(float(p.unrealized_pl), 2), "P/L %": f"{float(p.unrealized_pl_pc)*100:.2f}%"
        })
    portfolio_html = pd.DataFrame(portfolio_list).to_html(index=False) if portfolio_list else "<p>No open positions.</p>"

    MAX_SETUPS = 20
    MAX_CASH = 5000.00
    
    # Filter out existing (Crash fix: ensure 'ticker' column exists)
    if 'ticker' not in winning_df.columns:
        return "Error: DataFrame missing ticker column.", portfolio_html

    fresh_setups = winning_df[~winning_df['ticker'].isin(existing_tickers)]
    num_setups = min(len(fresh_setups), MAX_SETUPS)
    
    if num_setups == 0:
        return "No new setups found or already owned.", portfolio_html
    
    slot_size = min((equity / num_setups), MAX_CASH)
    trade_log = []

    for _, stock in fresh_setups.sort_values(by="win_rate", ascending=False).head(MAX_SETUPS).iterrows():
        try:
            qty = int(slot_size / stock['price'])
            if qty > 0:
                order = MarketOrderRequest(
                    symbol=stock['ticker'], qty=qty, side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(stock['price'] * 1.03, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stock['price'] * 0.99, 2))
                )
                client.submit_order(order)
                trade_log.append(f"BOUGHT {qty} {stock['ticker']} @ ${stock['price']}")
        except Exception as e:
            trade_log.append(f"Error buying {stock['ticker']}: {e}")

    return "\n".join(trade_log) if trade_log else "No orders placed.", portfolio_html

# --- 5. EMAILER ---
def send_email(res_df, trade_log, port_html, ny_time, tide_msg):
    msg = EmailMessage()
    user = os.environ.get('EMAIL_USER')
    
    hits_count = len(res_df) if not res_df.empty else 0
    subject = f"Sniper Report: {hits_count} Hits"
    if "MARKET UNSAFE" in tide_msg: subject = "⛔ Sniper Report: MARKET UNSAFE"
    
    # Handle empty dataframe for HTML conversion
    hits_html = res_df.to_html(index=False) if not res_df.empty else "No new setups found."

    body = f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2 style="color: #2E86C1;">Sniper Command Center - {ny_time}</h2>
    
    <div style="background: #E8F8F5; padding: 10px; border: 1px solid #117864; margin-bottom: 20px;">
        <strong>MARKET TIDE:</strong> {tide_msg}
    </div>
    
    <div style="background: #f4f4f4; padding: 10px; border: 1px solid #ddd;">
        <h3 style="margin-top:0;">Execution Log</h3>
        <pre>{trade_log}</pre>
    </div>
    
    <h3>Current Portfolio</h3>
    {port_html}
    
    <h3>Scanner Hits (New Only)</h3>
    {hits_html}
    
    <p style="font-size: 10px; color: gray;">Settings: Anti-Block Session | VIX Filter | 3:45 PM Check</p>
    </body></html>
    """
    msg.add_alternative(body, subtype='html')
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = os.environ.get('EMAIL_RECEIVER')
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(user, os.environ.get('EMAIL_PASS'))
        smtp.send_message(msg)

# --- 6. MAIN LOGIC ---
def run_main():
    is_time, ny_time = is_market_closing_soon()
    if not is_time:
        print(f"Skipping: Time is {ny_time}. Not in closing window (3:40-3:59 PM).")
        return

    # 1. TIDE CHECK (With Retry)
    tide_safe, tide_msg = get_market_tide()
    print(f"Tide Check: {tide_msg}")
    
    if not tide_safe:
        send_email(pd.DataFrame(), "Trade Blocked: Market Unsafe", "<p>No Trades.</p>", ny_time, tide_msg)
        return

    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file): return
    with open(ticker_file, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    print(f"Executing Sniper Scan at {ny_time}...")
    
    # 2. DOWNLOAD (With Retry & User-Agent)
    session = get_yfinance_session()
    data = pd.DataFrame()
    
    # Retry loop for main download
    for attempt in range(3):
        try:
            print(f"Download Attempt {attempt+1}...")
            data = yf.download(tickers, period="250d", group_by='ticker', threads=True, progress=False, auto_adjust=True, session=session)
            if not data.empty: break
        except Exception as e:
            print(f"Rate Limit/Error: {e}. Waiting 20s...")
            time.sleep(20)
    
    hits = []
    
    # 3. PROCESS DATA
    if not data.empty:
        for i, symbol in enumerate(tickers):
            try:
                if len(tickers) > 1: df = data[symbol]
                else: df = data

                if df.empty or df['Close'].isna().all(): continue
                
                df = df.dropna()
                if len(df) < 50: continue
                
                price = df['Close'].iloc[-1]
                avg_vol = df['Volume'].iloc[-21:-1].mean()
                
                if price < 1.0 or (df['Volume'].iloc[-1] / avg_vol) < 1.5 or avg_vol < 300000: continue
                
                df = calculate_indicators(df)
                today, yesterday = df.iloc[-1], df.iloc[-2]
                
                if (today['Close'] > today['SMA10'] > today['SMA20']) and (today['ADX'] > yesterday['ADX']):
                    hist = df[(df['Close'] > df['SMA10']) & (df['SMA10'] > df['SMA20']) & (df['ADX'] > df['ADX'].shift(1))].index
                    wins, total = 0, 0
                    for d in hist:
                        idx = df.index.get_loc(d)
                        if idx + 3 < len(df):
                            if df.iloc[idx+3]['Close'] > df.iloc[idx]['Close']: wins += 1
                            total += 1
                    
                    if total > 0:
                        real_win_rate = (wins / total) * 100
                        if real_win_rate >= 55:
                            hits.append({
                                "ticker": symbol, 
                                "win_rate": round(real_win_rate, 1), 
                                "price": round(price, 2)
                            })
            except: continue
    else:
        print("CRITICAL: Failed to download data after retries.")

    res_df = pd.DataFrame(hits)
    
    # 4. EXECUTE & REPORT
    trade_log, port_html = execute_alpaca_trades(res_df)
    send_email(res_df, trade_log, port_html, ny_time, tide_msg)

if __name__ == "__main__":
    run_main()
