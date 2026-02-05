import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import smtplib
from datetime import datetime
import pytz
from email.message import EmailMessage
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# --- 1. THE BOUNCER (Time Gate) ---
def is_market_closing_soon():
    """Checks if current NY time is strictly between 3:40 PM and 3:59 PM."""
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # Check for the 3:40 PM - 3:59 PM window
    if now_ny.hour == 15 and 40 <= now_ny.minute <= 59:
        return True, now_ny.strftime("%I:%M %p %Z")
        
    return False, now_ny.strftime("%I:%M %p %Z")

# --- 2. TECHNICAL INDICATORS ---
def calculate_indicators(df):
    df = df.copy()
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['TR'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    
    period = 14
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    
    # ADX Calculation
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    df['+DI'] = 100 * (df['+DM'].rolling(period).mean() / df['TR'].rolling(period).mean())
    df['-DI'] = 100 * (df['-DM'].rolling(period).mean() / df['TR'].rolling(period).mean())
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(period).mean()
    return df

# --- 3. EXECUTION ENGINE ---
def execute_alpaca_trades(winning_df):
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        raise ValueError("Alpaca API credentials missing from Environment Variables.")
    
    # IMPORTANT: paper=True is set here
    client = TradingClient(api_key, secret_key, paper=True) 
    
    # Maintenance: Clear yesterday's leftover orders
    client.cancel_orders()
    time.sleep(1)

    account = client.get_account()
    equity = float(account.equity)
    
    # PDT Safeguard ($30k threshold)
    if equity < 30000:
        if int(account.daytrade_count) >= 2:
            return f"BLOCKED: PDT Safeguard Active. Trades used: {account.daytrade_count}/3", ""

    # Portfolio Check (Prevent Double Buying)
    positions = client.get_all_positions()
    existing_tickers = [p.symbol for p in positions]
    
    portfolio_list = []
    for p in positions:
        portfolio_list.append({
            "Symbol": p.symbol, "Qty": p.qty, "Price": round(float(p.current_price), 2),
            "P/L $": round(float(p.unrealized_pl), 2), "P/L %": f"{float(p.unrealized_pl_pc)*100:.2f}%"
        })
    portfolio_html = pd.DataFrame(portfolio_list).to_html(index=False) if portfolio_list else "<p>No open positions.</p>"

    # Sizing Logic
    MAX_SETUPS = 20
    MAX_CASH = 5000.00
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

# --- 4. EMAILER ---
def send_email(res_df, trade_log, port_html, ny_time):
    msg = EmailMessage()
    user = os.environ.get('EMAIL_USER')
    password = os.environ.get('EMAIL_PASS')
    receiver = os.environ.get('EMAIL_RECEIVER')
    
    if not user or not password or not receiver:
         raise ValueError("Email credentials missing from Environment Variables.")

    subject = f"Sniper Report: {len(res_df)} Hits" if not res_df.empty else "Sniper Report: No Trades"
    
    body = f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2 style="color: #2E86C1;">Sniper Command Center - {ny_time}</h2>
    
    <div style="background: #f4f4f4; padding: 10px; border: 1px solid #ddd;">
        <h3 style="margin-top:0;">Execution Log</h3>
        <pre>{trade_log}</pre>
    </div>
    
    <h3>Current Portfolio</h3>
    {port_html}
    
    <h3>Scanner Hits (New Only)</h3>
    {res_df.to_html(index=False) if not res_df.empty else "No new setups found."}
    
    <p style="font-size: 10px; color: gray;">Settings: 3:1 RR | 55% Win Rate | 3:45 PM Check</p>
    </body></html>
    """
    msg.add_alternative(body, subtype='html')
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = receiver
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)

# --- 5. MAIN LOGIC (UPDATED FOR SAFETY) ---
def run_main():
    # 1. Check Time
    is_time, ny_time = is_market_closing_soon()
    # UNCOMMENT THE NEXT TWO LINES TO ENFORCE TIME GATE
    if not is_time:
        print(f"Skipping: Time is {ny_time}. Not in closing window (3:40-3:59 PM).")
        return

    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file):
        print("ERROR: tickers.txt file not found.")
        return
        
    with open(ticker_file, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    print(f"Executing Sniper Scan at {ny_time}...")
    
    # 2. Robust Download
    try:
        data = yf.download(tickers, period="250d", group_by='ticker', threads=True, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"CRITICAL DOWNLOAD ERROR: {e}")
        return

    hits = []
    
    # 3. Process Tickers
    print(f"Scanning {len(tickers)} tickers...")
    for i, symbol in enumerate(tickers):
        try:
            # Handle MultiIndex logic
            if len(tickers) > 1:
                df = data[symbol]
            else:
                df = data

            if df.empty or df['Close'].isna().all(): continue
            
            df = df.dropna()
            if len(df) < 50: continue
            
            price = df['Close'].iloc[-1]
            avg_vol = df['Volume'].iloc[-21:-1].mean()
            
            # Filter 1: Price & Volume
            if price < 1.0 or (df['Volume'].iloc[-1] / avg_vol) < 1.5 or avg_vol < 300000: continue
            
            # Filter 2: Technicals
            df = calculate_indicators(df)
            today, yesterday = df.iloc[-1], df.iloc[-2]
            
            if (today['Close'] > today['SMA10'] > today['SMA20']) and (today['ADX'] > yesterday['ADX']):
                
                # Filter 3: REAL BACKTEST
                hist = df[(df['Close'] > df['SMA10']) & (df['SMA10'] > df['SMA20']) & (df['ADX'] > df['ADX'].shift(1))].index
                wins, total = 0, 0
                
                for d in hist:
                    idx = df.index.get_loc(d)
                    if idx + 3 < len(df):
                        if df.iloc[idx+3]['Close'] > df.iloc[idx]['Close']: 
                            wins += 1
                        total += 1
                
                if total > 0:
                    real_win_rate = (wins / total) * 100
                    if real_win_rate >= 55:
                        hits.append({
                            "ticker": symbol, 
                            "win_rate": round(real_win_rate, 1), 
                            "price": round(price, 2)
                        })
        except Exception:
            continue

    res_df = pd.DataFrame(hits)
    print(f"Scan complete. Found {len(res_df)} hits.")
    
    # 4. Execute & Report
    # We use try/except blocks here so one failure doesn't crash the whole script
    trade_log = "Trade Execution Skipped (Error)"
    port_html = "<p>Portfolio Data Unavailable</p>"

    try:
        print("Attempting to execute trades...")
        trade_log, port_html = execute_alpaca_trades(res_df)
        print("Trades processed successfully.")
    except Exception as e:
        print(f"CRITICAL ALPACA ERROR: {e}")
        trade_log = f"Alpaca Failed: {e}"

    try:
        print("Attempting to send email...")
        send_email(res_df, trade_log, port_html, ny_time)
        print("Email sent successfully.")
    except Exception as e:
        print(f"CRITICAL EMAIL ERROR: {e}")

if __name__ == "__main__":
    run_main()

