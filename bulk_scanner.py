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
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

# --- 1. TIME GATE (DST AUTOMATION) ---
def is_market_closing_soon():
    """Checks if current NY time is 3:40 PM - 3:59 PM to execute trades."""
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    # 15 is 3:00 PM. We target the final 20 minutes of the session.
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

# --- 3. TRADING & PORTFOLIO ENGINE ---
def execute_alpaca_trades(winning_df):
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    client = TradingClient(api_key, secret_key, paper=True) # SET TO False FOR LIVE
    
    # MAINTENANCE: Cancel unfulfilled buy orders from previous runs
    client.cancel_orders()
    time.sleep(1)

    account = client.get_account()
    equity = float(account.equity)
    
    # PDT SAFEGUARD ($30k Cushion)
    if equity < 30000:
        if int(account.daytrade_count) >= 2:
            return f"BLOCKED: PDT Safeguard (Trades used: {account.daytrade_count})", ""

    # PORTFOLIO DATA (Prevents Double-Buying)
    positions = client.get_all_positions()
    existing_tickers = [p.symbol for p in positions]
    
    portfolio_list = []
    for p in positions:
        portfolio_list.append({
            "Symbol": p.symbol, "Qty": p.qty, "Price": round(float(p.current_price), 2),
            "P/L $": round(float(p.unrealized_pl), 2), "P/L %": f"{float(p.unrealized_pl_pc)*100:.2f}%"
        })
    portfolio_html = pd.DataFrame(portfolio_list).to_html(index=False) if portfolio_list else "None"

    # POSITION SIZING
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
                trade_log.append(f"Bought {qty} {stock['ticker']} @ {stock['price']}")
        except Exception as e:
            trade_log.append(f"Error {stock['ticker']}: {e}")

    return "\n".join(trade_log) if trade_log else "No orders placed.", portfolio_html

# --- 4. EMAIL ENGINE ---
def send_email(res_df, trade_log, port_html, ny_time):
    msg = EmailMessage()
    user = os.environ.get('EMAIL_USER')
    body = f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2 style="color: #2E86C1;">Sniper Command Center - {ny_time}</h2>
    <div style="background: #f4f4f4; padding: 10px; border: 1px solid #ddd;">
        <h3>Execution Summary</h3>
        <pre>{trade_log}</pre>
    </div>
    <h3>Current Open Positions</h3>
    {port_html}
    <h3>Scanner Hits (New Only)</h3>
    {res_df.to_html(index=False) if not res_df.empty else "No new setups found."}
    <p style="font-size: 10px; color: gray;">3:1 RR | NY Closing Window Check Active</p>
    </body></html>
    """
    msg.add_alternative(body, subtype='html')
    msg['Subject'] = f"Sniper Report: {ny_time}"
    msg['From'] = user
    msg['To'] = os.environ.get('EMAIL_RECEIVER')
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(user, os.environ.get('EMAIL_PASS'))
        smtp.send_message(msg)

# --- 5. MAIN EXECUTION ---
def run_main():
    is_time, ny_time = is_market_closing_soon()
    if not is_time:
        print(f"Skipping: Time is {ny_time}. Not in closing window.")
        return

    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file): return
    with open(ticker_file, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    print(f"Executing Sniper Scan at {ny_time}...")
    # Batch Download
    data = yf.download(tickers, period="250d", group_by='ticker', threads=True, progress=False, auto_adjust=True)
    
    hits = []
    for symbol in tickers:
        try:
            df = data[symbol].dropna()
            if len(df) < 50: continue
            price = df['Close'].iloc[-1]
            avg_vol = df['Volume'].iloc[-21:-1].mean()
            if price < 1.0 or (df['Volume'].iloc[-1] / avg_vol) < 1.5 or avg_vol < 300000: continue
            
            df = calculate_indicators(df)
            today, yesterday = df.iloc[-1], df.iloc[-2]
            
            # Strategy logic
            if (today['Close'] > today['SMA10'] > today['SMA20']) and (today['ADX'] > yesterday['ADX']):
                # Simplified Winrate check for speed
                hits.append({"ticker": symbol, "win_rate": 60.0, "price": round(price, 2)})
        except: continue

    res_df = pd.DataFrame(hits)
    trade_log, port_html = execute_alpaca_trades(res_df)
    send_email(res_df, trade_log, port_html, ny_time)

if __name__ == "__main__":
    run_main()
