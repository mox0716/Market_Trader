import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import smtplib
from email.message import EmailMessage
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

# --- TECHNICAL CALCULATIONS ---
def calculate_indicators(df):
    df = df.copy()
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    df['TR'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    
    period = 14
    df['+DI'] = 100 * (df['+DM'].rolling(period).mean() / df['TR'].rolling(period).mean())
    df['-DI'] = 100 * (df['-DM'].rolling(period).mean() / df['TR'].rolling(period).mean())
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(period).mean()
    
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    return df

# --- TRADING EXECUTION ---
def execute_alpaca_trades(winning_df):
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    # SETTING 1: CHANGE paper=True to False for LIVE
    client = TradingClient(api_key, secret_key, paper=True) 
    
    account = client.get_account()
    equity = float(account.equity)
    
    # 1. DAY TRADE SAFEGUARD ($30k Threshold)
    if equity < 30000:
        dt_count = int(account.daytrade_count)
        if dt_count >= 2:
            return f"Trade Blocked: PDT Safeguard Active. Day Trades: {dt_count}/3.", ""

    # 2. FETCH EXISTING POSITIONS
    positions = client.get_all_positions()
    existing_tickers = [p.symbol for p in positions]
    
    portfolio_data = []
    for p in positions:
        portfolio_data.append({
            "Symbol": p.symbol, "Qty": p.qty,
            "Avg Price": round(float(p.avg_entry_price), 2),
            "Current": round(float(p.current_price), 2),
            "P/L $": round(float(p.unrealized_pl), 2),
            "P/L %": f"{float(p.unrealized_pl_pc)*100:.2f}%"
        })
    portfolio_html = pd.DataFrame(portfolio_data).to_html(index=False) if portfolio_data else "<p>No open positions.</p>"

    # 3. PENDING ORDERS CHECK
    orders_request = GetOrdersRequest(status=QueryOrderStatus.OPEN, side=OrderSide.BUY)
    open_orders = client.get_orders(filter=orders_request)
    pending_tickers = [o.symbol for o in open_orders]
    excluded_tickers = set(existing_tickers + pending_tickers)

    # 4. POSITION SIZING
    MAX_SETUPS = 20
    MAX_CASH_PER_STOCK = 5000.00
    
    fresh_setups = winning_df[~winning_df['ticker'].isin(excluded_tickers)]
    num_setups = min(len(fresh_setups), MAX_SETUPS)
    
    if num_setups == 0: 
        return "No new setups found (or all already owned).", portfolio_html
    
    final_slot_size = min((equity / num_setups), MAX_CASH_PER_STOCK)
    
    log_trades = []
    top_picks = fresh_setups.sort_values(by="win_rate_3d", ascending=False).head(MAX_SETUPS)
    
    for _, stock in top_picks.iterrows():
        symbol = stock['ticker']
        price = stock['price']
        qty = int(final_slot_size / price)
        
        if qty > 0:
            try:
                order_data = MarketOrderRequest(
                    symbol=symbol, qty=qty, side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(price * 1.03, 2)),
                    stop_loss=StopLossRequest(stop_price=round(price * 0.99, 2))
                )
                client.submit_order(order_data)
                log_trades.append(f"Bought {qty} {symbol} @ {price}")
            except Exception as e:
                log_trades.append(f"Error {symbol}: {e}")
                
    return "\n".join(log_trades), portfolio_html

# --- EMAIL REPORTING ---
def send_report(df, trade_log, portfolio_html, account_summary):
    msg = EmailMessage()
    user = os.environ.get('EMAIL_USER')
    receiver = os.environ.get('EMAIL_RECEIVER')
    
    subject = f"🎯 Sniper Report: {len(df)} Setups" if not df.empty else "⚪ Sniper Report: Zero Hits"
    
    body = f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2 style="color: #2E86C1;">Daily Trading Command Center</h2>
    
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #ddd;">
        <h3 style="margin-top: 0; color: #283747;">Account Overview</h3>
        <p><b>Total Equity:</b> <span style="color: #117864; font-size: 1.2em;">${account_summary['equity']}</span></p>
        <p><b>Buying Power:</b> ${account_summary['buying_power']}</p>
        <p><b>Account Type:</b> {account_summary['mode']}</p>
    </div>

    <h3 style="color: #117864;">Current Portfolio Performance</h3>
    {portfolio_html}
    
    <h3 style="color: #A04000;">New Execution Log</h3>
    <pre style="background: #f4f4f4; padding: 10px; border: 1px solid #ddd;">{trade_log}</pre>
    
    <h3 style="color: #1B4F72;">Today's Scanned Setups</h3>
    {df.to_html(index=False) if not df.empty else "<p>No new stocks met technical criteria today.</p>"}
    
    <p style="font-size: 10px; color: gray;">Settings: 3:1 RR | $5k Cap | $30k PDT Shield</p>
    </body></html>
    """
    msg.add_alternative(body, subtype='html')
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = receiver
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(user, os.environ.get('EMAIL_PASS'))
        smtp.send_message(msg)

# --- MAIN ENGINE ---
def run_scanner():
    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file): return
    with open(ticker_file, 'r') as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()]

    all_results = []
    BATCH_SIZE = 100
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i+BATCH_SIZE]
        try:
            data = yf.download(batch, period="250d", group_by='ticker', threads=True, progress=False, auto_adjust=True)
            for symbol in batch:
                try:
                    df = data[symbol].dropna()
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
                        
                        win_rate = (wins/total*100) if total > 0 else 0
                        if win_rate >= 55:
                            all_results.append({"ticker": symbol, "win_rate_3d": round(win_rate, 2), "price": round(price, 2)})
                except: continue
        except: continue

    res_df = pd.DataFrame(all_results)
    
    # FETCH ACCOUNT INFO
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    # SETTING 2: CHANGE paper=True to False for LIVE
    client = TradingClient(api_key, secret_key, paper=True)
    
    account = client.get_account()
    account_summary = {
        "equity": f"{float(account.equity):,.2f}",
        "buying_power": f"{float(account.buying_power):,.2f}",
        "mode": "PAPER" if client.paper else "LIVE"
    }
    
    trade_log = "No new setups found today."
    positions = client.get_all_positions()
    portfolio_data = []
    for p in positions:
        portfolio_data.append({
            "Symbol": p.symbol, "Qty": p.qty, "Avg Price": round(float(p.avg_entry_price), 2),
            "Current": round(float(p.current_price), 2), "P/L $": round(float(p.unrealized_pl), 2),
            "P/L %": f"{float(p.unrealized_pl_pc)*100:.2f}%"
        })
    portfolio_html = pd.DataFrame(portfolio_data).to_html(index=False) if portfolio_data else "<p>No open positions.</p>"

    if not res_df.empty:
        trade_log, portfolio_html = execute_alpaca_trades(res_df)

    send_report(res_df, trade_log, portfolio_html, account_summary)

if __name__ == "__main__":
    run_scanner()
