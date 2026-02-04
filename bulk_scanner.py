import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import smtplib
from email.message import EmailMessage
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

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

def get_market_tide():
    try:
        spy = yf.Ticker("SPY").history(period="50d")
        spy_sma20 = spy['Close'].rolling(window=20).mean().iloc[-1]
        current_spy = spy['Close'].iloc[-1]
        return current_spy >= spy_sma20, f"SPY: {current_spy:.2f} (SMA20: {spy_sma20:.2f})"
    except:
        return True, "Market Tide check failed, proceeding."

# --- TRADING EXECUTION ---
def execute_alpaca_trades(winning_df):
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    
    # Change paper=True to False when ready for real money
    client = TradingClient(api_key, secret_key, paper=True)
    
    account = client.get_account()
    equity = float(account.equity)
    
    # POSITION SIZING LOGIC
    MAX_SETUPS = 20
    MAX_CASH_PER_STOCK = 5000.00
    
    # Divide total equity by number of hits (up to 20)
    num_setups = min(len(winning_df), MAX_SETUPS)
    if num_setups == 0: return "No setups to trade."
    
    raw_slot_size = equity / num_setups
    final_slot_size = min(raw_slot_size, MAX_CASH_PER_STOCK)
    
    log_trades = []
    
    # Sort by Win Rate and pick top 20
    top_picks = winning_df.sort_values(by="win_rate_3d", ascending=False).head(MAX_SETUPS)
    
    for _, stock in top_picks.iterrows():
        symbol = stock['ticker']
        price = stock['price']
        qty = int(final_slot_size / price)
        
        if qty > 0:
            try:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(price * 1.03, 2)),
                    stop_loss=StopLossRequest(stop_price=round(price * 0.99, 2))
                )
                client.submit_order(order_data)
                log_trades.append(f"Bought {qty} {symbol} @ {price}")
            except Exception as e:
                log_trades.append(f"Error {symbol}: {e}")
                
    return "\n".join(log_trades)

# --- EMAIL REPORTING ---
def send_report(df, status_msg, trade_log):
    msg = EmailMessage()
    user = os.environ.get('EMAIL_USER')
    receiver = os.environ.get('EMAIL_RECEIVER')
    
    subject = f"🎯 Sniper Report: {len(df)} Setups" if not df.empty else "⚪ Sniper Report: Zero Hits"
    
    body = f"""
    <html><body>
    <h2>Trading System Report</h2>
    <p><b>Market Tide:</b> {status_msg}</p>
    <hr>
    <h3>Execution Log:</h3>
    <pre>{trade_log}</pre>
    <hr>
    <h3>Scanned Setups:</h3>
    {df.to_html(index=False) if not df.empty else "<p>No stocks met criteria.</p>"}
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

    tide_ok, tide_msg = get_market_tide()
    all_results = []
    
    # Batch Processing to prevent Timeouts/Rate Limits
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
                    rel_vol = df['Volume'].iloc[-1] / avg_vol if avg_vol > 300000 else 0
                    
                    if price < 1.0 or rel_vol < 1.5: continue
                    
                    df = calculate_indicators(df)
                    today, yesterday = df.iloc[-1], df.iloc[-2]
                    
                    if (today['Close'] > today['SMA10'] > today['SMA20']) and (today['ADX'] > 20 > yesterday['ADX'] or today['ADX'] > yesterday['ADX']):
                        # Backtest
                        hist = df[(df['Close'] > df['SMA10']) & (df['SMA10'] > df['SMA20']) & (today['ADX'] > yesterday['ADX'])].index
                        wins, total = 0, 0
                        for d in hist:
                            idx = df.index.get_loc(d)
                            if idx + 3 < len(df):
                                if df.iloc[idx+3]['Close'] > df.iloc[idx]['Close']: wins += 1
                                total += 1
                        
                        win_rate = (wins/total*100) if total > 0 else 0
                        if win_rate >= 55:
                            all_results.append({"ticker": symbol, "win_rate_3d": win_rate, "price": price})
                except: continue
        except: continue

    res_df = pd.DataFrame(all_results)
    
    # Execute Trades if Tide is Healthy
    trade_log = "Skipped: Market Tide is LOW"
    if tide_ok and not res_df.empty:
        trade_log = execute_alpaca_trades(res_df)
    elif res_df.empty:
        trade_log = "No setups found today."

    send_report(res_df, tide_msg, trade_log)

if __name__ == "__main__":
    run_scanner()
