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
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- 1. THE BOUNCER ---
def is_market_closing_soon():
    # FORCE TRUE FOR TESTING NOW
    return True, "DEBUG MODE"

# --- 2. MARKET TIDE ---
def get_market_tide():
    session = get_yfinance_session()
    try:
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
        # PANDAS-TA CALCULATIONS
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
    
    # Portfolio Check
    positions = client.get_all_positions()
    existing = [p.symbol for p in positions]
    
    port_list = [{"Symbol": p.symbol, "P/L": p.unrealized_pl} for p in positions]
    port_html = pd.DataFrame(port_list).to_html() if port_list else "<p>No positions.</p>"

    if 'ticker' not in winning_df.columns: return "Error: Missing ticker col", port_html
    
    fresh = winning_df[~winning_df['ticker'].isin(existing)]
    if fresh.empty: return "No new setups.", port_html

    # EXECUTION LOOP
    account = client.get_account()
    equity = float(account.equity)
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
                log.append(f"BOUGHT {stock['ticker']}")
        except Exception as e: log.append(f"Err {stock['ticker']}: {e}")
            
    return "\n".join(log), port_html

# --- 5. MAIN LOGIC ---
def run_main():
    # SKIP TIME CHECK FOR DIAGNOSTIC RUN
    tide_safe, tide_msg = get_market_tide()
    print(f"Tide: {tide_msg}")

    ticker_file = "tickers.txt"
    with open(ticker_file, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    print(f"Downloading {len(tickers)} tickers...")
    session = get_yfinance_session()
    
    try:
        # DOWNLOAD
        data = yf.download(tickers, period="250d", group_by='ticker', threads=True, progress=False, auto_adjust=True, session=session)
        print(f"Download Complete. Data Shape: {data.shape}") # DIAGNOSTIC PRINT 1
    except Exception as e:
        print(f"CRITICAL DOWNLOAD FAIL: {e}")
        return

    hits = []
    debug_counter = 0 # Counter for first 5 logs
    
    for symbol in tickers:
        try:
            # Multi-Index Handling
            if len(tickers) > 1:
                # Check if symbol exists in columns
                if symbol not in data.columns.levels[0]:
                    if debug_counter < 5: print(f"DEBUG: {symbol} not in data columns."); debug_counter+=1
                    continue
                df = data[symbol]
            else:
                df = data

            # Basic Validation
            if df.empty or df['Close'].isna().all(): 
                if debug_counter < 5: print(f"DEBUG: {symbol} empty/NaN."); debug_counter+=1
                continue

            df = df.dropna()
            if len(df) < 50: 
                if debug_counter < 5: print(f"DEBUG: {symbol} too short ({len(df)} rows)."); debug_counter+=1
                continue
            
            price = df['Close'].iloc[-1]
            avg_vol = df['Volume'].iloc[-21:-1].mean()
            current_vol = df['Volume'].iloc[-1]
            
            # --- FILTER DIAGNOSTIC ---
            # 1. Price
            if price < 1.0:
                if debug_counter < 5: print(f"REJECT {symbol}: Price ${price:.2f}"); debug_counter+=1
                continue
                
            # 2. Volume (Lowered to 1.2 for test)
            rel_vol = current_vol / avg_vol if avg_vol > 0 else 0
            if avg_vol < 300000 or rel_vol < 1.2:
                if debug_counter < 5: print(f"REJECT {symbol}: Vol {int(avg_vol)}, RelVol {rel_vol:.2f}"); debug_counter+=1
                continue

            # 3. Technicals
            df = calculate_indicators(df)
            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            
            trend_ok = (today['Close'] > today['SMA10'] > today['SMA20'])
            adx_ok = (today['ADX'] > yesterday['ADX'])
            
            if not trend_ok:
                if debug_counter < 5: print(f"REJECT {symbol}: No Uptrend"); debug_counter+=1
                continue
            
            if not adx_ok:
                if debug_counter < 5: print(f"REJECT {symbol}: ADX Falling"); debug_counter+=1
                continue
                
            # 4. Backtest (Lowered to 50% for test)
            hist = df[(df['Close'] > df['SMA10']) & (df['SMA10'] > df['SMA20']) & (df['ADX'] > df['ADX'].shift(1))].index
            wins, total = 0, 0
            for d in hist:
                idx = df.index.get_loc(d)
                if idx + 3 < len(df):
                    if df.iloc[idx+3]['Close'] > df.iloc[idx]['Close']: wins += 1
                    total += 1
            
            win_rate = (wins/total)*100 if total > 0 else 0
            
            if win_rate >= 50:
                print(f"*** HIT FOUND: {symbol} (WR: {win_rate:.1f}%) ***")
                hits.append({"ticker": symbol, "win_rate": win_rate, "price": price})
            else:
                if debug_counter < 5: print(f"REJECT {symbol}: WinRate {win_rate:.1f}%"); debug_counter+=1
                
        except Exception as e:
            if debug_counter < 5: print(f"ERROR {symbol}: {e}"); debug_counter+=1
            continue

    res_df = pd.DataFrame(hits)
    
    # Send Email
    msg = EmailMessage()
    msg['Subject'] = f"Diagnostic Report: {len(hits)} Hits"
    msg['From'] = os.environ.get('EMAIL_USER')
    msg['To'] = os.environ.get('EMAIL_RECEIVER')
    
    # HTML Body with Debug Info
    body = f"""
    <h3>Run Complete</h3>
    <p>Data Shape: {data.shape if 'data' in locals() else 'Failed'}</p>
    <p>Hits Found: {len(hits)}</p>
    {res_df.to_html() if not res_df.empty else "No Hits"}
    """
    msg.add_alternative(body, subtype='html')
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(os.environ.get('EMAIL_USER'), os.environ.get('EMAIL_PASS'))
        smtp.send_message(msg)

if __name__ == "__main__":
    run_main()
