import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import smtplib
from email.message import EmailMessage

def calculate_indicators(df):
    """Calculates technical indicators (ADX, SMA, etc.) on the dataframe."""
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
    """Checks SPY to determine if the market is safe to trade."""
    try:
        spy = yf.Ticker("SPY").history(period="50d")
        if spy.empty: return True, "SPY Data Unavailable"
        spy_sma20 = spy['Close'].rolling(window=20).mean().iloc[-1]
        current_spy = spy['Close'].iloc[-1]
        if current_spy < spy_sma20:
            return False, f"Market Tide is LOW (SPY {current_spy:.2f} < {spy_sma20:.2f})"
        return True, "Market Tide is Healthy"
    except:
        return True, "Market Tide Check Failed"

def run_hybrid_scan(ticker_file="tickers.txt"):
    all_results = []
    
    # 1. Tide Check
    tide_ok, tide_msg = get_market_tide()
    # Note: We continue even if tide is low, but the email will warn us.

    if not os.path.exists(ticker_file):
        return pd.DataFrame(), f"Error: {ticker_file} not found."

    with open(ticker_file, 'r') as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()]

    print(f"Loaded {len(all_tickers)} tickers. Starting Hybrid Batch Scan...")

    # 2. BATCH SETTINGS
    BATCH_SIZE = 100  # Scans 100 stocks in 1 request
    
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i+BATCH_SIZE]
        print(f"Processing Batch {i}-{i+len(batch)} / {len(all_tickers)}...")
        
        try:
            # BULK DOWNLOAD (The speed secret)
            # auto_adjust=True fixes split/dividend data issues
            data = yf.download(batch, period="250d", group_by='ticker', threads=True, progress=False, auto_adjust=True)
            
            # Iterate through the batch
            for symbol in batch:
                try:
                    # Handle Single Ticker vs Multi-Ticker return structure
                    if len(batch) == 1:
                        df = data
                    else:
                        df = data[symbol]

                    # Basic Data Validation
                    if df.empty or len(df) < 50: continue
                    
                    # Clean bad data (NaNs)
                    df = df.dropna(subset=['Close', 'Volume'])
                    if df.empty: continue

                    # --- FILTER 1: PRICE & VOLUME (From History) ---
                    price = df['Close'].iloc[-1]
                    if price < 1.00: continue
                    
                    # Volume Check (Using History, not .info)
                    avg_vol_20d = df['Volume'].iloc[-21:-1].mean()
                    if avg_vol_20d < 300_000: continue

                    # --- FILTER 2: INDICATORS ---
                    df = calculate_indicators(df)
                    today = df.iloc[-1]
                    yesterday = df.iloc[-2]

                    # Strategy: Price > SMA10 > SMA20 AND Rising ADX
                    is_trending = (today['Close'] > today['SMA10']) and (today['SMA10'] > today['SMA20'])
                    is_accelerating = (today['ADX'] > 20) and (today['ADX'] > yesterday['ADX'])

                    if is_trending and is_accelerating:
                        # --- FILTER 3: RELATIVE VOLUME ---
                        today_vol = df['Volume'].iloc[-1]
                        rel_vol = today_vol / avg_vol_20d if avg_vol_20d > 0 else 0
                        
                        if rel_vol >= 1.5:
                            # --- FINAL GATE: .INFO CHECK (Only runs on winners) ---
                            # This is where we safely call .info without timeouts
                            try:
                                t_info = yf.Ticker(symbol).info
                                mkt_cap = t_info.get('marketCap', 0)
                                if mkt_cap < 100_000_000: continue
                            except:
                                mkt_cap = 0 # If info fails, we might still want to see it if chart is good

                            # --- BACKTEST ---
                            hist_signals = df[(df['Close'] > df['SMA10']) & 
                                              (df['SMA10'] > df['SMA20']) & 
                                              (df['ADX'] > 20) & 
                                              (df['ADX'] > df['ADX'].shift(1))].index
                            
                            wins, total = 0, 0
                            total_ret = 0
                            for date in hist_signals:
                                idx = df.index.get_loc(date)
                                if idx + 3 < len(df):
                                    ret = (df.iloc[idx + 3]['Close'] - df.iloc[idx]['Close']) / df.iloc[idx]['Close']
                                    if ret > 0: wins += 1
                                    total_ret += ret
                                    total += 1
                            
                            win_rate = (wins/total * 100) if total > 0 else 0
                            avg_ret = (total_ret/total * 100) if total > 0 else 0

                            if win_rate >= 55 and avg_ret >= 3.0:
                                all_results.append({
                                    "ticker": symbol,
                                    "win_rate": f"{win_rate:.1f}%",
                                    "exp_return": f"{avg_ret:.2f}%",
                                    "price": round(price, 2),
                                    "target": round(price * 1.03, 2),
                                    "stop": round(price * 0.99, 2),
                                    "rel_vol": round(rel_vol, 2),
                                    "mkt_cap": f"{mkt_cap/1e6:.1f}M"
                                })
                except Exception as e:
                    continue # Skip individual bad tickers in batch

            # Sleep slightly between batches to be polite
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Batch Failed: {e}")
            continue

    return pd.DataFrame(all_results), tide_msg

def send_email(df, status):
    msg = EmailMessage()
    repo = os.environ.get('GITHUB_REPOSITORY', 'Scanner')
    
    if df.empty:
        subject = "⚪ Zero Hits: Scanner Report"
        body = f"<h3>Status: {status}</h3><p>Scanned full list. No stocks met the 3:1 criteria.</p><p>Source: {repo}</p>"
    else:
        subject = f"🎯 SNIPER ALERT: {len(df)} Setups Found"
        body = f"""
        <html><body>
        <h2 style="color:darkgreen">High Conviction Setups</h2>
        <p><b>Status:</b> {status}</p>
        {df.to_html(index=False)}
        <p>Source: {repo}</p>
        </body></html>
        """
    
    msg.add_alternative(body, subtype='html')
    msg['Subject'] = subject
    msg['From'] = os.environ.get('EMAIL_USER')
    msg['To'] = os.environ.get('EMAIL_RECEIVER')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(os.environ.get('EMAIL_USER'), os.environ.get('EMAIL_PASS'))
        smtp.send_message(msg)

if __name__ == "__main__":
    res, status = run_hybrid_scan()
    send_email(res, status)
