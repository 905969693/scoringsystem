"""
Data fetching and technical indicators calculation.
"""
import yfinance as yf
import talib
import numpy as np
import pandas as pd


def fetch_stock_data(symbol, start, end, interval="1d"):
    """Fetch stock data from Yahoo Finance."""
    data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    # Handle MultiIndex columns (yfinance returns MultiIndex when downloading single symbol sometimes)
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex: keep only the first level (OHLCV names)
        data.columns = data.columns.get_level_values(0)
    return data


def calculate_indicators(df):
    """
    Calculate technical indicators: RSI, KDJ, Bollinger Bands, Volume, Momentum.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with added indicator columns
    """
    # #region agent log
    import json
    try:
        with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"data_fetcher.py:16","message":"calculate_indicators entry","data":{"df_shape":list(df.shape),"df_columns":list(df.columns),"has_volume":"Volume" in df.columns},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    except: pass
    # #endregion
    # Ensure we have proper 1D numpy arrays (handle MultiIndex if still present)
    # #region agent log
    try:
        with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"F","location":"data_fetcher.py:41","message":"before array conversion","data":{"close_shape":list(df['Close'].values.shape) if 'Close' in df.columns else None,"close_dtype":str(df['Close'].values.dtype) if 'Close' in df.columns else None},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    except: pass
    # #endregion
    close = np.asarray(df['Close'].values, dtype=np.float64).flatten()
    high = np.asarray(df['High'].values, dtype=np.float64).flatten()
    low = np.asarray(df['Low'].values, dtype=np.float64).flatten()
    volume = np.asarray(df['Volume'].values, dtype=np.float64).flatten() if 'Volume' in df.columns else None
    # #region agent log
    try:
        with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"F","location":"data_fetcher.py:46","message":"after array conversion","data":{"close_shape":list(close.shape),"close_dtype":str(close.dtype),"close_len":len(close)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    except: pass
    # #endregion
    # #region agent log
    try:
        with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"data_fetcher.py:33","message":"volume check","data":{"volume_is_none":volume is None,"volume_len":len(volume) if volume is not None else 0},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    except: pass
    # #endregion

    # Existing indicators
    # #region agent log
    try:
        with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"F","location":"data_fetcher.py:54","message":"before RSI calculation","data":{},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    except: pass
    # #endregion
    df['rsi'] = talib.RSI(close, timeperiod=14)
    # #region agent log
    try:
        with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"F","location":"data_fetcher.py:57","message":"after RSI calculation","data":{},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    except: pass
    # #endregion
    
    df['k'], df['d'] = talib.STOCH(
        high, low, close,
        fastk_period=9,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0
    )
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    upper, middle, lower = talib.BBANDS(
        close,
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
        matype=0
    )
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_position'] = 100 * (close - lower) / (upper - lower + 1e-8)
    df['bb_position'] = np.clip(df['bb_position'], 0, 100)
    
    # NEW: Volume indicators
    if volume is not None and len(volume) > 0:
        # Volume Rate of Change (5-period)
        volume_series = pd.Series(volume, index=df.index)
        volume_roc = volume_series.pct_change(periods=5).fillna(0)
        # Normalize to 0-100: negative ROC = low score, positive ROC = high score
        df['volume_roc'] = np.clip((volume_roc + 1) * 50, 0, 100)
        
        # OBV (On-Balance Volume) trend
        obv = talib.OBV(close, volume)
        obv_trend = pd.Series(obv, index=df.index).pct_change(periods=10).fillna(0)
        df['obv_trend'] = np.clip((obv_trend + 1) * 50, 0, 100)
        
        # Combined volume score
        df['volume_score'] = (df['volume_roc'] * 0.5 + df['obv_trend'] * 0.5)
    else:
        # If no volume data, set neutral scores (as Series with same index)
        # #region agent log
        try:
            with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"data_fetcher.py:76","message":"no volume data, setting neutral scores","data":{"df_index_len":len(df.index)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except: pass
        # #endregion
        df['volume_roc'] = pd.Series(50.0, index=df.index)
        df['obv_trend'] = pd.Series(50.0, index=df.index)
        df['volume_score'] = pd.Series(50.0, index=df.index)
    
    # NEW: Momentum indicators
    # MACD
    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # Normalize MACD histogram to 0-100
    macd_normalized = (hist / (close + 1e-8)) * 1000  # Scale factor
    df['macd_score'] = np.clip(macd_normalized + 50, 0, 100)
    
    # Rate of Change (10-period)
    roc = talib.ROC(close, timeperiod=10)
    df['roc_score'] = np.clip(roc + 50, 0, 100)  # ROC is already in percentage
    
    # Combined momentum score
    df['momentum_score'] = (df['macd_score'] * 0.6 + df['roc_score'] * 0.4)
    
    return df
