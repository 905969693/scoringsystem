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
    return data


def calculate_indicators(df):
    """
    Calculate technical indicators: RSI, KDJ, Bollinger Bands.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with added indicator columns
    """
    close = df['Close'].values.flatten()
    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()

    df['rsi'] = talib.RSI(close, timeperiod=14)
    
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
    return df
