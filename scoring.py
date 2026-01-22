"""
OBOS (Overbought/Oversold) scoring and TD Nine signal detection.
"""
import numpy as np
import pandas as pd
from data_fetcher import calculate_indicators, fetch_stock_data


def calculate_obos_score(df, weights=None, use_enhanced=False):
    """
    Calculate OBOS (Overbought/Oversold) score from technical indicators.
    
    Args:
        df: DataFrame with RSI, KDJ-J, BB position, volume, and momentum columns
        weights: Dictionary with weights for each indicator
        use_enhanced: If True, use enhanced scoring with volume and momentum (default: True)
        
    Returns:
        Series with OBOS scores (0-100)
    """
    if use_enhanced:
        # Enhanced scoring with volume and momentum confirmation
        if weights is None:
            weights = {
                'rsi': 0.25,
                'kdj': 0.20,
                'bb': 0.20,
                'volume': 0.15,
                'momentum': 0.20
            }
        
        kdj_score = np.clip(df['j'], 0, 100)
        rsi_score = df['rsi']
        bb_score = df['bb_position']
        
        # Get volume and momentum scores (should be calculated in calculate_indicators)
        volume_score = df.get('volume_score', pd.Series(50.0, index=df.index))
        momentum_score = df.get('momentum_score', pd.Series(50.0, index=df.index))
        
        # Weighted combination
        score = (
            weights['rsi'] * rsi_score +
            weights['kdj'] * kdj_score +
            weights['bb'] * bb_score +
            weights['volume'] * volume_score +
            weights['momentum'] * momentum_score
        ) / sum(weights.values())
        
    else:
        # Original scoring (backward compatibility)
        if weights is None:
            weights = {'rsi': 0.4, 'kdj': 0.3, 'bb': 0.3}
        
        kdj_score = np.clip(df['j'], 0, 100)
        rsi_score = df['rsi']
        bb_score = df['bb_position']
        
        score = (
            weights['rsi'] * rsi_score +
            weights['kdj'] * kdj_score +
            weights['bb'] * bb_score
        ) / sum(weights.values())
    
    return np.clip(score, 0, 100)


def check_td_nine(df):
    """
    Detect TD Nine (神奇九转) signals.
    
    Args:
        df: DataFrame with Close prices
        
    Returns:
        dict: {'buy': bool, 'sell': bool, 'buy_count': int, 'sell_count': int}
    """
    close = pd.Series(df['Close'].values.flatten())
    if not isinstance(close, pd.Series):
        raise TypeError("Input 'close' must be a pandas Series.")
    
    close_clean = close.dropna()
    if len(close_clean) < 9:
        return {'buy': False, 'sell': False, 'buy_count': 0, 'sell_count': 0}
    
    n = len(close_clean)
    buy_seq = [0] * n
    sell_seq = [0] * n

    for i in range(4, n):
        current = close_clean.iloc[i]
        ref = close_clean.iloc[i - 4]
        if current < ref:
            buy_seq[i] = min(buy_seq[i - 1] + 1, 9)
        if current > ref:
            sell_seq[i] = min(sell_seq[i - 1] + 1, 9)
    
    return {
        'buy': buy_seq[-1] >= 9,
        'sell': sell_seq[-1] >= 9,
        'buy_count': buy_seq[-1],
        'sell_count': sell_seq[-1]
    }


def analyze_single_stock(symbol, start, end, interval="1d"):
    """
    Analyze a single stock and return comprehensive results.
    
    Args:
        symbol: Stock ticker symbol
        start: Start date (string)
        end: End date (string)
        interval: Data interval ("1d" or "1wk")
        
    Returns:
        dict with analysis results or None if failed
    """
    # #region agent log
    import json
    try:
        with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"scoring.py:118","message":"analyze_single_stock entry","data":{"symbol":symbol,"start":start,"end":end,"interval":interval},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    except: pass
    # #endregion
    try:
        df = fetch_stock_data(symbol, start=start, end=end, interval=interval)
        # #region agent log
        try:
            with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"scoring.py:121","message":"after fetch_stock_data","data":{"symbol":symbol,"df_empty":df.empty,"df_shape":list(df.shape) if not df.empty else None,"df_columns":list(df.columns) if not df.empty else None},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except: pass
        # #endregion
        if df.empty:
            # #region agent log
            try:
                with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"scoring.py:124","message":"df.empty=True, returning None","data":{"symbol":symbol},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            except: pass
            # #endregion
            return None
        
        # #region agent log
        try:
            with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"scoring.py:127","message":"before calculate_indicators","data":{"symbol":symbol,"has_volume":"Volume" in df.columns},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except: pass
        # #endregion
        df = calculate_indicators(df)
        # #region agent log
        try:
            with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"scoring.py:130","message":"after calculate_indicators","data":{"symbol":symbol,"has_volume_score":"volume_score" in df.columns,"has_momentum_score":"momentum_score" in df.columns},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except: pass
        # #endregion
        df['obos_score'] = calculate_obos_score(df)
        td_signal = check_td_nine(df)
        
        def rolling_pct_rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        df['obos_score_pct'] = df['obos_score'].rolling(window=60, min_periods=30).apply(
            rolling_pct_rank, raw=False
        )

        # Calculate consecutive overbought/oversold days
        score_pct = df['obos_score_pct'].dropna()
        if len(score_pct) == 0:
            consecutive_days = 0
        else:
            low_thresh = 0.10
            high_thresh = 0.90
            
            consecutive_days = 0
            for i in range(len(score_pct) - 1, -1, -1):
                pct_val = score_pct.iloc[i]
                if i == len(score_pct) - 1:
                    if pct_val < low_thresh:
                        consecutive_days = 1
                        direction = 'oversold'
                    elif pct_val > high_thresh:
                        consecutive_days = -1
                        direction = 'overbought'
                    else:
                        consecutive_days = 0
                        break
                else:
                    if consecutive_days > 0 and direction == 'oversold' and pct_val < low_thresh:
                        consecutive_days += 1
                    elif consecutive_days < 0 and direction == 'overbought' and pct_val > high_thresh:
                        consecutive_days -= 1
                    else:
                        break

        latest = df.iloc[-1]
        return {
            'symbol': symbol,
            'price': float(latest['Close']),
            'rsi': float(latest['rsi']),
            'j': float(latest['j']),
            'bb_position': float(latest['bb_position']),
            'score': float(latest['obos_score']),
            'score_pct': float(latest['obos_score_pct']),
            'td_buy': td_signal['buy'],
            'td_sell': td_signal['sell'],
            'td_buy_count': td_signal['buy_count'],
            'td_sell_count': td_signal['sell_count'],
            'consecutive_days': consecutive_days,
            'history': df[['Close', 'obos_score', 'obos_score_pct']].copy()
        }
    except Exception as e:
        # #region agent log
        try:
            with open('/Users/yanyunfeng/文稿/github-quant trading/scoringsystem/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"scoring.py:192","message":"exception caught","data":{"symbol":symbol,"exception_type":type(e).__name__,"exception_msg":str(e)[:200]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except: pass
        # #endregion
        # Handle errors gracefully - works with or without streamlit
        error_msg = f"⚠️ {symbol} 分析失败: {str(e)[:60]}..."
        try:
            import streamlit as st
            st.warning(error_msg)
        except ImportError:
            # Not in streamlit environment, just print
            print(error_msg)
        return None
