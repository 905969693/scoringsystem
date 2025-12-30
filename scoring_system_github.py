# app.py
import streamlit as st
import yfinance as yf
import talib
import numpy as np
import pandas as pd

# ========== 你的函数（保持不变）==========
def fetch_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, auto_adjust=True)
    return data


def check_td_nine(df):
    """
    检测神奇九转（TD 9-Count）信号，适配任意 pandas Series。
    
    参数:
        close (pd.Series): 收盘价序列，时间顺序（最新在最后），索引可为任意类型（日期、整数等）
    
    返回:
        dict: 包含 buy/sell 布尔信号及计数值
    """

    # 输入校验：必须是 pandas Series
    close = pd.Series(df['Close'].values.flatten())
    
    if not isinstance(close, pd.Series):
        raise TypeError("Input 'close' must be a pandas Series.")
    
    # 去除 NaN 并确保至少有 9 个有效价格
    close_clean = close.dropna()
    if len(close_clean) < 9:
        return {
            'buy': False,
            'sell': False,
            'buy_count': 0,
            'sell_count': 0
        }
    
    n = len(close_clean)
    buy_seq = [0] * n
    sell_seq = [0] * n

    # 从第 4 个位置开始（i=4），因为需比较 i 与 i-4
    for i in range(4, n):
        # ✅ 使用 .iloc 确保按位置访问，彻底避免 FutureWarning
        current = close_clean.iloc[i]
        ref = close_clean.iloc[i - 4]
        
        if current < ref:
            buy_seq[i] = min(buy_seq[i - 1] + 1, 9)
        if current > ref:
            sell_seq[i] = min(sell_seq[i - 1] + 1, 9)
        # 若 current == ref，buy/sell 计数自动重置为 0（因初始化为 0）
    
    return {
        'buy': buy_seq[-1] >= 9,
        'sell': sell_seq[-1] >= 9,
        'buy_count': buy_seq[-1],
        'sell_count': sell_seq[-1]
    }

def calculate_indicators(df):
    # 确保输入是 DataFrame，并提取 numpy array
    close = df['Close'].values.flatten()
    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()

    # RSI (14日)
    df['rsi'] = talib.RSI(close, timeperiod=14)
    
    # KDJ (常用参数 9,3,3)
    df['k'], df['d'] = talib.STOCH(
        high, low, close,
        fastk_period=9,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0
    )
    df['j'] = 3 * df['k'] - 2 * df['d']  # J = 3K - 2D
    
    # 布林带 (20日, 2标准差)
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
    
    # 布林带位置百分比（0=下轨, 100=上轨）
    df['bb_position'] = 100 * (close - lower) / (upper - lower)
    df['bb_position'] = np.clip(df['bb_position'], 0, 100)  # 防止除零或溢出

    
    return df
def calculate_obos_score(df, weights=None):
    if weights is None:
        weights = {'rsi': 0.4, 'kdj': 0.3, 'bb': 0.3}
    
    # KDJ 打分：用 K 或 J（这里用 J，更敏感）
    kdj_score = np.clip(df['j'], 0, 100)
    
    # RSI 打分：直接使用（30=30分，70=70分）
    rsi_score = df['rsi']
    
    # 布林带打分：价格靠近上轨 → 超买（高分）
    bb_score = df['bb_position']
    
    # 加权融合
    score = (
        weights['rsi'] * rsi_score +
        weights['kdj'] * kdj_score +
        weights['bb'] * bb_score
    ) / sum(weights.values())
    
    return np.clip(score, 0, 100)


# ========== Streamlit 界面 ==========
st.set_page_config(page_title="量化超买超卖评分", layout="centered")
st.title("📊 股票超买超卖评分系统")
st.caption("0 = 极端超卖，100 = 极端超买 | 手机端可直接访问")

# 输入框
symbol = st.text_input("请输入股票代码（如 0700.HK, AAPL, 600519.SS）", value="0700.HK")
end_date = st.date_input("截止日期", value=pd.to_datetime("2025-12-29"))
months_back = st.slider("回溯月数", min_value=1, max_value=12, value=6)

if st.button("📊 计算评分"):
    with st.spinner("正在获取数据并计算..."):
        try:
            start_date = pd.to_datetime(end_date) - pd.DateOffset(months=months_back)
            df = fetch_stock_data(symbol, start=start_date.strftime("%Y-%m-%d"), end=(pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            
            if df.empty:
                st.error("❌ 未获取到数据，请检查股票代码格式")
            else:
                df = calculate_indicators(df)
                df['obos_score'] = calculate_obos_score(df)
                latest_score = df['obos_score'].dropna().iloc[-1]

                # 显示结果
                st.subheader(f"{symbol} 评分结果")
                st.metric("当前超买超卖分", f"{latest_score:.1f} / 100")
                st.progress(int(latest_score))

                # 显示最近60天趋势
                st.line_chart(df['obos_score'].dropna().tail(60))

                # 可选：显示原始数据
                with st.expander("📈 查看原始指标数据"):
                    st.dataframe(df[['Close', 'rsi', 'k', 'd', 'j', 'bb_position', 'obos_score']].tail(10))

        except Exception as e:
            st.error(f"❌ 发生错误: {str(e)}")
