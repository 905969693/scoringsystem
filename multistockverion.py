# app.py
import streamlit as st
import yfinance as yf
import talib
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import urllib.parse
from scipy.stats import zscore

def get_watchlist_from_url():
    """从 URL query 参数获取关注列表"""
    query_params = st.experimental_get_query_params()
    tickers = query_params.get("tickers", [""])
    return [t.strip().upper() for t in tickers[0].split(",") if t.strip()] if tickers[0] else []

def set_watchlist_to_url(tickers):
    """将关注列表写入 URL"""
    if tickers:
        st.experimental_set_query_params(tickers=",".join(tickers))
    else:
        st.experimental_set_query_params()  # 清空参数

# ========== 保持你原有的函数不变 ==========
def fetch_stock_data(symbol, start, end,interval):
    data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    return data

def check_td_nine(df):
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

def calculate_indicators(df):
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

def calculate_obos_score(df, weights=None):
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

# ========== 多股票分析函数 ==========
def analyze_single_stock(symbol, start, end,interval):
    """分析单只股票，返回结果字典"""
    try:
        df = fetch_stock_data(symbol, start=start, end=end,interval=interval)
        if df.empty:
            return None
        
        df = calculate_indicators(df)
        df['obos_score'] = calculate_obos_score(df)
        td_signal = check_td_nine(df)
        def rolling_zscore_last(x):
            if len(x) < 2:
                return np.nan
            zs = zscore(x, nan_policy='omit')
            return zs[-1] if not np.isnan(zs[-1]) else np.nan

        # 替换上面的 Z-Score 计算部分为：
        def rolling_pct_rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        df['obos_score_pct'] = df['obos_score'].rolling(window=60, min_periods=30).apply(
            rolling_pct_rank, raw=False
        )
        # 然后返回 'score_pct': float(latest['obos_score_pct'])
        '''
        # 计算Zscore - 使用过去 60 天作为窗口（可调整）
        df['obos_score_zscore'] = df['obos_score'].rolling(window=60, min_periods=30).apply(
            rolling_zscore_last, raw=False
        )
        '''
        latest = df.iloc[-1]
        return {
            'symbol': symbol,
            'price': float(latest['Close']),
            'rsi': float(latest['rsi']),
            'j': float(latest['j']),
            'bb_position': float(latest['bb_position']),
            'score': float(latest['obos_score']),
            'score_zscore': float(latest['obos_score_pct']),  # ← 新增字段 either 'obos_score_pct' or 'obos_score_zscore'
            'td_buy': td_signal['buy'],
            'td_sell': td_signal['sell'],
            'td_buy_count': td_signal['buy_count'],
            'td_sell_count': td_signal['sell_count'],
            'history': df[['Close', 'obos_score','obos_score_zscore']].copy()
        }
    except Exception as e:
        st.warning(f"⚠️ {symbol} 分析失败: {str(e)[:60]}...")
        return None

# ========== Streamlit 界面 ==========
st.set_page_config(page_title="Stock Scoring System", layout="wide")

# 设置字体大小
st.markdown("""
<style>
    /* 全局字体大小 */
    html, body, [class*="css"] {
        font-size: 12px !important;
    }
    
    /* 标题调整 */
    h1 { font-size: 24px !important; }
    h2 { font-size: 20px !important; }
    h3 { font-size: 16px !important; }
    
    /* 输入框、按钮等 */
    .stTextInput, .stButton, .stSelectbox {
        font-size: 12px !important;
    }
    
    /* 表格字体 */
    .stDataFrame {
        font-size: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


st.title("📊 Stock Scoring System")
st.caption("0 = Extreme Oversold，100 = Extreme Overbought")

# 初始化关注列表（从 URL 加载）
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = get_watchlist_from_url()

# 输入区域
col1, col2, col3 = st.columns([3, 1, 1])
ticker_list = "QQQ, SPY, TLT, GLD, USDJPY=X"

with col1:
    symbols_input = st.text_input(
        "Input Ticker(use comma to separate)",
        value=ticker_list,
        help="E.g: QQQ, 0700.HK, USDJPY=X"
    )

with col2:
    months_back = st.slider("Lookback Months", 1, 24, 6)

with col3:
    # 👇 新增：下拉菜单选择 interval
    interval = st.selectbox(
        "Data Interval",
        options=["1d", "1wk"],
        format_func=lambda x: {"1d": "Daily", "1wk": "Weekly"}[x]
    )

# ========== 我的关注列表（通过 URL 保存）==========
st.subheader("📌 Save Your Watchlist")

new_tickers = st.text_input(
    "Add Tickers (separate with comma) ",
    placeholder="e.g.：QQQ, NVDA, 0700.HK",
    key="new_watchlist_input"
)

col_add, col_clear = st.columns([1, 1])
with col_add:
    if st.button("➕ Add & Update URL"):
        if new_tickers.strip():
            added = [s.strip().upper() for s in new_tickers.split(",") if s.strip()]
            current = set(st.session_state.watchlist)
            current.update(added)
            st.session_state.watchlist = sorted(list(current))
            set_watchlist_to_url(st.session_state.watchlist)
            st.success("✅ Updated! Pls save the current URL")
        else:
            st.warning("pls at least input one ticker")

with col_clear:
    if st.button("🗑️ Clear the Watchlist"):
        st.session_state.watchlist = []
        set_watchlist_to_url([])
        st.success("Cleared")

# 显示当前列表
if st.session_state.watchlist:
    st.dataframe(
        pd.DataFrame({"Ticker": st.session_state.watchlist}),
        use_container_width=True,
        hide_index=True
    )
    st.info("🔗 Watchlist saved to current URL. SAVE THIS LINK FOR PERMANENT USE!")
else:
    st.info("After you added ticker, the URL will automatically update. SAVE THIS LINK FOR PERMANENT USE!")


today = date.today()
end_date = st.date_input("End Date", value=today)



if st.button("📊 Analyze All", type="primary"):
    # ✅ 优先使用用户的关注列表
    if st.session_state.watchlist:
        symbols = st.session_state.watchlist
        st.info(f"Analyzing {len(symbols)} Tickers in the Watchlist")
    else:
       # 回退到顶部输入框
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.error("Pls add tickers to watchlist, or simply input ticker(s)")
            st.stop()
            
    # 计算日期范围
    start_date = pd.to_datetime(end_date) - pd.DateOffset(months=months_back)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    # 分析所有股票
    results = []
    with st.spinner(f"Analyzing {len(symbols)} Stocks..."):
        for symbol in symbols:
            result = analyze_single_stock(symbol, start_str, end_str,interval)
            if result:
                results.append(result)
    
    if not results:
        st.error("All failed, pls check formating")
    else:
        # 构建结果表格
        df_results = pd.DataFrame(results)
        df_results = df_results.round(2)
        
        # 显示汇总表
        st.subheader(f"📈 Result ( {len(results)} Stocks)")
        
        # 格式化 TD 信号
        def format_td(row):
            signals = []
            if row['td_buy']:
                signals.append(f"🟢 TD Buy ({int(row['td_buy_count'])})")
            if row['td_sell']:
                signals.append(f"🔴 TD Sell ({int(row['td_sell_count'])})")
            return "; ".join(signals) if signals else "—"
        
        # 选择需要的列，包括 TD 计数
        df_display = df_results[[
            'symbol', 'score', 'td_buy_count', 'td_sell_count', 'rsi', 'j', 'bb_position'
        ]].copy()
        
        # 可选：重命名列，更清晰
        df_display.columns = [
            'Ticker', 'Score','TD Buy', 'TD Sell', 'RSI', 'KDJ-J', 'Bollinger%']
        


        # 使用背景色渐变突出评分 / 无matplot
        st.dataframe(df_display, use_container_width=True, height=500)


    
    with st.expander("Check the Score & Price Trend of Each Ticker"):
        for result in results:
            st.markdown(f"### {result['symbol']}")
            hist = result['history'].dropna()
            
            if len(hist) < 10:
                st.write("⚠️ Not Enough Data (Need at least 10 data points)")
                continue
            
            #hist_plot = hist.tail(60)  这里只取了最后60个数据点
            
            hist_plot = hist #这里全都取了，试试看
            
            fig, ax1 = plt.subplots(figsize=(10, 4))
            
            # 评分（左轴）
            ax1.plot(hist_plot.index, hist_plot['obos_score_zscore'], color='red', linewidth=1.5)
            ax1.set_ylabel('Z-Score', color='red')
            ax1.tick_params(axis='y', labelcolor='red')
            ax1.set_ylim(-3, 3)
            ax1.axhline(1.5, color='orange', linestyle='--', alpha=0.6)
            ax1.axhline(-1.5, color='green', linestyle='--', alpha=0.6)
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            # 股价（右轴）
            ax2 = ax1.twinx()
            ax2.plot(hist_plot.index, hist_plot['Close'], color='blue', linewidth=1.5)
            ax2.set_ylabel('Price', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            # 格式化
            ax1.set_title(f"{result['symbol']} — Technical Score (Red, LHS) vs Price (Blue, RHS)", fontsize=12)
            fig.autofmt_xdate()  # 自动旋转日期
            fig.tight_layout()
            
            st.pyplot(fig)
            plt.close(fig)

        
