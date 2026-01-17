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
from scipy.stats import percentileofscore


####以下是回测代码部分####
class StrategyParams:
    def __init__(self,
                 lookback_window=60,
                 signal_threshold_low=0.10,
                 signal_threshold_high=0.90,
                 consecutive_days=2,
                 max_position_per_stock=0.15,
                 total_capital=1_000_000,
                 commission_rate=0.001,
                 risk_free_rate=0.02,
                 max_gross_exposure=2.0,
                 allow_shorting=False):  # ← 新增开关，默认开启
        self.lookback_window = lookback_window
        self.signal_threshold_low = signal_threshold_low
        self.signal_threshold_high = signal_threshold_high
        self.consecutive_days = consecutive_days
        self.max_position_per_stock = max_position_per_stock
        self.total_capital = total_capital
        self.commission_rate = commission_rate
        self.risk_free_rate = risk_free_rate
        self.max_gross_exposure = max_gross_exposure
        self.allow_shorting = allow_shorting  # ← 新增

def run_backtest(stock_data_dict, params):
    """
    执行双向回测（支持做多、做空、补仓/加空）
    
    Returns:
    - portfolio_history: DataFrame (date, value, cash)
    - trades_log: list of trade records
    - positions: dict {symbol: {'shares': int, 'entry_price': float}}
    """
    # === 1. 对齐日期 ===
    all_dates = set()
    for df in stock_data_dict.values():
        all_dates.update(df.index)
    all_dates = sorted(pd.to_datetime(list(all_dates)))
    symbols = list(stock_data_dict.keys())
    
    # 初始化信号计数器
    signal_count = {sym: {'buy': 0, 'sell': 0} for sym in symbols}
    
    # 初始化投资组合
    portfolio = {
        'cash': float(params.total_capital),
        'positions': {},  # sym -> {'shares': int (可负), 'entry_price': float}
        'history': [],
        'trades': []
    }
    
    # === 2. 主回测循环 ===
    for date in all_dates:
        if not isinstance(date, pd.Timestamp) or pd.isna(date):
            continue
        
        # --- 2.1 计算当日账户权益（Equity）---
        current_equity = portfolio['cash']
        for sym, pos in portfolio['positions'].items():
            if date in stock_data_dict[sym].index:
                price_val = stock_data_dict[sym].loc[date, 'Close']
                if isinstance(price_val, pd.Series):
                    price_val = price_val.iloc[-1]
                price = float(price_val)
                current_equity += pos['shares'] * price  # 空头 shares 为负，自动减
        
        # 记录历史（value = equity）
        portfolio['history'].append({
            'date': date,
            'value': current_equity,
            'cash': portfolio['cash']
        })
        
        # --- 2.2 检查信号 ---
        buy_signals = []
        sell_signals = []
        
        for sym in symbols:
            if date not in stock_data_dict[sym].index:
                continue
            
            try:
                pct_val = stock_data_dict[sym].loc[date, 'score_percentile']
                if isinstance(pct_val, pd.Series):
                    pct_val = pct_val.iloc[-1]
                pct = float(pct_val)
            except Exception as e:
                continue
            
            # 更新信号计数
            if pct < params.signal_threshold_low:
                signal_count[sym]['buy'] += 1
                signal_count[sym]['sell'] = 0
            elif pct > params.signal_threshold_high:
                signal_count[sym]['sell'] += 1
                signal_count[sym]['buy'] = 0
            else:
                signal_count[sym]['buy'] = 0
                signal_count[sym]['sell'] = 0
            
            if signal_count[sym]['buy'] >= params.consecutive_days:
                buy_signals.append(sym)
            if signal_count[sym]['sell'] >= params.consecutive_days:
                sell_signals.append(sym)
        
        # --- 2.3 先处理平仓（反向信号）---
        # 平多仓
        for sym in sell_signals:
            if sym in portfolio['positions'] and portfolio['positions'][sym]['shares'] > 0:
                shares = portfolio['positions'][sym]['shares']
                price_val = stock_data_dict[sym].loc[date, 'Close']
                if isinstance(price_val, pd.Series):
                    price_val = price_val.iloc[-1]
                price = float(price_val)
                proceeds = shares * price
                commission = proceeds * params.commission_rate
                portfolio['cash'] += proceeds - commission
                portfolio['trades'].append({
                    'date': date, 'symbol': sym, 'action': 'SELL',
                    'shares': shares, 'price': price, 'commission': commission
                })
                del portfolio['positions'][sym]
        
        # 平空仓
        for sym in buy_signals:
            if sym in portfolio['positions'] and portfolio['positions'][sym]['shares'] < 0:
                short_shares = -portfolio['positions'][sym]['shares']  # 正数
                price_val = stock_data_dict[sym].loc[date, 'Close']
                if isinstance(price_val, pd.Series):
                    price_val = price_val.iloc[-1]
                price = float(price_val)
                cost = short_shares * price
                commission = cost * params.commission_rate
                portfolio['cash'] -= cost + commission
                portfolio['trades'].append({
                    'date': date, 'symbol': sym, 'action': 'BUY_TO_COVER',
                    'shares': short_shares, 'price': price, 'commission': commission
                })
                del portfolio['positions'][sym]
        
        # --- 2.4 再处理开仓/加仓 ---
        # 处理买入/做多
        if buy_signals:
            for sym in buy_signals:
                if sym in portfolio['positions'] and portfolio['positions'][sym]['shares'] < 0:
                    continue  # 应已平空，安全跳过
                
                price_val = stock_data_dict[sym].loc[date, 'Close']
                if isinstance(price_val, pd.Series):
                    price_val = price_val.iloc[-1]
                price = float(price_val)
                
                max_allowed_value = params.max_position_per_stock * current_equity
                current_value = 0.0
                if sym in portfolio['positions']:
                    current_value = portfolio['positions'][sym]['shares'] * price
                
                remaining_capacity = max_allowed_value - current_value
                alloc_per_stock = portfolio['cash'] / len(buy_signals)
                amount_to_invest = min(alloc_per_stock, remaining_capacity)
                
                if amount_to_invest > price and portfolio['cash'] > 0:
                    new_shares = int(amount_to_invest // price)
                    if new_shares <= 0:
                        continue
                    
                    cost = new_shares * price
                    commission = cost * params.commission_rate
                    total_cost = cost + commission
                    
                    if total_cost <= portfolio['cash']:
                        portfolio['cash'] -= total_cost
                        
                        if sym in portfolio['positions']:
                            old_shares = portfolio['positions'][sym]['shares']
                            old_cost_basis = old_shares * portfolio['positions'][sym]['entry_price']
                            new_cost_basis = new_shares * price
                            avg_price = (old_cost_basis + new_cost_basis) / (old_shares + new_shares)
                            portfolio['positions'][sym]['shares'] += new_shares
                            portfolio['positions'][sym]['entry_price'] = avg_price
                            action = 'ADD'
                        else:
                            portfolio['positions'][sym] = {
                                'shares': new_shares,
                                'entry_price': price
                            }
                            action = 'BUY'
                        
                        portfolio['trades'].append({
                            'date': date, 'symbol': sym, 'action': action,
                            'shares': new_shares, 'price': price, 'commission': commission
                        })
        
        # 处理卖出/做空
        if params.allow_shorting and sell_signals:
            for sym in sell_signals:
                if sym in portfolio['positions'] and portfolio['positions'][sym]['shares'] > 0:
                    continue  # 应已平多，安全跳过
                
                price_val = stock_data_dict[sym].loc[date, 'Close']
                if isinstance(price_val, pd.Series):
                    price_val = price_val.iloc[-1]
                price = float(price_val)
                
                max_allowed_value = params.max_position_per_stock * current_equity
                current_short_value = 0.0
                if sym in portfolio['positions']:
                    current_short_value = -portfolio['positions'][sym]['shares'] * price  # 空头市值（正数）
                
                remaining_capacity = max_allowed_value - current_short_value
                # 做空额度：简化使用固定比例或剩余容量
                amount_to_short = remaining_capacity
                
                if amount_to_short > price:
                    new_shares = int(amount_to_short // price)
                    if new_shares <= 0:
                        continue
                    
                    proceeds = new_shares * price
                    commission = proceeds * params.commission_rate
                    portfolio['cash'] += proceeds - commission
                    
                    if sym in portfolio['positions']:
                        portfolio['positions'][sym]['shares'] -= new_shares
                        action = 'ADD_SHORT'
                    else:
                        portfolio['positions'][sym] = {
                            'shares': -new_shares,
                            'entry_price': price
                        }
                        action = 'SELL_SHORT'
                    
                    portfolio['trades'].append({
                        'date': date, 'symbol': sym, 'action': action,
                        'shares': new_shares, 'price': price, 'commission': commission
                    })
    
    # === 3. 返回结果 ===
    history_df = pd.DataFrame(portfolio['history']).set_index('date')
    return history_df[['value', 'cash']], portfolio['trades'], portfolio['positions']

def prepare_stock_data_dict(symbols, start_date, end_date, interval="1d"):
    stock_data_dict = {}
    for sym in symbols:
        try:
            df = fetch_stock_data(sym, start=start_date, end=end_date, interval=interval)
            if df.empty or len(df) < 30:
                continue

            # === 关键：标准化索引 ===
            df.index = pd.to_datetime(df.index)  # 转为 datetime
            df.index = df.index.tz_localize(None)  # 去掉时区（yfinance 常带 UTC）
            df = df[~df.index.duplicated(keep='first')].sort_index()
            # ======================

            df = calculate_indicators(df)
            df['obos_score'] = calculate_obos_score(df)

            # 截断 NaN
            first_valid = df['obos_score'].first_valid_index()
            if pd.isna(first_valid):
                continue
            df = df.loc[first_valid:].copy()

            def rolling_pct_rank(x):
                return pd.Series(x).rank(pct=True).iloc[-1]

            df['score_percentile'] = df['obos_score'].rolling(
                window=60, min_periods=30
            ).apply(rolling_pct_rank, raw=False)

            # 在计算完 score_percentile 后
            df['score_percentile'] = pd.to_numeric(df['score_percentile'], errors='coerce')
            df['obos_score'] = pd.to_numeric(df['obos_score'], errors='coerce')

            first_valid_pct = df['score_percentile'].first_valid_index()
            if pd.isna(first_valid_pct):
                continue
            df = df.loc[first_valid_pct:].copy()

            stock_data_dict[sym] = df[['Close', 'obos_score', 'score_percentile']].copy()

        except Exception as e:
            print(f"⚠️ 跳过 {sym}: {str(e)[:60]}")
            continue

    return stock_data_dict


def calculate_performance(portfolio_history, params):
    """计算绩效指标"""
    # 净值序列
    nav = portfolio_history['value'] / params.total_capital

    # 总收益
    total_return = nav.iloc[-1] - 1.0
    
    # 日收益率
    daily_returns = nav.pct_change().dropna()
    # 年化夏普比率（假设252交易日）
    annualized_return = daily_returns.mean() * 252
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (annualized_return - params.risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
    
    # 最大回撤
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    dd_start = drawdown.idxmin()
    dd_peak = rolling_max[:dd_start].idxmax()
    dd_end = nav[dd_start:].idxmax()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'drawdown_period': (dd_peak, dd_start, dd_end),
        'daily_returns': daily_returns,
        'nav': nav
    }


def plot_performance(perf_result, title="Strategy Performance"):
    nav = perf_result['nav']
    _, dd_start, dd_end = perf_result['drawdown_period']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(nav.index, nav, color='blue', label='Portfolio NAV')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8)
    
    # === 修复：提取回撤区间内的完整 x 和 y ===
    if pd.notna(dd_start) and pd.notna(dd_end):
        mask = (nav.index >= dd_start) & (nav.index <= dd_end)
        if mask.any():
            x_fill = nav.index[mask]
            y_fill = nav[mask]
            y_cummax = y_fill.cummax()
            
            ax.fill_between(
                x_fill,
                y_fill,
                y_cummax,
                color='red', alpha=0.3,
                label=f'Max Drawdown ({perf_result["max_drawdown"]:.1%})'
            )
    # ======================================
    
    ax.set_title(f"{title}\n"
                 f"Total Return: {perf_result['total_return']:.1%} | "
                 f"Sharpe: {perf_result['sharpe_ratio']:.2f} | "
                 f"Max DD: {perf_result['max_drawdown']:.1%}")
    ax.set_ylabel("Normalized Value (Base=1.0)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# ==============================
# 回测主入口函数（供你调用）
# ==============================
def run_full_backtest(symbols, start_date, end_date, params=None):
    """
    完整回测流程：数据准备 → 回测 → 绩效 → 返回结果
    """
    if params is None:
        params = StrategyParams()
    
    print("📥 Loading Data...")
    stock_data_dict = prepare_stock_data_dict(symbols, start_date, end_date)
    print(f"✅ Loaded {len(stock_data_dict)} Stocks")
    
    print("⚙️ Backtesting...")
    history, trades, final_positions = run_backtest(stock_data_dict, params)
    
    print("📊 Performance...")
    perf = calculate_performance(history, params)
    
    print("📈 Charts...")
    fig = plot_performance(perf)

    return {
        'portfolio_history': history,
        'trades': trades,
        'final_positions': final_positions,
        'performance': perf,
        'figure': fig,
        'stock_data_dict': stock_data_dict 
    }

#### 以上是回测代码部分####

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
        def rolling_pct_rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        df['obos_score_pct'] = df['obos_score'].rolling(window=60, min_periods=30).apply(
            rolling_pct_rank, raw=False
        )

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
            'history': df[['Close', 'obos_score','obos_score_pct']].copy()
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
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.error("Pls add tickers to watchlist, or simply input ticker(s)")
            st.stop()
            
    # 计算日期范围
    start_date = pd.to_datetime(end_date) - pd.DateOffset(months=months_back)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # ===== 缓存到 session_state 供回测使用 =====
    st.session_state.last_symbols = symbols
    st.session_state.last_start_str = start_str
    st.session_state.last_end_str = end_str
    # =========================================

    
    # 分析所有股票（用于个股图表）
    results = []
    with st.spinner(f"Analyzing {len(symbols)} Stocks..."):
        for symbol in symbols:
            result = analyze_single_stock(symbol, start_str, end_str, interval)
            if result:
                results.append(result)
    
    if not results:
        st.error("All failed, pls check formating")
    else:
        df_results = pd.DataFrame(results).round(2)
        df_display = df_results[[
            'symbol', 'score', 'score_pct', 'td_buy_count', 'td_sell_count', 'rsi', 'j', 'bb_position'
        ]].copy()
        df_display.columns = [
            'Ticker', 'Score', 'Score in Percentile', 'TD Buy', 'TD Sell', 'RSI', 'KDJ-J', 'Bollinger%']
        st.subheader(f"📈 Result ( {len(results)} Stocks)")
        st.dataframe(df_display, use_container_width=True, height=500)

    with st.expander("🔍 Check the Score & Price Trend of Each Ticker (Click to Expand)"):
        for result in results:
            st.markdown(f"### {result['symbol']}")
            hist = result['history'].dropna()
            if len(hist) < 10:
                st.write("⚠️ Not Enough Data (Need at least 10 data points)")
                continue
            hist_plot = hist
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(hist_plot.index, hist_plot['obos_score'], color='red', linewidth=1.5)
            ax1.set_ylabel('Technical Score ', color='red')
            ax1.tick_params(axis='y', labelcolor='red')
            ax1.set_ylim(0, 100)
            ax1.axhline(90, color='orange', linestyle='--', alpha=0.6)
            ax1.axhline(10, color='green', linestyle='--', alpha=0.6)
            ax1.grid(True, linestyle='--', alpha=0.3)
            dates = hist_plot.index
            overbought = hist_plot['obos_score_pct'] > 0.9
            ax1.fill_between(dates, 0, 100, where=overbought, 
                             color='red', alpha=0.2, label='Overbought (pct > 0.9)')
            oversold = hist_plot['obos_score_pct'] < 0.1
            ax1.fill_between(dates, 0, 100, where=oversold, 
                             color='green', alpha=0.2, label='Oversold (pct < 0.1)')
            ax2 = ax1.twinx()
            ax2.plot(hist_plot.index, hist_plot['Close'], color='blue', linewidth=1.5)
            ax2.set_ylabel('Price', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax1.set_title(f"{result['symbol']} — Technical Score (Red, LHS) vs Price (Blue, RHS)", fontsize=12)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    

# ========== 回测功能（默认折叠）==========
with st.expander("🔍 Run Full Backtest (Click to Expand)"):
    if 'last_symbols' not in st.session_state:
        st.warning("⚠️ Please click 'Analyze All' first to load stock data.")
    else:
        st.subheader("⚙️ Backtest Configuration")
        
        # --- 是否允许做空 ---
        allow_shorting = st.checkbox("-Allow Shorting", value=True)
        
        # --- 其他参数 ---
        # --- 信号阈值参数 ---
        col_thresh1, col_thresh2 = st.columns(2)
        with col_thresh1:
            signal_threshold_low = st.slider(
                "Signal Threshold Low (超卖线)",
                0.01, 0.49, 0.10, step=0.01,
                help="Score percentile below this triggers BUY"
            )
        with col_thresh2:
            signal_threshold_high = st.slider(
                "Signal Threshold High (超买线)",
                0.51, 0.99, 0.90, step=0.01,
                help="Score percentile above this triggers SELL/SHORT"
            )
        
        # --- 其他参数 ---
        col_bt1, col_bt2, col_bt3 = st.columns(3)
        with col_bt1:
            max_position_per_stock = st.slider("Max Position per Stock (%)", 5, 30, 20) / 100.0
        with col_bt2:
            consecutive_days = st.number_input("Consecutive Days for Signal", 1, 5, 2, step=1)
        with col_bt3:
            total_capital = st.number_input("Initial Capital ($)", 10_000, 10_000_000, 1_000_000, step=100_000)
        
        if st.button("🚀 Run Backtest", type="primary"):
            symbols = st.session_state.last_symbols
            start_str = st.session_state.last_start_str
            end_str = st.session_state.last_end_str
            
            # 构建策略参数
            params = StrategyParams(
                    consecutive_days=int(consecutive_days),
                    signal_threshold_low=signal_threshold_low,
                    signal_threshold_high=signal_threshold_high,
                    max_position_per_stock=max_position_per_stock,
                    total_capital=total_capital,
                    commission_rate=0.001,
                    allow_shorting=allow_shorting
                )
            
            # 运行回测
            with st.spinner("Running backtest..."):
                result_backtest = run_full_backtest(symbols, start_str, end_str, params)
            
            # === 1. 绩效指标 ===
            perf = result_backtest['performance']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{perf['total_return']:.1%}")
            with col2:
                st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{perf['max_drawdown']:.1%}")
            
            # === 2. 净值曲线 ===
            st.pyplot(result_backtest['figure'])
            
            # === 3. 当前持仓 ===
            st.subheader("💼 Current Holdings")
            final_positions = result_backtest['final_positions']
            stock_data_dict = result_backtest['stock_data_dict']
            
            if final_positions:
                pos_df = pd.DataFrame.from_dict(final_positions, orient='index')
                pos_df.index.name = 'Ticker'
                
                current_prices = []
                for sym in pos_df.index:
                    if sym in stock_data_dict and len(stock_data_dict[sym]) > 0:
                        price_val = stock_data_dict[sym].iloc[-1]['Close']
                        if isinstance(price_val, pd.Series):
                            price_val = price_val.iloc[-1]
                        try:
                            current_prices.append(float(price_val))
                        except (TypeError, ValueError):
                            current_prices.append(np.nan)
                    else:
                        current_prices.append(np.nan)
                
                pos_df['shares'] = pd.to_numeric(pos_df['shares'], errors='coerce')
                pos_df['entry_price'] = pd.to_numeric(pos_df['entry_price'], errors='coerce')
                pos_df['current_price'] = pd.to_numeric(current_prices, errors='coerce')
                pos_df['current_MV'] = pos_df['shares'] * pos_df['current_price']
                
                total_equity = float(result_backtest['portfolio_history']['value'].iloc[-1])
                if total_equity != 0:
                    pos_df['position %'] = pos_df['current_MV'] / abs(total_equity)
                else:
                    pos_df['position %'] = 0.0
                
                for col in ['shares', 'entry_price', 'current_price', 'position %']:
                    pos_df[col] = pd.to_numeric(pos_df[col], errors='coerce').fillna(0.0)
                
                display_df = pos_df[['shares', 'entry_price', 'current_price', 'position %']].copy()
                st.dataframe(display_df.style.format({
                    'shares': "{:+,.0f}",
                    'entry_price': "{:.2f}",
                    'current_price': "{:.2f}",
                    'position %': "{:.1%}"
                }))
            else:
                st.info("📭 No positions at end of backtest.")
            
            # === 4. 交易历史 ===
            st.subheader("📜 Trade History")
            trades = result_backtest['trades']
            
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df['market_value'] = abs(trades_df['shares']) * trades_df['price']

                trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                trades_df = trades_df.rename(columns={
                    'date': 'Date',
                    'symbol': 'Ticker',
                    'action': 'Action',
                    'shares': 'Shares',
                    'price': 'Price'
                })
                trades_df = trades_df.sort_values('Date', ascending=False).reset_index(drop=True)
                display_cols = ['Date', 'Ticker', 'Action', 'Shares', 'Price']
                st.dataframe(
                    trades_df[display_cols].style.format({
                        'Price': "${:.2f}"
                    }),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("📭 No trades executed during backtest period.")
