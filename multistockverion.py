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
                 max_position_per_stock=0.15,  # 单票最大仓位比例
                 total_capital=1_000_000,
                 commission_rate=0.001,        # 佣金率
                 risk_free_rate=0.02):         # 年化无风险利率
        self.lookback_window = lookback_window
        self.signal_threshold_low = signal_threshold_low
        self.signal_threshold_high = signal_threshold_high
        self.consecutive_days = consecutive_days
        self.max_position_per_stock = max_position_per_stock
        self.total_capital = total_capital
        self.commission_rate = commission_rate
        self.risk_free_rate = risk_free_rate

def run_backtest(stock_data_dict, params):
    """
    执行回测
    
    Parameters:
    - stock_data_dict: dict {symbol: DataFrame}，每个 DataFrame 必须包含 'Close' 和 'score_percentile'
    - params: StrategyParams 实例
    
    Returns:
    - portfolio_history: DataFrame (date, value, cash)
    - trades_log: list of trade records
    - positions_log: dict {symbol: [position records]}
    """
    # === 1. 对齐所有股票的日期索引 ===
    all_dates = set()
    for df in stock_data_dict.values():
        all_dates.update(df.index)  # df.index 已是 DatetimeIndex
    all_dates = sorted(pd.to_datetime(list(all_dates)))  # 确保是 Timestamp 列表

    symbols = list(stock_data_dict.keys())
    
    # 初始化信号计数器
    signal_count = {sym: {'buy': 0, 'sell': 0} for sym in symbols}
    
    # 初始化投资组合
    portfolio = {
        'cash': float(params.total_capital),
        'positions': {},  # sym -> {'shares': int, 'entry_price': float}
        'history': [],
        'trades': []
    }
    
    # === 2. 主回测循环 ===
    for date in all_dates:
        
    
        # === 新增：确保 date 是标量 Timestamp ===
        if not isinstance(date, pd.Timestamp):
            continue
        if pd.isna(date):
            continue
        # ======================================

        
        # --- 2.1 更新当前持仓市值 ---
        current_value = portfolio['cash']
        for sym, pos in portfolio['positions'].items():
            if date in stock_data_dict[sym].index:
                try:
                    price = stock_data_dict[sym].loc[date, 'Close'].iloc[0]
                except Exception as e:
                    print(f"⚠️ Price access error for {sym} on {date}: {e}")
                    continue
                current_value += pos['shares'] * price
        
        portfolio['history'].append({
            'date': date,
            'value': current_value,
            'cash': portfolio['cash']
        })
        
        # --- 2.2 检查当日信号（仅在有数据的股票上）---
        buy_signals = []
        sell_signals = []
        
        for sym in symbols:
            if date not in stock_data_dict[sym].index:
                continue
        
            try:
                pct = stock_data_dict[sym].loc[date, 'score_percentile']
            except Exception as e:
                print('date',date)
                print('stock_data_dict[sym]',stock_data_dict[sym])
                print(f"⚠️ Percentile access error for {sym} on {date}: {e}")
                continue
            #pct = stock_data_dict[sym].at[date, 'score_percentile']
            pct = float(pct.iloc[0])  # 直接转 float，若 Series 会报错，但可提前暴露问题
            
            # 更新信号计数
            if pct < params.signal_threshold_low:
                signal_count[sym]['buy'] += 1
                signal_count[sym]['sell'] = 0  # 重置反向计数
            elif pct > params.signal_threshold_high:
                signal_count[sym]['sell'] += 1
                signal_count[sym]['buy'] = 0
            else:
                signal_count[sym]['buy'] = 0
                signal_count[sym]['sell'] = 0
            
            # 判断是否触发信号
            if signal_count[sym]['buy'] >= params.consecutive_days:
                buy_signals.append(sym)
            if signal_count[sym]['sell'] >= params.consecutive_days:
                sell_signals.append(sym)
        
        # --- 2.3 先处理卖出（释放资金）---
        for sym in sell_signals:
            if sym in portfolio['positions']:
                shares = portfolio['positions'][sym]['shares']
                price = stock_data_dict[sym].loc[date, 'Close'].iloc[0]
                proceeds = shares * price
                commission = proceeds * params.commission_rate
                portfolio['cash'] += proceeds - commission
                
                portfolio['trades'].append({
                    'date': date,
                    'symbol': sym,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'commission': commission
                })
                
                del portfolio['positions'][sym]
        
        # --- 2.4 再处理买入（使用当前可用现金）---
        if buy_signals:
            # 计算每只股票可分配的最大金额
            max_per_stock = params.total_capital * params.max_position_per_stock
            alloc_per_stock = min(portfolio['cash'] / len(buy_signals), max_per_stock)
            
            for sym in buy_signals:
                if sym not in portfolio['positions']:  # 避免重复买入
                    price = float(stock_data_dict[sym].loc[date, 'Close'].iloc[0])
                    amount_to_invest = float(min(alloc_per_stock, portfolio['cash']))
                    
                    if amount_to_invest > price:  # 至少买1股
                        shares = int(amount_to_invest // price)
                        cost = shares * price
                        commission = cost * params.commission_rate
                        total_cost = cost + commission
                        
                        if total_cost <= portfolio['cash']:
                            portfolio['cash'] -= total_cost
                            portfolio['positions'][sym] = {
                                'shares': shares,
                                'entry_price': price
                            }
                            
                            portfolio['trades'].append({
                                'date': date,
                                'symbol': sym,
                                'action': 'BUY',
                                'shares': shares,
                                'price': price,
                                'commission': commission
                            })
    
    # === 3. 转换历史记录为 DataFrame ===
    history_df = pd.DataFrame(portfolio['history']).set_index('date')
    return history_df, portfolio['trades'], portfolio['positions']

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

            def rolling_pct(x):
                return percentileofscore(x, x.iloc[-1], kind='mean') / 100.0

            df['score_percentile'] = df['obos_score'].rolling(
                window=60, min_periods=30
            ).apply(rolling_pct, raw=False)

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
        # 确保 dd_end >= dd_start
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

        latest = df.iloc[-1]
        return {
            'symbol': symbol,
            'price': float(latest['Close']),
            'rsi': float(latest['rsi']),
            'j': float(latest['j']),
            'bb_position': float(latest['bb_position']),
            'score': float(latest['obos_score']),
            'score_pct': float(latest['obos_score_pct']),  # ← 新增字段 either 'obos_score_pct' or 'obos_score_pct'
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

    # #####回测代码： ######

    #回测相关内容输出#
    params = StrategyParams(
        consecutive_days=2,
        signal_threshold_low=0.10,
        signal_threshold_high=0.90,
        max_position_per_stock=0.20,
        total_capital=1_000_000,
        commission_rate=0.001
    
    )
    

    # 运行回测
    result_backtest = run_full_backtest(symbols, start_str, end_str, params)
    
    # === 1. 绩效指标（使用 st.metric，美观且突出）===
    perf = result_backtest['performance']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Return", f"{perf['total_return']:.1%}")
    with col2:
        st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
    with col3:
        st.metric("Max Drawdown", f"{perf['max_drawdown']:.1%}")
    
    # === 2. 净值曲线图 ===
    st.subheader("📊 NAV Plot")
    st.pyplot(result_backtest['figure'])
    
    # === 3. 当前持仓（表格形式，更清晰）===
    st.subheader("💼 Current Holdings")
    final_positions = result_backtest['final_positions']

    stock_data_dict = result_backtest['stock_data_dict']
    
    if final_positions:
        # 转换为 DataFrame 便于展示
        pos_df = pd.DataFrame.from_dict(final_positions, orient='index')
        pos_df.index.name = 'Ticker'
        pos_df = pos_df.rename(columns={'shares': 'shares', 'entry_price': 'entry_price'})
        pos_df['current_price'] = pos_df.index.map(
            lambda sym: stock_data_dict[sym].iloc[-1]['Close'].iloc[0] 
            if sym in stock_data_dict else "N/A"
        )
        pos_df['current_MV'] = pos_df['shares'] * pos_df['current_price']
        # 计算总持仓市值（不含现金）
        total_position_value = pos_df['current_MV'].sum()
        
        # 计算占比（百分比）
        pos_df['position %'] = pos_df['current_MV'] / total_position_value if total_position_value > 0 else 0.0
        
        # 只显示需要的列
        display_df = pos_df[['shares', 'entry_price', 'current_price', 'position %']].copy()
    

        
        st.dataframe(display_df.style.format({
            'entry_price': "{:.2f}",
            'current_price': "{:.2f}",
            'position %': "{:.0f}"
        }))
    else:
        st.info("📭 回测结束时无持仓")
    
    # === 4. （可选）交易记录 ===
    # st.subheader("📜 最近交易记录")
    # trades_df = pd.DataFrame(result_backtest['trades'])
    # if not trades_df.empty:
    #     st.dataframe(trades_df.tail(10))
    
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
            'symbol', 'score', 'score_pct', 'td_buy_count', 'td_sell_count', 'rsi', 'j', 'bb_position'
        ]].copy()
        
        # 可选：重命名列，更清晰
        df_display.columns = [
            'Ticker', 'Score', 'Score in Percentile', 'TD Buy', 'TD Sell', 'RSI', 'KDJ-J', 'Bollinger%']
        


        # 使用背景色渐变突出评分 / 无matplot
        st.dataframe(df_display, use_container_width=True, height=500)


    
    with st.expander("Check the Score & Price Trend of Each Ticker. Apart from showing the technical score, we highlight the Overbought(red) / Oversold(green) area by using the rolling 60 days techncial score percentile (ranging from 0 to 1)"):
        for result in results:
            st.markdown(f"### {result['symbol']}")
            hist = result['history'].dropna()
            
            if len(hist) < 10:
                st.write("⚠️ Not Enough Data (Need at least 10 data points)")
                continue
            
            
            hist_plot = hist #这里全都取了，试试看
            
            fig, ax1 = plt.subplots(figsize=(10, 4))
            
            # 评分（左轴）
            ax1.plot(hist_plot.index, hist_plot['obos_score'], color='red', linewidth=1.5)
            ax1.set_ylabel('Technical Score ', color='red')
            ax1.tick_params(axis='y', labelcolor='red')
            ax1.set_ylim(0, 100)
            ax1.axhline(90, color='orange', linestyle='--', alpha=0.6)
            ax1.axhline(10, color='green', linestyle='--', alpha=0.6)
            ax1.grid(True, linestyle='--', alpha=0.3)

            dates = hist_plot.index
            
            # 填充超买区域（pct > 0.9）
            overbought = hist_plot['obos_score_pct'] > 0.9
            ax1.fill_between(dates, 0, 100, where=overbought, 
                             color='red', alpha=0.2, label='Overbought (pct > 0.9)')
            
            # 填充超卖区域（pct < 0.1）
            oversold = hist_plot['obos_score_pct'] < 0.1
            ax1.fill_between(dates, 0, 100, where=oversold, 
                             color='green', alpha=0.2, label='Oversold (pct < 0.1)')
                    
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

        
