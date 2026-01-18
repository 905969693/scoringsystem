"""
Main Streamlit application for Scoring & Backtesting System.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from utils import get_watchlist_from_url, set_watchlist_to_url
from scoring import analyze_single_stock
from backtest import StrategyParams, run_full_backtest

# ========== Streamlit Configuration ==========
st.set_page_config(page_title="Scoring & Backtesting System", layout="wide")

# Custom CSS for font sizing
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

st.title("📊 Scoring & Backtesting System")
st.caption("0 = Extreme Oversold，100 = Extreme Overbought")

# Initialize watchlist (load from URL)
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = get_watchlist_from_url()

# Input area
col1, col2, col3 = st.columns([3, 1, 1])
ticker_list = "QQQ, SPY, TLT, GLD, USO"

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

# Watchlist management section
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

# Display current watchlist
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

# Analyze all stocks
if st.button("📊 Analyze All", type="primary"):
    # Prioritize watchlist if available
    if st.session_state.watchlist:
        symbols = st.session_state.watchlist
        st.info(f"Analyzing {len(symbols)} Tickers in the Watchlist")
    else:
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.error("Pls add tickers to watchlist, or simply input ticker(s)")
            st.stop()
    
    # Calculate date range
    start_date = pd.to_datetime(end_date) - pd.DateOffset(months=months_back)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Cache to session_state for backtesting
    st.session_state.last_symbols = symbols
    st.session_state.last_start_str = start_str
    st.session_state.last_end_str = end_str

    # Analyze all stocks
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
            'symbol', 'score', 'score_pct', 'consecutive_days', 'td_buy_count', 'td_sell_count', 'rsi', 'j', 'bb_position'
        ]].copy()
        df_display.columns = [
            'Ticker', 'Score', 'Score in Percentile', 'Consecutive Signal Days', 'TD Buy', 'TD Sell', 'RSI', 'KDJ-J', 'Bollinger%'
        ]
        st.subheader(f"📈 Result ( {len(results)} Stocks)")
        st.dataframe(df_display, use_container_width=True, height=500)

    # Individual stock charts
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

# Backtesting section
with st.expander("🔍 Run Full Backtest (Click to Expand)"):
    if 'last_symbols' not in st.session_state:
        st.warning("⚠️ Please click 'Analyze All' first to load stock data.")
    else:
        st.subheader("⚙️ Backtest Configuration")
        
        allow_shorting = st.checkbox("-Allow Shorting", value=True)
        
        # Signal threshold parameters
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
                0.51, 0.99, 0.95, step=0.01,
                help="Score percentile above this triggers SELL/SHORT"
            )
        
        # Other parameters
        col_bt1, col_bt2, col_bt3 = st.columns(3)
        with col_bt1:
            max_position_per_stock = st.slider("Max Position per Stock (%)", 5, 30, 20) / 100.0
        with col_bt2:
            consecutive_days = st.number_input("Consecutive Days for Signal", 1, 5, 3, step=1)
        with col_bt3:
            total_capital = st.number_input("Initial Capital ($)", 10_000, 10_000_000, 1_000_000, step=100_000)
        
        if st.button("🚀 Run Backtest", type="primary"):
            symbols = st.session_state.last_symbols
            start_str = st.session_state.last_start_str
            end_str = st.session_state.last_end_str
            
            # Build strategy parameters
            params = StrategyParams(
                consecutive_days=int(consecutive_days),
                signal_threshold_low=signal_threshold_low,
                signal_threshold_high=signal_threshold_high,
                max_position_per_stock=max_position_per_stock,
                total_capital=total_capital,
                commission_rate=0.001,
                allow_shorting=allow_shorting
            )
            
            # Run backtest
            with st.spinner("Running backtest..."):
                result_backtest = run_full_backtest(symbols, start_str, end_str, params)
            
            # Performance metrics
            perf = result_backtest['performance']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{perf['total_return']:.1%}")
            with col2:
                st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{perf['max_drawdown']:.1%}")
            
            # NAV chart
            st.pyplot(result_backtest['figure'])
            
            # Current holdings
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
            
            # Trade history
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
