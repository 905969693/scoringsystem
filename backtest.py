"""
Backtesting engine for the scoring strategy.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_fetcher import fetch_stock_data, calculate_indicators, detect_trend
from scoring import calculate_obos_score


class StrategyParams:
    """Strategy parameters for backtesting."""
    def __init__(self,
                 lookback_window=60,
                 signal_threshold_low=0.10,
                 signal_threshold_high=0.95,
                 consecutive_days=3,
                 max_position_per_stock=0.15,
                 total_capital=1_000_000,
                 commission_rate=0.001,
                 risk_free_rate=0.02,
                 max_gross_exposure=2.0,
                 allow_shorting=False):
        self.lookback_window = lookback_window
        self.signal_threshold_low = signal_threshold_low
        self.signal_threshold_high = signal_threshold_high
        self.consecutive_days = consecutive_days
        self.max_position_per_stock = max_position_per_stock
        self.total_capital = total_capital
        self.commission_rate = commission_rate
        self.risk_free_rate = risk_free_rate
        self.max_gross_exposure = max_gross_exposure
        self.allow_shorting = allow_shorting


def prepare_stock_data_dict(symbols, start_date, end_date, interval="1d"):
    """
    Prepare stock data dictionary for backtesting.
    
    Returns:
        dict: {symbol: DataFrame with Close, obos_score, score_percentile, trend_direction, regime, trend_strength}
    """
    stock_data_dict = {}
    for sym in symbols:
        try:
            df = fetch_stock_data(sym, start=start_date, end=end_date, interval=interval)
            if df.empty or len(df) < 30:
                continue

            # Standardize index
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep='first')].sort_index()

            df = calculate_indicators(df)
            df['obos_score'] = calculate_obos_score(df)

            # Truncate NaN
            first_valid = df['obos_score'].first_valid_index()
            if pd.isna(first_valid):
                continue
            df = df.loc[first_valid:].copy()

            def rolling_pct_rank(x):
                return pd.Series(x).rank(pct=True).iloc[-1]

            df['score_percentile'] = df['obos_score'].rolling(
                window=60, min_periods=30
            ).apply(rolling_pct_rank, raw=False)

            df['score_percentile'] = pd.to_numeric(df['score_percentile'], errors='coerce')
            df['obos_score'] = pd.to_numeric(df['obos_score'], errors='coerce')

            first_valid_pct = df['score_percentile'].first_valid_index()
            if pd.isna(first_valid_pct):
                continue
            df = df.loc[first_valid_pct:].copy()

            # Calculate trend information for each date
            # For each date, use data up to that date to determine trend
            trend_directions = []
            regimes = []
            trend_strengths = []
            
            for date in df.index:
                df_upto_date = df.loc[:date].copy()
                trend_info = detect_trend(df_upto_date)
                trend_directions.append(trend_info['trend_direction'])
                regimes.append(trend_info['regime'])
                trend_strengths.append(trend_info['trend_strength'])
            
            df['trend_direction'] = trend_directions
            df['regime'] = regimes
            df['trend_strength'] = trend_strengths

            stock_data_dict[sym] = df[['Close', 'obos_score', 'score_percentile', 'trend_direction', 'regime', 'trend_strength']].copy()

        except Exception as e:
            print(f"⚠️ 跳过 {sym}: {str(e)[:60]}")
            continue

    return stock_data_dict


def run_backtest(stock_data_dict, params):
    """
    Execute backtest (supports long/short positions).
    
    Returns:
        portfolio_history: DataFrame (date, value, cash)
        trades_log: list of trade records
        positions: dict {symbol: {'shares': int, 'entry_price': float}}
    """
    # Align dates
    all_dates = set()
    for df in stock_data_dict.values():
        all_dates.update(df.index)
    all_dates = sorted(pd.to_datetime(list(all_dates)))
    symbols = list(stock_data_dict.keys())
    
    # Initialize signal counters
    signal_count = {sym: {'buy': 0, 'sell': 0} for sym in symbols}
    
    # Initialize portfolio
    portfolio = {
        'cash': float(params.total_capital),
        'positions': {},
        'history': [],
        'trades': []
    }
    
    # Main backtest loop
    for date in all_dates:
        if not isinstance(date, pd.Timestamp) or pd.isna(date):
            continue
        
        # Calculate current equity
        current_equity = portfolio['cash']
        for sym, pos in portfolio['positions'].items():
            if date in stock_data_dict[sym].index:
                price_val = stock_data_dict[sym].loc[date, 'Close']
                if isinstance(price_val, pd.Series):
                    price_val = price_val.iloc[-1]
                price = float(price_val)
                current_equity += pos['shares'] * price
        
        portfolio['history'].append({
            'date': date,
            'value': current_equity,
            'cash': portfolio['cash']
        })
        
        # Force close counter-trend positions in STRONG_TREND regime
        for sym in symbols:
            if date not in stock_data_dict[sym].index:
                continue
            
            if sym not in portfolio['positions']:
                continue
            
            try:
                regime_val = stock_data_dict[sym].loc[date, 'regime']
                trend_dir_val = stock_data_dict[sym].loc[date, 'trend_direction']
                
                if isinstance(regime_val, pd.Series):
                    regime_val = regime_val.iloc[-1]
                if isinstance(trend_dir_val, pd.Series):
                    trend_dir_val = trend_dir_val.iloc[-1]
                
                regime = str(regime_val) if pd.notna(regime_val) else 'RANGING'
                trend_dir = str(trend_dir_val) if pd.notna(trend_dir_val) else 'RANGE'
            except Exception:
                # If trend data is missing, skip force-close (backward compatible)
                continue
            
            if regime == 'STRONG_TREND':
                position = portfolio['positions'][sym]
                shares = position['shares']
                
                # Force close shorts in uptrend
                if trend_dir == 'UPTREND' and shares < 0:
                    price_val = stock_data_dict[sym].loc[date, 'Close']
                    if isinstance(price_val, pd.Series):
                        price_val = price_val.iloc[-1]
                    price = float(price_val)
                    short_shares = -shares
                    cost = short_shares * price
                    commission = cost * params.commission_rate
                    portfolio['cash'] -= cost + commission
                    portfolio['trades'].append({
                        'date': date, 'symbol': sym, 'action': 'FORCE_CLOSE_SHORT',
                        'shares': short_shares, 'price': price, 'commission': commission
                    })
                    del portfolio['positions'][sym]
                
                # Force close longs in downtrend
                elif trend_dir == 'DOWNTREND' and shares > 0:
                    price_val = stock_data_dict[sym].loc[date, 'Close']
                    if isinstance(price_val, pd.Series):
                        price_val = price_val.iloc[-1]
                    price = float(price_val)
                    proceeds = shares * price
                    commission = proceeds * params.commission_rate
                    portfolio['cash'] += proceeds - commission
                    portfolio['trades'].append({
                        'date': date, 'symbol': sym, 'action': 'FORCE_CLOSE_LONG',
                        'shares': shares, 'price': price, 'commission': commission
                    })
                    del portfolio['positions'][sym]
        
        # Check signals
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
            except Exception:
                continue
            
            # Update signal counts
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
        
        # Filter out counter-trend signals in STRONG_TREND regime
        filtered_buy_signals = []
        filtered_sell_signals = []
        
        for sym in buy_signals:
            if date not in stock_data_dict[sym].index:
                continue
            
            try:
                regime_val = stock_data_dict[sym].loc[date, 'regime']
                trend_dir_val = stock_data_dict[sym].loc[date, 'trend_direction']
                
                if isinstance(regime_val, pd.Series):
                    regime_val = regime_val.iloc[-1]
                if isinstance(trend_dir_val, pd.Series):
                    trend_dir_val = trend_dir_val.iloc[-1]
                
                regime = str(regime_val) if pd.notna(regime_val) else 'RANGING'
                trend_dir = str(trend_dir_val) if pd.notna(trend_dir_val) else 'RANGE'
            except Exception:
                # If trend data is missing, allow signal (backward compatible)
                filtered_buy_signals.append(sym)
                continue
            
            # Block new longs in strong downtrend
            if regime == 'STRONG_TREND' and trend_dir == 'DOWNTREND':
                continue  # Skip this buy signal
            
            filtered_buy_signals.append(sym)
        
        for sym in sell_signals:
            if date not in stock_data_dict[sym].index:
                continue
            
            try:
                regime_val = stock_data_dict[sym].loc[date, 'regime']
                trend_dir_val = stock_data_dict[sym].loc[date, 'trend_direction']
                
                if isinstance(regime_val, pd.Series):
                    regime_val = regime_val.iloc[-1]
                if isinstance(trend_dir_val, pd.Series):
                    trend_dir_val = trend_dir_val.iloc[-1]
                
                regime = str(regime_val) if pd.notna(regime_val) else 'RANGING'
                trend_dir = str(trend_dir_val) if pd.notna(trend_dir_val) else 'RANGE'
            except Exception:
                # If trend data is missing, allow signal (backward compatible)
                filtered_sell_signals.append(sym)
                continue
            
            # Block new shorts in strong uptrend
            if regime == 'STRONG_TREND' and trend_dir == 'UPTREND':
                continue  # Skip this sell signal
            
            filtered_sell_signals.append(sym)
        
        # Use filtered signals
        buy_signals = filtered_buy_signals
        sell_signals = filtered_sell_signals
        
        # Close positions (reverse signals)
        # Close long positions
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
        
        # Close short positions
        for sym in buy_signals:
            if sym in portfolio['positions'] and portfolio['positions'][sym]['shares'] < 0:
                short_shares = -portfolio['positions'][sym]['shares']
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
        
        # Open/add positions
        # Handle long positions
        if buy_signals:
            for sym in buy_signals:
                if sym in portfolio['positions'] and portfolio['positions'][sym]['shares'] < 0:
                    continue
                
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
        
        # Handle short positions
        if params.allow_shorting and sell_signals:
            for sym in sell_signals:
                if sym in portfolio['positions'] and portfolio['positions'][sym]['shares'] > 0:
                    continue
                
                price_val = stock_data_dict[sym].loc[date, 'Close']
                if isinstance(price_val, pd.Series):
                    price_val = price_val.iloc[-1]
                price = float(price_val)
                
                max_allowed_value = params.max_position_per_stock * current_equity
                current_short_value = 0.0
                if sym in portfolio['positions']:
                    current_short_value = -portfolio['positions'][sym]['shares'] * price
                
                remaining_capacity = max_allowed_value - current_short_value
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
    
    history_df = pd.DataFrame(portfolio['history']).set_index('date')
    return history_df[['value', 'cash']], portfolio['trades'], portfolio['positions']


def calculate_performance(portfolio_history, params):
    """Calculate performance metrics."""
    nav = portfolio_history['value'] / params.total_capital
    total_return = nav.iloc[-1] - 1.0
    
    daily_returns = nav.pct_change().dropna()
    annualized_return = daily_returns.mean() * 252
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (annualized_return - params.risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
    
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
    """Plot performance chart with drawdown."""
    nav = perf_result['nav']
    _, dd_start, dd_end = perf_result['drawdown_period']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(nav.index, nav, color='blue', label='Portfolio NAV')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8)
    
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
    
    ax.set_title(f"{title}\n"
                 f"Total Return: {perf_result['total_return']:.1%} | "
                 f"Sharpe: {perf_result['sharpe_ratio']:.2f} | "
                 f"Max DD: {perf_result['max_drawdown']:.1%}")
    ax.set_ylabel("Normalized Value (Base=1.0)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


def run_full_backtest(symbols, start_date, end_date, params=None):
    """
    Complete backtest workflow: data preparation → backtest → performance → return results.
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
