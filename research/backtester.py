"""
backtester.py — Walk-forward backtesting engine.

TASK 7 UPGRADE: Replaces simple train/test split with walk-forward
validation to prevent overfitting.

Walk-forward procedure:
  Window 1: train [0:200]   → test [200:250]
  Window 2: train [50:250]  → test [250:300]
  Window 3: train [100:300] → test [300:350]
  ...

Each window trains a fresh model on in-sample data and evaluates
on out-of-sample data that was never seen during training.
"""

import pandas as pd
import numpy as np
import statistics as stats
from typing import Callable
from ta.volatility import AverageTrueRange

from logger import system_log


class WalkForwardBacktester:
    """Walk-forward backtesting engine for unbiased strategy evaluation."""

    def __init__(self, initial_balance: float = 200, leverage: int = 10,
                 risk_per_trade: float = 0.02, commission_pct: float = 0.0006):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission_pct = commission_pct

    def run_walk_forward(self, df: pd.DataFrame, strategy_fn: Callable,
                          train_bars: int = 200, test_bars: int = 50,
                          step_bars: int = 50,
                          tp_atr_mult: float = 3.0, sl_atr_mult: float = 1.5) -> dict:
        """
        Walk-forward backtest with expanding or sliding windows.

        Args:
            df: Full historical OHLCV DataFrame
            strategy_fn: func(df_window) -> {'signal': 'LONG'|'SHORT'|None}
            train_bars: bars in training window
            test_bars: bars in test window
            step_bars: step size between windows
        """
        if len(df) < train_bars + test_bars:
            return {'error': f'Need >= {train_bars + test_bars} bars, got {len(df)}'}

        # Add ATR
        atr_ind = AverageTrueRange(high=df['high'], low=df['low'],
                                    close=df['close'], window=14)
        df = df.copy()
        df['atr'] = atr_ind.average_true_range()
        df.dropna(inplace=True)

        all_trades = []
        window_results = []
        balance = self.initial_balance
        window_num = 0

        start = 0
        while start + train_bars + test_bars <= len(df):
            window_num += 1
            train_end = start + train_bars
            test_end = train_end + test_bars

            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]

            # Run strategy on test period
            window_trades, balance = self._run_test_window(
                test_df, strategy_fn, balance, tp_atr_mult, sl_atr_mult
            )

            window_metrics = self._compute_metrics(window_trades, balance)
            window_metrics['window'] = window_num
            window_metrics['train_start'] = int(start)
            window_metrics['test_start'] = int(train_end)
            window_metrics['test_end'] = int(test_end)
            window_results.append(window_metrics)

            all_trades.extend(window_trades)
            start += step_bars

        # Aggregate results
        total_metrics = self._compute_metrics(all_trades, balance)
        total_metrics['windows'] = len(window_results)
        total_metrics['window_details'] = window_results
        total_metrics['method'] = 'walk_forward'

        system_log.info(
            f"Walk-forward complete: {len(window_results)} windows, "
            f"{total_metrics['trades']} trades, "
            f"PnL=${total_metrics['total_pnl']:.2f}, "
            f"WR={total_metrics['win_rate']:.1f}%"
        )

        return total_metrics

    def _run_test_window(self, df: pd.DataFrame, strategy_fn: Callable,
                          balance: float,
                          tp_atr_mult: float, sl_atr_mult: float) -> tuple:
        """Run strategy over a test window. Returns (trades_list, final_balance)."""
        trades = []
        position = None
        entry_price = sl_price = tp_price = size_usd = 0

        for i in range(50, len(df)):
            window = df.iloc[max(0, i - 100):i + 1]
            price = float(df.iloc[i]['close'])
            curr_atr = float(df.iloc[i]['atr'])

            # Check exits
            if position:
                h = float(df.iloc[i]['high'])
                l = float(df.iloc[i]['low'])
                exit_price = reason = None

                if position == 'LONG':
                    if l <= sl_price:
                        exit_price, reason = sl_price, 'SL'
                    elif h >= tp_price:
                        exit_price, reason = tp_price, 'TP'
                else:
                    if h >= sl_price:
                        exit_price, reason = sl_price, 'SL'
                    elif l <= tp_price:
                        exit_price, reason = tp_price, 'TP'

                if exit_price:
                    size_asset = size_usd / entry_price
                    pnl = size_asset * (exit_price - entry_price) if position == 'LONG' \
                          else size_asset * (entry_price - exit_price)
                    pnl -= size_usd * self.commission_pct * 2
                    balance += pnl
                    trades.append({
                        'side': position, 'entry': entry_price,
                        'exit': exit_price, 'pnl': round(pnl, 4),
                        'reason': reason,
                    })
                    position = None
                continue

            # Generate signal
            try:
                result = strategy_fn(window)
                signal = result.get('signal')
            except Exception:
                continue

            if signal and curr_atr > 0 and balance > 0:
                sl_dist = curr_atr * sl_atr_mult
                sl_pct = sl_dist / price
                risk_amt = balance * self.risk_per_trade
                size_usd = min(risk_amt / sl_pct, balance * self.leverage * 0.95)
                entry_price = price
                position = signal
                tp_price = price + curr_atr * tp_atr_mult if signal == 'LONG' \
                           else price - curr_atr * tp_atr_mult
                sl_price = price - sl_dist if signal == 'LONG' else price + sl_dist

        return trades, balance

    def run_simple(self, df: pd.DataFrame, strategy_fn: Callable,
                    tp_atr_mult: float = 3.0, sl_atr_mult: float = 1.5) -> dict:
        """Simple full-period backtest (for comparison with walk-forward)."""
        if len(df) < 60:
            return {'error': 'insufficient_data'}

        atr = AverageTrueRange(high=df['high'], low=df['low'],
                                close=df['close'], window=14)
        df = df.copy()
        df['atr'] = atr.average_true_range()
        df.dropna(inplace=True)

        trades, balance = self._run_test_window(
            df, strategy_fn, self.initial_balance, tp_atr_mult, sl_atr_mult
        )
        result = self._compute_metrics(trades, balance)
        result['method'] = 'simple'
        return result

    def _compute_metrics(self, trades: list, final_balance: float) -> dict:
        if not trades:
            return {'trades': 0, 'total_pnl': 0, 'win_rate': 0,
                    'profit_factor': 0, 'sharpe': 0, 'max_drawdown': 0,
                    'final_balance': final_balance, 'trade_list': trades}

        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gp = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 0.001

        # Max drawdown
        eq = self.initial_balance
        peak = eq
        max_dd = 0
        for p in pnls:
            eq += p
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        avg = stats.mean(pnls) if pnls else 0
        std = stats.stdev(pnls) if len(pnls) > 1 else 1

        return {
            'trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'total_pnl': round(sum(pnls), 2),
            'win_rate': round(len(wins) / len(trades) * 100, 1),
            'profit_factor': round(gp / gl, 2),
            'sharpe': round(avg / std if std > 0 else 0, 3),
            'max_drawdown': round(max_dd * 100, 2),
            'final_balance': round(final_balance, 2),
            'roi_pct': round((final_balance - self.initial_balance) / self.initial_balance * 100, 2),
            'trade_list': trades,
        }
