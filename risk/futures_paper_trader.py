"""
Futures Paper Trader module to simulate trades and PnL correctly.
Handles multi-position portfolio safely.
"""
import json
import os
import time
import statistics
from datetime import datetime, date
from pathlib import Path

TRADES_FILE = Path(__file__).parent / "trades.json"

class FuturesPaperTrader:
    def __init__(
        self,
        initial_balance: float = 200.0,
        leverage: int = 10,
        risk_per_trade_pct: float = 0.02,
        max_daily_loss_pct: float = 0.10,
        circuit_breaker_losses: int = 3,
        circuit_breaker_cooldown: int = 6,
        time_stop_candles: int = 8,
        daily_profit_target: float | None = None,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.circuit_breaker_losses = circuit_breaker_losses
        self.circuit_breaker_cooldown = circuit_breaker_cooldown
        self.time_stop_candles = time_stop_candles

        # ACTIVE POSITIONS: Dict mapping symbol to position configuration
        self.positions: dict[str, dict] = {}

        # Stats tracking
        self.trade_history: list[dict] = []
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.daily_pnl = 0.0
        self.daily_pnl_date = str(date.today())
        self.daily_halted = False
        self.daily_profit_target = daily_profit_target

        # Load persisted trades
        self._load_history()

    def calculate_position_size(self, sl_distance_pct: float) -> float:
        if sl_distance_pct <= 0:
            return 0.0
        equity = self.balance + sum(p.get('unrealized_pnl', 0) for p in self.positions.values())
        risk_amount = equity * self.risk_per_trade_pct
        position_size = risk_amount / sl_distance_pct
        max_size = equity * self.leverage
        return min(position_size, max_size * 0.95)

    def get_max_position_size(self) -> float:
        equity = self.balance + sum(p.get('unrealized_pnl', 0) for p in self.positions.values())
        return max(equity * self.leverage, 0)

    def open_position(self, side: str, price: float, size_usd: float, tp_price: float = 0.0, sl_price: float = 0.0, grade: str = '', regime: str = '', symbol: str = '') -> bool:
        if symbol in self.positions:
            print(f"Already in a position for {symbol}. Close it first.")
            return False

        if self.cooldown_remaining > 0:
            return False

        self._check_daily_reset()
        if self.daily_halted:
            return False

        max_size = self.get_max_position_size()
        if size_usd > max_size:
            size_usd = max_size * 0.95
        if price <= 0:
            return False

        self.positions[symbol] = {
            'side': side,
            'entry_price': price,
            'position_size_usd': size_usd,
            'position_size_asset': size_usd / price,
            'original_size_usd': size_usd,
            'original_size_asset': size_usd / price,
            'unrealized_pnl': 0.0,
            'current_tp_price': tp_price,
            'current_sl_price': sl_price,
            'trailing_stop_active': False,
            'trailing_stop_price': 0.0,
            'highest_since_entry': price,
            'lowest_since_entry': price,
            'partial_taken': False,
            'partial_pnl_realized': 0.0,
            'candles_in_trade': 0,
            'entry_time': time.time(),
            'trade_grade': grade,
            'trade_regime': regime
        }

        margin_used = size_usd / self.leverage
        print(f"[OPEN] {side} {symbol} | notional ${size_usd:.2f} | margin ${margin_used:.2f} | entry {price:.2f} | TP {tp_price:.2f} | SL {sl_price:.2f}")
        return True

    def close_position(self, current_price: float, reason: str = "MANUAL", symbol: str = "") -> float:
        if symbol not in self.positions:
            return 0.0

        p = self.positions[symbol]
        self.update_pnl(current_price, symbol)

        remaining_pnl = p['unrealized_pnl']
        realized = remaining_pnl + p['partial_pnl_realized']
        self.balance += remaining_pnl
        self.balance = max(self.balance, 0)

        self._check_daily_reset()
        self.daily_pnl += realized

        if realized < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.circuit_breaker_losses:
                self.cooldown_remaining = self.circuit_breaker_cooldown
        else:
            self.consecutive_losses = 0

        if self.daily_pnl < 0 and abs(self.daily_pnl) > self.initial_balance * self.max_daily_loss_pct:
            self.daily_halted = True

        if self.daily_profit_target and self.daily_pnl >= self.daily_profit_target:
            self.daily_halted = True

        hold_time = time.time() - p['entry_time'] if p['entry_time'] else 0
        print(f"[CLOSE] {symbol} {p['side']} @ {current_price:.2f} | Reason: {reason} | PnL ${realized:.2f} | Balance ${self.balance:.2f}")
        # FIX BUG 7: Record trade BEFORE deleting from positions
        self._record_trade(current_price, realized, reason, hold_time, symbol)
        del self.positions[symbol]
        return realized

    def _check_partial_profit(self, current_price: float, atr_value: float, symbol: str):
        if symbol not in self.positions: return
        p = self.positions[symbol]
        if p['partial_taken'] or atr_value <= 0: return

        if p['side'] == 'LONG':
            profit_distance = current_price - p['entry_price']
            target = atr_value * 1.5
        else:
            profit_distance = p['entry_price'] - current_price
            target = atr_value * 1.5

        if profit_distance >= target:
            take_pct = 0.50
            take_size = p['position_size_asset'] * take_pct
            partial_pnl = take_size * profit_distance

            p['partial_pnl_realized'] += partial_pnl
            self.balance += partial_pnl

            p['position_size_asset'] *= (1 - take_pct)
            p['position_size_usd'] *= (1 - take_pct)
            p['current_sl_price'] = p['entry_price']
            p['partial_taken'] = True

    def update_pnl(self, current_price: float, symbol: str):
        if symbol not in self.positions: return
        p = self.positions[symbol]
        
        if p['side'] == 'LONG':
            p['unrealized_pnl'] = p['position_size_asset'] * (current_price - p['entry_price'])
            p['highest_since_entry'] = max(p['highest_since_entry'], current_price)
        elif p['side'] == 'SHORT':
            p['unrealized_pnl'] = p['position_size_asset'] * (p['entry_price'] - current_price)
            p['lowest_since_entry'] = min(p['lowest_since_entry'], current_price)

        if self._check_liquidation(current_price, symbol):
            liquidation_loss = -self.balance
            self.daily_pnl += liquidation_loss
            self._record_trade(current_price, liquidation_loss, "LIQUIDATION", time.time() - p['entry_time'], symbol)
            self.balance = 0.0
            del self.positions[symbol]

    def _check_liquidation(self, current_price: float, symbol: str) -> bool:
        if symbol not in self.positions: return False
        p = self.positions[symbol]
        
        liq_threshold = 1.0 / self.leverage
        if p['side'] == 'LONG':
            loss_pct = (p['entry_price'] - current_price) / p['entry_price']
        else:
            loss_pct = (current_price - p['entry_price']) / p['entry_price']
        return loss_pct >= liq_threshold

    def check_exit_conditions(self, current_price: float, atr_value: float = 0.0, symbol: str = "") -> str | None:
        if symbol not in self.positions: return None
        p = self.positions[symbol]

        self.update_pnl(current_price, symbol)
        p['candles_in_trade'] += 1

        self._check_partial_profit(current_price, atr_value, symbol)

        if p['current_sl_price'] > 0:
            if p['side'] == 'LONG' and current_price <= p['current_sl_price']: return "STOP_LOSS"
            if p['side'] == 'SHORT' and current_price >= p['current_sl_price']: return "STOP_LOSS"

        if p['current_tp_price'] > 0:
            if p['side'] == 'LONG' and current_price >= p['current_tp_price']: return "TAKE_PROFIT"
            if p['side'] == 'SHORT' and current_price <= p['current_tp_price']: return "TAKE_PROFIT"

        if atr_value > 0 and p['entry_price'] > 0:
            atr_pct = atr_value / p['entry_price']
            if p['side'] == 'LONG':
                profit_pct = (current_price - p['entry_price']) / p['entry_price']
                if profit_pct >= atr_pct:
                    p['trailing_stop_active'] = True
                    trail_dist = atr_value * 1.0  # FIXED: 1.0 ATR trail
                    new_trail = p['highest_since_entry'] - trail_dist
                    p['trailing_stop_price'] = max(p['trailing_stop_price'], new_trail)

                if p['trailing_stop_active'] and current_price <= p['trailing_stop_price']:
                    return "TRAILING_STOP"

            elif p['side'] == 'SHORT':
                profit_pct = (p['entry_price'] - current_price) / p['entry_price']
                if profit_pct >= atr_pct:
                    p['trailing_stop_active'] = True
                    trail_dist = atr_value * 1.0  # FIXED: 1.0 ATR trail
                    new_trail = p['lowest_since_entry'] + trail_dist
                    p['trailing_stop_price'] = min(
                        p['trailing_stop_price'] if p['trailing_stop_price'] > 0 else float('inf'),
                        new_trail
                    )
                if p['trailing_stop_active'] and current_price >= p['trailing_stop_price']:
                    return "TRAILING_STOP"

        if p['candles_in_trade'] >= self.time_stop_candles and atr_value > 0:
            profit = (current_price - p['entry_price']) if p['side'] == 'LONG' else (p['entry_price'] - current_price)
            if profit < (atr_value * 0.3):
                return "TIME_STOP"

        return None

    def tick_cooldown(self):
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def _check_daily_reset(self):
        today = str(date.today())
        if today != self.daily_pnl_date:
            self.daily_pnl = 0.0
            self.daily_pnl_date = today
            self.daily_halted = False

    def _record_trade(self, exit_price: float, pnl: float, reason: str, hold_time: float, symbol: str):
        p = self.positions[symbol]
        trade = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'side': p['side'],
            'entry': round(p['entry_price'], 4),
            'exit': round(exit_price, 4),
            'size_usd': round(p['original_size_usd'], 2),
            'pnl': round(pnl, 4),
            'balance_after': round(max(self.balance, 0), 4),
            'reason': reason,
            'partial_taken': p['partial_taken'],
            'partial_pnl': round(p['partial_pnl_realized'], 4),
            'hold_seconds': round(hold_time, 0),
            'candles_held': p['candles_in_trade'],
            'grade': p['trade_grade'],
            'regime': p['trade_regime'],
        }
        self.trade_history.append(trade)
        self._save_history()

    def _save_history(self):
        try:
            with open(TRADES_FILE, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except: pass

    def _load_history(self):
        if TRADES_FILE.exists():
            try:
                with open(TRADES_FILE, 'r') as f:
                    self.trade_history = json.load(f)
                
                # Recalculate balance from history
                if self.trade_history:
                    total_pnl = sum(t.get('pnl', 0.0) for t in self.trade_history)
                    self.balance = self.initial_balance + total_pnl
                    
                    # Recalculate daily PnL
                    today = str(date.today())
                    self.daily_pnl = sum(
                        t.get('pnl', 0.0) 
                        for t in self.trade_history 
                        if t.get('timestamp', '').startswith(today)
                    )
                    
                    # Recalculate consecutive losses
                    losses = 0
                    for t in reversed(self.trade_history):
                        if t.get('pnl', 0.0) < 0:
                            losses += 1
                        else:
                            break
                    self.consecutive_losses = losses
            except: self.trade_history = []

    def get_statistics(self) -> dict:
        trades = self.trade_history
        if not trades:
            return {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
                    'profit_factor': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                    'max_drawdown': 0.0, 'total_pnl': 0.0, 'sharpe': 0.0,
                    'sortino': 0.0, 'calmar': 0.0, 'expectancy': 0.0,
                    'avg_hold_s': 0.0, 'best_trade': 0.0, 'worst_trade': 0.0,
                    'grade_breakdown': {}}

        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown from equity curve
        eq = self.initial_balance
        peak = eq
        max_dd = 0
        for p in pnls:
            eq += p
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe (per-trade)
        avg_pnl = statistics.mean(pnls) if pnls else 0
        std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

        # Sortino (downside deviation only)
        neg_pnls = [p for p in pnls if p < 0]
        if len(neg_pnls) > 1:
            downside_std = statistics.stdev(neg_pnls)
            sortino = avg_pnl / downside_std if downside_std > 0 else sharpe
        else:
            sortino = sharpe

        # Calmar
        total_ret_pct = (sum(pnls) / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        calmar = total_ret_pct / (max_dd * 100) if max_dd > 0 else 0

        # Expectancy
        wr = len(wins) / len(trades) if trades else 0
        avg_w = statistics.mean(wins) if wins else 0
        avg_l = abs(statistics.mean(losses)) if losses else 0
        expectancy = (wr * avg_w) - ((1 - wr) * avg_l)

        # Avg hold time
        holds = [t.get('hold_seconds', 0) for t in trades if t.get('hold_seconds', 0) > 0]
        avg_hold = statistics.mean(holds) if holds else 0

        # Grade breakdown
        gb = {}
        for t in trades:
            g = t.get('grade', '?')
            if g not in gb:
                gb[g] = {'count': 0, 'pnl': 0, 'wins': 0}
            gb[g]['count'] += 1
            gb[g]['pnl'] += t['pnl']
            if t['pnl'] > 0:
                gb[g]['wins'] += 1

        return {
            'total_trades': len(trades), 'wins': len(wins), 'losses': len(losses),
            'win_rate': round(wr * 100, 1),
            'profit_factor': round(pf, 2),
            'avg_win': round(avg_w, 2),
            'avg_loss': round(statistics.mean(losses) if losses else 0, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'total_pnl': round(sum(pnls), 2),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'calmar': round(calmar, 3),
            'expectancy': round(expectancy, 2),
            'avg_hold_s': round(avg_hold, 0),
            'best_trade': round(max(pnls), 2) if pnls else 0,
            'worst_trade': round(min(pnls), 2) if pnls else 0,
            'grade_breakdown': gb,
        }

    @property
    def position(self):
        """Backward compat: returns first position side or None."""
        if self.positions:
            first = next(iter(self.positions.values()))
            return first['side']
        return None

    def get_state(self) -> dict:
        equity = self.balance + sum(p.get('unrealized_pnl', 0) for p in self.positions.values())
        stats = self.get_statistics()
        return {
            'balance': round(self.balance, 4),
            'leverage': self.leverage,
            'position': self.position,  # backward compat
            'positions': {k: {'side': v['side'], 'entry': v['entry_price'], 'size_usd': v['position_size_usd'], 'pnl': v.get('unrealized_pnl', 0.0), 'tp': v['current_tp_price'], 'sl': v['current_sl_price'], 'grade': v['trade_grade']} for k, v in self.positions.items()},
            'unrealized_pnl': round(sum(p.get('unrealized_pnl', 0) for p in self.positions.values()), 4),
            'equity': round(equity, 4),
            'roi_pct': round(((equity - self.initial_balance) / self.initial_balance) * 100, 2) if self.initial_balance else 0,
            'trade_count': stats['total_trades'],  # backward compat
            'win_rate': stats['win_rate'],  # backward compat
            'initial_balance': self.initial_balance,
            'consecutive_losses': self.consecutive_losses,
            'cooldown_remaining': self.cooldown_remaining,
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_halted': self.daily_halted,
            'stats': stats,
        }
