"""
execution_engine.py — Trade execution orchestration layer.

TASK 1: Routes signals through quality filter → risk validation →
order manager → exchange client → confirmation.

Handles the full lifecycle:
  signal → validate → size → submit → monitor → confirm/cancel

Supports three modes:
  - BACKTEST: instant fill at close price
  - PAPER: simulated fill with slippage model
  - LIVE: real exchange order submission
"""

import time
from enum import Enum
from logger import log_order, error_log, system_log


class TradingMode(Enum):
    BACKTEST = 'backtest'
    PAPER = 'paper'
    LIVE = 'live'


class ExecutionEngine:
    """
    Orchestrates signal-to-order pipeline.

    Signal → SignalQualityFilter → PortfolioManager → PositionSizer →
    OrderManager → Exchange (or Paper Trader) → Confirmation
    """

    def __init__(self, mode: TradingMode = TradingMode.PAPER,
                 paper_trader=None, order_manager=None, portfolio_manager=None):
        self.mode = mode
        self.paper_trader = paper_trader
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.pending_orders: list = []
        self.executed_orders: list = []
        self.last_execution: dict = {}

    GRADE_ORDER = {'A': 4, 'B': 3, 'C': 2, 'D': 1}

    def execute_signal(self, signal: str, symbol: str, price: float,
                       quality_result: dict, sizing_result: dict,
                       tp_price: float = 0, sl_price: float = 0,
                       grade: str = '', regime: str = '',
                       min_grade: str = 'C') -> dict:
        """
        Execute a validated signal through the appropriate mode.

        Returns execution result dict.
        """
        if not quality_result.get('passed'):
            return {'executed': False, 'reason': 'quality_filter_rejected'}

        # FIX: enforce min_grade so config's "min_trade_grade" is respected
        trade_grade = quality_result.get('grade', grade or 'D')
        if self.GRADE_ORDER.get(trade_grade, 0) < self.GRADE_ORDER.get(min_grade, 0):
            return {'executed': False, 'reason': f'grade_{trade_grade}_below_min_{min_grade}'}

        # FIX: Protect against extreme sell pressure
        of_pressure = quality_result.get('breakdown', {}).get('order_flow', 0.5)
        # If signal is LONG and order flow breakdown score implies extreme sell imbalance
        if signal == 'LONG' and of_pressure < 0.3:
            return {'executed': False, 'reason': 'rejected_due_to_heavy_sell_pressure'}

        size_usd = sizing_result.get('size_usd', 0)

        # ═══ CRITICAL FIX: Hard cap max loss per trade ═══
        # This prevents the $11.24 SOL wipeout from ever happening again.
        # Max loss = 2% of equity. If position × SL distance > max_loss → shrink position.
        MAX_LOSS_PCT = 0.02  # 2% of equity
        if sl_price > 0 and price > 0:
            sl_dist_pct = abs(sl_price - price) / price
            if sl_dist_pct > 0:
                # What would max loss be at this size?
                potential_loss = size_usd * sl_dist_pct
                equity = getattr(self.paper_trader, 'balance', 200)
                max_loss = equity * MAX_LOSS_PCT
                if potential_loss > max_loss:
                    # Shrink position to cap loss
                    old_size = size_usd
                    size_usd = max_loss / sl_dist_pct
                    sizing_result['size_usd'] = round(size_usd, 2)
                    system_log.info(
                        f'LOSS CAP: Reduced ${old_size:.0f} → ${size_usd:.0f} '
                        f'(SL {sl_dist_pct*100:.2f}% × ${old_size:.0f} = ${potential_loss:.2f} > max ${max_loss:.2f})')

        # Removed artificial RR widening that pushed TP beyond structural barriers

        system_log.info(f'Execution attempt symbol={symbol} signal={signal} mode={self.mode.value} size={size_usd:.2f}')
        if size_usd <= 0:
            return {'executed': False, 'reason': 'zero_size'}

        # Portfolio-level risk check
        if self.portfolio_manager:
            risk_check = self.portfolio_manager.validate_new_trade(
                symbol=symbol, side=signal, size_usd=size_usd
            )
            if not risk_check.get('approved'):
                log_order('RISK_REJECT', symbol, signal, size_usd, price,
                          reason=risk_check.get('reason', 'portfolio_risk'))
                return {'executed': False, 'reason': risk_check.get('reason')}

        # Mode-specific execution
        # Fetch spread from context via market feed logic if possible, defaulting to 2 bps
        spread_bps = 2.0
        try:
            if isinstance(quality_result, dict) and 'breakdown' in quality_result:
                # the pipeline could pass full orderbook data here eventually, but for now we pull safely
                # If ob dict is bound somewhere we could extract it. 
                pass
        except:
            pass

        if self.mode == TradingMode.PAPER:
            return self._execute_paper(signal, symbol, price, size_usd,
                                       tp_price, sl_price, grade, regime, spread_bps)
        elif self.mode == TradingMode.BACKTEST:
            return self._execute_backtest(signal, symbol, price, size_usd,
                                          tp_price, sl_price, grade, regime)
        elif self.mode == TradingMode.LIVE:
            return self._execute_live(signal, symbol, price, size_usd,
                                      tp_price, sl_price, grade, regime)

        return {'executed': False, 'reason': 'unknown_mode'}

    def _execute_paper(self, signal, symbol, price, size_usd,
                       tp_price, sl_price, grade, regime,
                       spread_bps: float = 0.0) -> dict:
        """Paper trading: simulated fill with slippage based on real spread."""
        if not self.paper_trader:
            return {'executed': False, 'reason': 'no_paper_trader'}

        # Simulate slippage: base latency + half spread
        base_slippage = 0.00015  # 1.5 bps latency impact
        spread_impact = (spread_bps / 10000.0) / 2.0  # Pay the spread to cross the book
        slippage_pct = base_slippage + spread_impact
        
        if signal == 'LONG':
            fill_price = price * (1 + slippage_pct)
        else:
            fill_price = price * (1 - slippage_pct)

        success = self.paper_trader.open_position(
            signal, fill_price, size_usd,
            tp_price=tp_price, sl_price=sl_price,
            grade=grade, regime=regime, symbol=symbol,
        )

        if success:
            log_order('PAPER_FILL', symbol, signal, size_usd, fill_price,
                      slippage_bps=round(slippage_pct * 10000, 1),
                      grade=grade)
            self.last_execution = {
                'executed': True, 'mode': 'paper',
                'fill_price': fill_price, 'size_usd': size_usd,
                'slippage_bps': round(slippage_pct * 10000, 1),
            }
        else:
            log_order('EXEC_REJECT', symbol, signal, size_usd, fill_price,
                      reason=f'paper_trader_rejected_grade_{grade}')
            self.last_execution = {'executed': False, 'reason': 'paper_trader_rejected'}

        return self.last_execution

    def _execute_backtest(self, signal, symbol, price, size_usd,
                          tp_price, sl_price, grade, regime) -> dict:
        """Backtest: instant fill at close price, no slippage."""
        if not self.paper_trader:
            return {'executed': False, 'reason': 'no_paper_trader'}

        success = self.paper_trader.open_position(
            signal, price, size_usd,
            tp_price=tp_price, sl_price=sl_price,
            grade=grade, regime=regime, symbol=symbol,
        )
        self.last_execution = {
            'executed': success, 'mode': 'backtest',
            'fill_price': price, 'size_usd': size_usd,
        }
        return self.last_execution

    def _execute_live(self, signal, symbol, price, size_usd,
                      tp_price, sl_price, grade, regime) -> dict:
        """
        Live execution via OrderManager.
        Routes to exchange with retry/cancel logic.
        """
        if not self.order_manager:
            return {'executed': False, 'reason': 'no_order_manager'}

        try:
            result = self.order_manager.submit_order(
                symbol=symbol, side=signal,
                size_usd=size_usd, price=price,
                tp_price=tp_price, sl_price=sl_price,
            )
            log_order('LIVE_SUBMIT', symbol, signal, size_usd, price,
                      order_id=result.get('order_id'))
            self.last_execution = result
            return result
        except Exception as e:
            error_log.error(f"Live execution failed {symbol}: {e}")
            return {'executed': False, 'reason': str(e)}
