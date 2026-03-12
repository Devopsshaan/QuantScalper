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

        size_usd = sizing_result.get('size_usd', 0)
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
        if self.mode == TradingMode.PAPER:
            return self._execute_paper(signal, symbol, price, size_usd,
                                       tp_price, sl_price, grade, regime)
        elif self.mode == TradingMode.BACKTEST:
            return self._execute_backtest(signal, symbol, price, size_usd,
                                          tp_price, sl_price, grade, regime)
        elif self.mode == TradingMode.LIVE:
            return self._execute_live(signal, symbol, price, size_usd,
                                      tp_price, sl_price, grade, regime)

        return {'executed': False, 'reason': 'unknown_mode'}

    def _execute_paper(self, signal, symbol, price, size_usd,
                       tp_price, sl_price, grade, regime) -> dict:
        """Paper trading: simulated fill with slippage."""
        if not self.paper_trader:
            return {'executed': False, 'reason': 'no_paper_trader'}

        # Simulate slippage (0.01-0.03% on crypto)
        slippage_pct = 0.0002  # 2 bps average
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
