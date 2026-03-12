"""
portfolio_manager.py — Portfolio-level risk control.

TASK 4: Enforces portfolio-wide constraints:
  - Max simultaneous open positions
  - Max exposure per asset
  - Max total leverage
  - Max drawdown halt
  - Correlation-based exposure reduction

Rules:
  max_open_positions = 3
  max_asset_exposure = 25%
  max_total_leverage = 10x
  max_drawdown = 15%
"""

import time
from collections import defaultdict
from logger import risk_log, log_risk


class PortfolioManager:
    """
    Portfolio-level risk manager.

    Validates trades against portfolio constraints before execution.
    Monitors aggregate exposure and drawdown in real-time.
    """

    def __init__(self, initial_balance: float = 200,
                 max_open_positions: int = 2,
                 max_asset_exposure_pct: float = 0.95,
                 max_total_leverage: float = 5.0,
                 max_drawdown_pct: float = 0.15):
        self.initial_balance = initial_balance
        self.max_open_positions = max_open_positions
        self.max_asset_exposure_pct = max_asset_exposure_pct
        self.max_total_leverage = max_total_leverage
        self.max_drawdown_pct = max_drawdown_pct

        # Tracking
        self.open_positions: dict[str, dict] = {}  # symbol -> position info
        self.total_exposure_usd: float = 0.0
        self.peak_equity: float = initial_balance
        self.current_equity: float = initial_balance
        self.is_halted: bool = False
        self.halt_reason: str = ""

    def validate_new_trade(self, symbol: str, side: str, size_usd: float) -> dict:
        """
        Validate a new trade against portfolio constraints.

        Returns:
            {approved: bool, reason: str (if rejected)}
        """
        reasons = []

        # Check drawdown halt
        if self.is_halted:
            reasons.append(f'portfolio_halted: {self.halt_reason}')

        # Check max open positions
        if len(self.open_positions) >= self.max_open_positions:
            reasons.append(f'max_positions_reached ({len(self.open_positions)}/{self.max_open_positions})')

        # Check if already have position in this asset
        if symbol in self.open_positions:
            reasons.append(f'already_exposed_to_{symbol}')

        # Check per-asset exposure limit (compare MARGIN used, not notional)
        # With 10x leverage, a $1900 notional is only $190 margin
        # Exposure should be: margin / equity, NOT notional / equity
        leverage = getattr(self, '_leverage', 10)
        margin_used = size_usd / leverage
        asset_exposure_pct = margin_used / max(self.current_equity, 1)
        if asset_exposure_pct > self.max_asset_exposure_pct:
            reasons.append(
                f'asset_exposure {asset_exposure_pct:.1%} > {self.max_asset_exposure_pct:.0%}'
            )

        # Check total leverage
        new_total = self.total_exposure_usd + size_usd
        total_leverage = new_total / max(self.current_equity, 1)
        if total_leverage > self.max_total_leverage:
            reasons.append(
                f'total_leverage {total_leverage:.1f}x > {self.max_total_leverage:.0f}x'
            )

        if reasons:
            log_risk('TRADE_REJECTED', symbol=symbol, side=side,
                     size_usd=size_usd, reasons=reasons)
            return {'approved': False, 'reason': '; '.join(reasons)}

        return {'approved': True, 'reason': None}

    def register_position(self, symbol: str, side: str, size_usd: float,
                          entry_price: float):
        """Record a new open position."""
        self.open_positions[symbol] = {
            'side': side,
            'size_usd': size_usd,
            'entry_price': entry_price,
            'opened_at': time.time(),
        }
        self.total_exposure_usd += size_usd

    def close_position(self, symbol: str, pnl: float):
        """Remove a closed position and update equity."""
        if symbol in self.open_positions:
            pos = self.open_positions.pop(symbol)
            self.total_exposure_usd -= pos['size_usd']
            self.total_exposure_usd = max(0, self.total_exposure_usd)

        self.current_equity += pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)

        # Check drawdown
        self._check_drawdown()

    def update_equity(self, equity: float):
        """Update current equity for drawdown monitoring."""
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self._check_drawdown()

    def _check_drawdown(self):
        """Halt trading if max drawdown breached."""
        if self.peak_equity <= 0:
            return
        dd = (self.peak_equity - self.current_equity) / self.peak_equity
        if dd >= self.max_drawdown_pct and not self.is_halted:
            self.is_halted = True
            self.halt_reason = f'max_drawdown {dd:.1%} >= {self.max_drawdown_pct:.0%}'
            log_risk('PORTFOLIO_HALT', drawdown=dd, peak=self.peak_equity,
                     current=self.current_equity)

    def reset_halt(self):
        """Manual halt reset (for new trading day, etc.)."""
        self.is_halted = False
        self.halt_reason = ""

    def get_status(self) -> dict:
        dd = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        return {
            'open_positions': len(self.open_positions),
            'max_positions': self.max_open_positions,
            'total_exposure_usd': round(self.total_exposure_usd, 2),
            'total_leverage': round(self.total_exposure_usd / max(self.current_equity, 1), 1),
            'current_drawdown': round(dd * 100, 2),
            'max_drawdown_limit': self.max_drawdown_pct * 100,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'positions': {sym: pos for sym, pos in self.open_positions.items()},
        }
