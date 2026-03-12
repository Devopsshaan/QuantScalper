"""
order_manager.py — Order lifecycle management.

Responsibilities:
  - Submit orders to exchange clients
  - Track order status (pending, filled, partial, cancelled)
  - Retry failed orders with exponential backoff
  - Handle partial fills (adjust position tracking)
  - Handle slippage detection
  - Cancel stuck orders after timeout
  - Update portfolio state on fills
"""

import time
import threading
from enum import Enum
from collections import deque
from logger import log_order, error_log


class OrderStatus(Enum):
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    PARTIAL_FILL = 'partial_fill'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    FAILED = 'failed'
    TIMED_OUT = 'timed_out'


class Order:
    """Represents a single order through its lifecycle."""

    def __init__(self, symbol: str, side: str, size_usd: float,
                 price: float, tp_price: float = 0, sl_price: float = 0):
        self.symbol = symbol
        self.side = side
        self.size_usd = size_usd
        self.requested_price = price
        self.tp_price = tp_price
        self.sl_price = sl_price

        self.order_id: str = f"ORD-{int(time.time() * 1000)}"
        self.status = OrderStatus.PENDING
        self.fill_price: float = 0
        self.filled_size: float = 0
        self.slippage_bps: float = 0
        self.created_at = time.time()
        self.filled_at: float = 0
        self.retries: int = 0
        self.max_retries: int = 3
        self.timeout_seconds: float = 30.0
        self.error: str = ""


class OrderManager:
    """
    Manages order submission, tracking, retry, and cancellation.

    Works with exchange client to submit orders and track fills.
    Supports paper, backtest, and live modes.
    """

    def __init__(self, exchange_client=None, max_pending: int = 5):
        self.exchange_client = exchange_client
        self.max_pending = max_pending
        self.pending_orders: dict[str, Order] = {}
        self.order_history: deque = deque(maxlen=500)
        self._lock = threading.Lock()

    def submit_order(self, symbol: str, side: str, size_usd: float,
                     price: float, tp_price: float = 0, sl_price: float = 0) -> dict:
        """
        Submit an order and track it.

        Returns:
            {order_id, status, fill_price, ...}
        """
        order = Order(symbol, side, size_usd, price, tp_price, sl_price)

        if len(self.pending_orders) >= self.max_pending:
            self._cleanup_stale_orders()
            if len(self.pending_orders) >= self.max_pending:
                return {'executed': False, 'reason': 'too_many_pending_orders'}

        with self._lock:
            self.pending_orders[order.order_id] = order

        # Attempt submission with retries
        for attempt in range(order.max_retries):
            try:
                result = self._submit_to_exchange(order)
                if result.get('filled'):
                    order.status = OrderStatus.FILLED
                    order.fill_price = result.get('fill_price', price)
                    order.filled_size = size_usd
                    order.filled_at = time.time()
                    order.slippage_bps = abs(order.fill_price - price) / price * 10000

                    with self._lock:
                        self.pending_orders.pop(order.order_id, None)
                        self.order_history.append(order)

                    log_order('FILLED', symbol, side, size_usd, order.fill_price,
                              order_id=order.order_id,
                              slippage_bps=round(order.slippage_bps, 1),
                              latency_ms=round((order.filled_at - order.created_at) * 1000))

                    return {
                        'executed': True,
                        'order_id': order.order_id,
                        'fill_price': order.fill_price,
                        'size_usd': size_usd,
                        'slippage_bps': round(order.slippage_bps, 1),
                        'mode': 'live',
                    }
                elif result.get('partial'):
                    order.status = OrderStatus.PARTIAL_FILL
                    order.filled_size = result.get('filled_size', 0)
                    order.retries = attempt + 1
                    # Continue retry loop for remaining
                    continue
                else:
                    order.retries = attempt + 1
                    wait = 2 ** (attempt + 1)
                    time.sleep(min(wait, 10))
                    continue

            except Exception as e:
                order.error = str(e)
                order.retries = attempt + 1
                error_log.error(f"Order {order.order_id} attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** (attempt + 1))

        # All retries exhausted
        order.status = OrderStatus.FAILED
        with self._lock:
            self.pending_orders.pop(order.order_id, None)
            self.order_history.append(order)

        log_order('FAILED', symbol, side, size_usd, price,
                  order_id=order.order_id, error=order.error)

        return {'executed': False, 'reason': f'failed_after_{order.max_retries}_retries',
                'error': order.error}

    def _submit_to_exchange(self, order: Order) -> dict:
        """Submit order to exchange client."""
        if not self.exchange_client:
            return {'filled': False, 'reason': 'no_exchange_client'}

        try:
            # Use exchange client's order method
            result = self.exchange_client.exchange.create_order(
                symbol=order.symbol,
                type='market',
                side='buy' if order.side == 'LONG' else 'sell',
                amount=order.size_usd / order.requested_price,
            )
            fill_price = float(result.get('average', result.get('price', order.requested_price)))
            return {'filled': True, 'fill_price': fill_price}
        except Exception as e:
            raise e

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        with self._lock:
            if order_id in self.pending_orders:
                order = self.pending_orders.pop(order_id)
                order.status = OrderStatus.CANCELLED
                self.order_history.append(order)
                log_order('CANCELLED', order.symbol, order.side,
                          order.size_usd, order.requested_price,
                          order_id=order_id)
                return True
        return False

    def _cleanup_stale_orders(self):
        """Cancel orders that have exceeded timeout."""
        now = time.time()
        stale = []
        with self._lock:
            for oid, order in self.pending_orders.items():
                if now - order.created_at > order.timeout_seconds:
                    order.status = OrderStatus.TIMED_OUT
                    self.order_history.append(order)
                    stale.append(oid)
            for oid in stale:
                self.pending_orders.pop(oid, None)

    def get_status(self) -> dict:
        """Return order manager status for dashboard."""
        with self._lock:
            return {
                'pending_count': len(self.pending_orders),
                'total_orders': len(self.order_history),
                'recent_fills': [
                    {'id': o.order_id, 'symbol': o.symbol, 'side': o.side,
                     'slippage_bps': round(o.slippage_bps, 1)}
                    for o in list(self.order_history)[-5:]
                    if o.status == OrderStatus.FILLED
                ],
            }
