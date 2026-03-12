from collections import deque 
from datetime import datetime, timedelta 
 
from logger import log_risk, system_log 
 
class GlobalRiskGuard: 
    def __init__(self, max_daily_loss=10.0, max_drawdown_pct=15.0, max_open_positions=2, max_consecutive_losses=5, max_trades_per_hour=None): 
        self.max_daily_loss = float(max_daily_loss) 
        self.max_drawdown_pct = float(max_drawdown_pct) 
        self.max_open_positions = int(max_open_positions) 
        self.max_consecutive_losses = int(max_consecutive_losses) 
        self.max_trades_per_hour = int(max_trades_per_hour) if max_trades_per_hour else None 
        self.hard_stop = False 
        self.can_open_new_positions = True 
        self.reason = '' 
        self.trade_timestamps = deque(maxlen=200) 
 
    def sync_trade_history(self, trade_history): 
        self.trade_timestamps.clear() 
        for trade in trade_history[-200:]: 
            timestamp = trade.get('timestamp') 
            if not timestamp: 
                continue 
            try: 
                self.trade_timestamps.append(datetime.fromisoformat(str(timestamp).replace('Z', ''))) 
            except ValueError: 
                continue 
 
    def evaluate(self, trader_state, portfolio_status, trade_history=None): 
        if trade_history is not None: 
            self.sync_trade_history(trade_history) 
        self.hard_stop = False 
        self.can_open_new_positions = True 
        self.reason = '' 
        daily_pnl = float(trader_state.get('daily_pnl', 0)) 
        current_drawdown = float(portfolio_status.get('current_drawdown', 0)) 
        open_positions = int(portfolio_status.get('open_positions', 0)) 
        consecutive_losses = int(trader_state.get('consecutive_losses', 0)) 
        if daily_pnl <= -self.max_daily_loss: 
            self._trigger('MAX_DAILY_LOSS', f'daily_pnl={daily_pnl}') 
        elif current_drawdown >= self.max_drawdown_pct: 
            self._trigger('MAX_DRAWDOWN', f'drawdown={current_drawdown}') 
        elif consecutive_losses >= self.max_consecutive_losses: 
            self._trigger('MAX_CONSECUTIVE_LOSSES', f'losses={consecutive_losses}') 
        if open_positions >= self.max_open_positions: 
            self.can_open_new_positions = False 
            self.reason = self.reason or f'max_open_positions={open_positions}' 
        if self.max_trades_per_hour: 
            cutoff = datetime.utcnow() - timedelta(hours=1) 
            recent = sum(1 for ts in self.trade_timestamps if ts >= cutoff) 
            if recent >= self.max_trades_per_hour: 
                self.can_open_new_positions = False 
                self.reason = self.reason or f'max_trades_per_hour={recent}' 
        return self.get_status() 
 
    def _trigger(self, event, detail): 
        self.hard_stop = True 
        self.can_open_new_positions = False 
        self.reason = f'{event}: {detail}' 
        log_risk(event, detail=detail) 
        system_log.error(f'Global risk guard triggered {self.reason}') 
 
    def get_status(self): 
        return {'hard_stop': self.hard_stop, 'can_trade': self.can_open_new_positions and not self.hard_stop, 'reason': self.reason, 'max_daily_loss': self.max_daily_loss, 'max_drawdown_pct': self.max_drawdown_pct, 'max_open_positions': self.max_open_positions, 'max_consecutive_losses': self.max_consecutive_losses, 'max_trades_per_hour': self.max_trades_per_hour}
