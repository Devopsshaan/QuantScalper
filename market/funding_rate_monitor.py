import ccxt 
 
from logger import data_log 
 
class FundingRateMonitor: 
    def __init__(self): 
        self.exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}}) 
        self.last_rates = {} 
 
    def fetch_rate(self, symbol): 
        try: 
            if hasattr(self.exchange, 'fetch_funding_rate'): 
                payload = self.exchange.fetch_funding_rate(symbol) 
            else: 
                payload = {} 
            rate = float(payload.get('fundingRate', 0) or 0) 
            self.last_rates[symbol] = rate 
            return rate 
        except Exception as exc: 
            data_log.warning(f'Funding rate fetch failed {symbol}: {exc}') 
            return float(self.last_rates.get(symbol, 0)) 
 
    def get_signal_boost(self, symbol, side): 
        rate = self.fetch_rate(symbol) 
        magnitude = abs(rate) 
        if magnitude < 0.0005: 
            return {'rate': rate, 'boost': 0.0} 
        aligned = (side == 'SHORT' and rate > 0) or (side == 'LONG' and rate < 0) 
        boost = min(0.08, magnitude * 20) if aligned else 0.0 
        return {'rate': rate, 'boost': round(boost, 4)}
