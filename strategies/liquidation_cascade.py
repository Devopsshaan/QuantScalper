import pandas as pd

class LiquidationCascadeDetector:
    def detect(self, df: pd.DataFrame) -> dict:
        """
        Detects liquidation cascades (long wicks + extreme volume) 
        and signals mean-reversion trades against the wick.
        """
        if df is None or len(df) < 25:
            return {'signal': None, 'confidence': 0.0, 'reason': 'insufficient_data'}
            
        latest = df.iloc[-1]
        volume_avg = float(df['volume'].tail(20).mean())
        candle_range = max(float(latest['high']) - float(latest['low']), 1e-9)
        
        upper_wick = float(latest['high']) - max(float(latest['open']), float(latest['close']))
        lower_wick = min(float(latest['open']), float(latest['close'])) - float(latest['low'])
        volume_spike = float(latest['volume']) / max(volume_avg, 1e-9)
        
        # Detect massive drops heavily bought up (long squeeze liquidated, mean reverting up)
        if lower_wick / candle_range > 0.60 and volume_spike > 2.5:
            conf = min(0.95, 0.60 + (volume_spike / 10))
            return {'signal': 'LONG', 'confidence': round(conf, 3), 'reason': 'short_squeeze_reversal'}
            
        # Detect massive pumps aggressively sold (short squeeze liquidated, mean reverting down)
        if upper_wick / candle_range > 0.60 and volume_spike > 2.5:
            conf = min(0.95, 0.60 + (volume_spike / 10))
            return {'signal': 'SHORT', 'confidence': round(conf, 3), 'reason': 'long_squeeze_reversal'}

        return {'signal': None, 'confidence': 0.0, 'reason': 'no_cascade'}
