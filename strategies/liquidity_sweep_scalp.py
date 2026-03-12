import pandas as pd 
 
class LiquiditySweepScalp: 
    def detect(self, df): 
        if df is None or len(df) < 25: 
            return {'signal': None, 'confidence': 0.0, 'reason': 'insufficient_data'} 
        latest = df.iloc[-1] 
        prev = df.iloc[-2] 
        volume_avg = float(df['volume'].tail(20).mean()) 
        recent_high = float(df['high'].tail(20).max()) 
        recent_low = float(df['low'].tail(20).min()) 
        candle_range = max(float(latest['high']) - float(latest['low']), 1e-9) 
        upper_wick = float(latest['high']) - max(float(latest['open']), float(latest['close'])) 
        lower_wick = min(float(latest['open']), float(latest['close'])) - float(latest['low']) 
        volume_spike = float(latest['volume']) / max(volume_avg, 1e-9) 
        sweep_high = float(latest['high']) >= recent_high and float(latest['close']) < float(latest['high']) and upper_wick / candle_range > 0.45 
        sweep_low = float(latest['low']) <= recent_low and float(latest['close']) > float(latest['low']) and lower_wick / candle_range > 0.45 
        if sweep_low and volume_spike > 1.2 and float(latest['close']) > float(prev['close']): 
            return {'signal': 'LONG', 'confidence': round(min(0.9, 0.55 + volume_spike / 4), 3), 'reason': 'liquidity_sweep_low'} 
        if sweep_high and volume_spike > 1.2 and float(latest['close']) < float(prev['close']): 
            return {'signal': 'SHORT', 'confidence': round(min(0.9, 0.55 + volume_spike / 4), 3), 'reason': 'liquidity_sweep_high'} 
        return {'signal': None, 'confidence': 0.0, 'reason': 'no_sweep'}
