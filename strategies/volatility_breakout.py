import pandas as pd
import numpy as np

class VolatilityBreakoutStrategy:
    def detect(self, df: pd.DataFrame) -> dict:
        """
        Detects volatility expansions (Bollinger Bands widening) and 
        rides the momentum breakout.
        """
        if df is None or len(df) < 50:
            return {'signal': None, 'confidence': 0.0, 'reason': 'insufficient_data'}
            
        # Calculate Bollinger Bands
        period = 20
        df['sma'] = df['close'].rolling(window=period).mean()
        df['std'] = df['close'].rolling(window=period).std()
        df['upper_band'] = df['sma'] + (df['std'] * 2)
        df['lower_band'] = df['sma'] - (df['std'] * 2)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma']
        
        # Calculate expansion (Current width > 1.5x the rolling 20-period average width)
        bb_width_avg = float(df['bb_width'].tail(20).mean())
        current_width = float(df.iloc[-1]['bb_width'])
        width_expansion = current_width / max(bb_width_avg, 1e-9)
        
        latest = df.iloc[-1]
        
        if width_expansion > 1.5:
            conf = min(0.95, 0.60 + (width_expansion / 5))
            
            # Breakout UP
            if float(latest['close']) > float(latest['upper_band']):
                return {'signal': 'LONG', 'confidence': round(conf, 3), 'reason': 'bb_breakout_up'}
            
            # Breakout DOWN    
            elif float(latest['close']) < float(latest['lower_band']):
                 return {'signal': 'SHORT', 'confidence': round(conf, 3), 'reason': 'bb_breakout_down'}

        return {'signal': None, 'confidence': 0.0, 'reason': 'no_breakout'}
