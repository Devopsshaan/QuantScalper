import pandas as pd

class OrderbookImbalanceDetector:
    def detect(self, ob_data: dict) -> dict:
        """
        Signals based on extreme order book imbalance.
        Uses order book 'pressure' and 'imbalance' score.
        """
        if not ob_data:
            return {'signal': None, 'confidence': 0.0, 'reason': 'no_ob_data'}

        imbalance = ob_data.get('imbalance', 0.0)
        
        # High positive imbalance means massive buy wall = bullish
        if imbalance > 0.40:
            conf = min(0.95, 0.5 + (imbalance / 2))
            return {'signal': 'LONG', 'confidence': round(conf, 3), 'reason': 'heavy_buy_wall'}
            
        # High negative imbalance means massive sell wall = bearish
        elif imbalance < -0.40:
            conf = min(0.95, 0.5 + (abs(imbalance) / 2))
            return {'signal': 'SHORT', 'confidence': round(conf, 3), 'reason': 'heavy_sell_wall'}

        return {'signal': None, 'confidence': 0.0, 'reason': 'balanced_book'}
