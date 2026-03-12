"""
volatility_ranker.py — Ranks crypto assets by real-time volatility.

Fetches ATR-based volatility for multiple assets and ranks them
so the bot prioritizes the most volatile (most profitable) coins.
"""

import ccxt
import pandas as pd
from ta.volatility import AverageTrueRange


ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'XRP/USDT']


class VolatilityRanker:
    def __init__(self, assets: list[str] = None):
        self.assets = assets or ASSETS
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.rankings: list[dict] = []
        self.last_update = 0

    def fetch_volatility(self, symbol: str, timeframe='1m', limit=50) -> dict:
        """Fetch OHLCV and compute normalized volatility for one asset."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 20:
                return {'symbol': symbol, 'atr': 0, 'atr_pct': 0, 'price': 0, 'change_1h': 0}

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            price = float(df.iloc[-1]['close'])

            atr = AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=14
            ).average_true_range().iloc[-1]

            # Normalized volatility = ATR / price * 100
            atr_pct = (atr / price * 100) if price > 0 else 0

            # 1h price change (approx from 60 1m candles)
            if len(df) >= 50:
                change_1h = (price - float(df.iloc[-50]['close'])) / float(df.iloc[-50]['close']) * 100
            else:
                change_1h = 0

            return {
                'symbol': symbol,
                'price': round(price, 4),
                'atr': round(atr, 4),
                'atr_pct': round(atr_pct, 4),
                'change_1h': round(change_1h, 2),
                'volume': round(float(df['volume'].iloc[-20:].mean()), 2),
            }
        except Exception as e:
            print(f"[VolRanker] Error fetching {symbol}: {e}")
            return {'symbol': symbol, 'atr': 0, 'atr_pct': 0, 'price': 0, 'change_1h': 0}

    def rank_all(self) -> list[dict]:
        """Fetch volatility for all assets and rank by ATR%."""
        results = []
        for symbol in self.assets:
            data = self.fetch_volatility(symbol)
            results.append(data)

        # Sort by normalized volatility (highest first)
        results.sort(key=lambda x: x['atr_pct'], reverse=True)

        # Add rank
        for i, r in enumerate(results):
            r['rank'] = i + 1

        self.rankings = results
        return results

    def get_top_n(self, n: int = 3) -> list[str]:
        """Return top N most volatile symbols."""
        if not self.rankings:
            self.rank_all()
        return [r['symbol'] for r in self.rankings[:n]]

    def get_rankings(self) -> list[dict]:
        return self.rankings


if __name__ == '__main__':
    ranker = VolatilityRanker()
    rankings = ranker.rank_all()
    print("\n📊 Volatility Rankings:")
    for r in rankings:
        print(f"  #{r['rank']} {r['symbol']:12} | Price: ${r['price']:>10} | ATR%: {r['atr_pct']:.4f}% | 1h: {r['change_1h']:+.2f}%")
    print(f"\n🔥 Top 3: {ranker.get_top_n(3)}")
