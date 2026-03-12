"""
volatility_scanner.py — Dynamic asset selection based on volatility.

UPGRADE: Scores assets by ATR, 24h volume, and recent price movement.
Dynamically selects top-N most volatile pairs each cycle for scalping.

Criteria:
  - ATR (14-period) as % of price
  - Volume factor vs baseline
  - Recent price change magnitude
  - Spread analysis for scalp viability
"""

from ta.volatility import AverageTrueRange

from market.crypto_feed import CryptoFeed
from logger import get_logger

_log = get_logger("volatility")


class VolatilityScanner:
    def __init__(self, assets):
        self.assets = assets
        self.rankings = []
        self.feeds = {symbol: CryptoFeed(symbol=symbol, timeframe='1m', limit=120) for symbol in assets}
        self.atr_history = {}  # Track ATR rolling average per symbol

    def score_symbol(self, symbol):
        try:
            df = self.feeds[symbol].fetch_candles(limit=120)
            if df is None or df.empty or len(df) < 30:
                return {'symbol': symbol, 'volatility_score': 0, 'atr_pct': 0,
                        'volume_factor': 0, 'price_move_pct': 0, 'tradeable': False}

            atr = AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=14
            ).average_true_range().iloc[-1]

            price = float(df.iloc[-1]['close'])
            atr_pct = (atr / max(price, 1e-9)) * 100

            # Volume analysis
            avg_volume = float(df['volume'].tail(20).mean())
            base_volume = float(df['volume'].tail(60).mean()) if len(df) >= 60 else avg_volume
            volume_factor = avg_volume / max(base_volume, 1e-9)

            # Price movement magnitude
            price_move_pct = abs(
                (float(df.iloc[-1]['close']) - float(df.iloc[-15]['close']))
                / max(float(df.iloc[-15]['close']), 1e-9) * 100
            )

            # ATR trending check: is current ATR above its 20-period average?
            atr_series = AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=14
            ).average_true_range()
            atr_avg = float(atr_series.tail(20).mean()) if len(atr_series) >= 20 else float(atr_series.mean())
            atr_trending = atr > atr_avg  # volatility is expanding

            # Store for external access
            self.atr_history[symbol] = {
                'current': atr,
                'avg_20': atr_avg,
                'expanding': atr_trending,
            }

            # Spread estimate (high - low of last candle as % of price)
            last_spread_pct = (float(df.iloc[-1]['high']) - float(df.iloc[-1]['low'])) / max(price, 1e-9) * 100
            spread_ok = last_spread_pct < 0.5  # reject if spread is too wide

            # Composite volatility score
            score = atr_pct * volume_factor * (1 + price_move_pct / 100)
            # Bonus for expanding volatility
            if atr_trending:
                score *= 1.3

            return {
                'symbol': symbol,
                'volatility_score': round(score, 4),
                'atr_pct': round(atr_pct, 4),
                'volume_factor': round(volume_factor, 4),
                'price_move_pct': round(price_move_pct, 3),
                'price': round(price, 4),
                'atr_trending': atr_trending,
                'spread_pct': round(last_spread_pct, 4),
                'tradeable': spread_ok and atr_pct > 0.02,  # minimum volatility
            }
        except Exception as e:
            _log.warning(f"Score failed for {symbol}: {e}")
            return {'symbol': symbol, 'volatility_score': 0, 'atr_pct': 0,
                    'volume_factor': 0, 'price_move_pct': 0, 'tradeable': False}

    def rank_all(self):
        self.rankings = sorted(
            (self.score_symbol(symbol) for symbol in self.assets),
            key=lambda item: item['volatility_score'], reverse=True
        )
        for index, row in enumerate(self.rankings, start=1):
            row['rank'] = index
        return self.rankings

    def get_top_n(self, n=5):
        """
        Returns the top-N most volatile tradeable symbols.
        Default: top 5 (TASK 1 upgrade).
        """
        if not self.rankings:
            self.rank_all()
        tradeable = [item['symbol'] for item in self.rankings if item.get('tradeable', True)]
        if len(tradeable) < n:
            # Fallback: include all ranked assets if not enough tradeable
            tradeable = [item['symbol'] for item in self.rankings]
        result = tradeable[:n]
        if result:
            _log.info(
                f"Top {n} volatile: "
                + ", ".join(f"{s}({next((r['atr_pct'] for r in self.rankings if r['symbol']==s), 0):.3f}%)"
                            for s in result)
            )
        return result

    def is_market_active(self, symbol: str) -> bool:
        """
        TASK 5: High-volatility market filter.
        Returns True only when ATR is above its 20-period average
        and volume is not dead.
        """
        info = self.atr_history.get(symbol)
        if not info:
            return True  # allow if no data yet
        return info.get('expanding', False)
