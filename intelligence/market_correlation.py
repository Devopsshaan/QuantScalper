"""
market_correlation.py — Cross-market correlation engine.

Fetches S&P 500, DXY (Dollar Index), Gold, and VIX via yfinance.
Computes correlation with BTC to generate RISK_ON / RISK_OFF bias
that feeds into the signal pipeline as an additional vote.

BTC correlations:
  • S&P 500 ↑ → BTC tends to follow (risk-on)
  • DXY ↑ → BTC tends to fall (inverse)
  • Gold ↑ → Mixed, but flight to safety can hurt BTC
  • VIX ↑ → Fear spike → BTC dumps
"""

import time
import threading
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Cross-market tickers
TICKERS = {
    'SP500': '^GSPC',
    'DXY': 'DX-Y.NYB',
    'GOLD': 'GC=F',
    'VIX': '^VIX',
}

# How each asset correlates with BTC direction
# positive = same direction, negative = inverse
CORRELATION_SIGNS = {
    'SP500': +1,   # S&P up → BTC up (risk-on)
    'DXY': -1,     # Dollar up → BTC down
    'GOLD': -0.3,  # Weak inverse
    'VIX': -1,     # Fear up → BTC down
}


class MarketCorrelation:
    """
    Cross-market correlation signal generator.

    Fetches traditional market data and generates a directional bias
    for crypto trading based on macro conditions.
    """

    CACHE_TTL = 300  # 5 minutes (trad markets move slower)

    def __init__(self):
        self.market_data: dict[str, dict] = {}
        self.bias = 'NEUTRAL'
        self.bias_score = 0.0
        self.confidence = 0.0
        self.last_update = 0
        self._lock = threading.Lock()

    def update(self) -> dict:
        """Fetch latest cross-market data and compute BTC bias."""
        if not HAS_YFINANCE:
            return self._neutral_result("yfinance not installed")

        # Cache check
        if time.time() - self.last_update < self.CACHE_TTL:
            return self.get_status()

        try:
            signals = {}

            for name, ticker in TICKERS.items():
                try:
                    data = yf.download(
                        ticker, period='2d', interval='1h',
                        progress=False, auto_adjust=True
                    )
                    if data is None or len(data) < 5:
                        signals[name] = {'direction': 'UNKNOWN', 'change': 0}
                        continue

                    close = data['Close']
                    if hasattr(close, 'values'):
                        close = close.values.flatten()

                    current = float(close[-1])
                    prev = float(close[-5]) if len(close) >= 5 else float(close[0])
                    change_pct = ((current - prev) / prev) * 100 if prev > 0 else 0

                    if change_pct > 0.1:
                        direction = 'UP'
                    elif change_pct < -0.1:
                        direction = 'DOWN'
                    else:
                        direction = 'FLAT'

                    signals[name] = {
                        'direction': direction,
                        'change': round(change_pct, 3),
                        'price': round(current, 2),
                    }
                except Exception as e:
                    print(f"[Correlation] Error fetching {name}: {e}")
                    signals[name] = {'direction': 'UNKNOWN', 'change': 0}

            # Compute composite BTC bias
            score = 0.0
            factors = []

            for name, sig in signals.items():
                if sig['direction'] == 'UNKNOWN':
                    continue

                corr_sign = CORRELATION_SIGNS.get(name, 0)
                change = sig['change']

                # Weight by magnitude and correlation direction
                impact = change * corr_sign

                if name == 'SP500':
                    weight = 1.5  # strongest correlation
                elif name == 'DXY':
                    weight = 1.2
                elif name == 'VIX':
                    weight = 1.0
                else:
                    weight = 0.5

                contribution = impact * weight
                score += contribution

                arrow = '↑' if sig['direction'] == 'UP' else ('↓' if sig['direction'] == 'DOWN' else '→')
                factors.append(f"{name} {arrow} {sig['change']:+.2f}%")

            # Normalize score to -1 to +1 range
            normalized = np.clip(score / 3.0, -1, 1)

            if normalized > 0.2:
                bias = 'RISK_ON'  # bullish for BTC
            elif normalized < -0.2:
                bias = 'RISK_OFF'  # bearish for BTC
            else:
                bias = 'NEUTRAL'

            confidence = min(abs(normalized), 1.0)

            with self._lock:
                self.market_data = signals
                self.bias = bias
                self.bias_score = round(float(normalized), 3)
                self.confidence = round(confidence, 3)
                self.last_update = time.time()

            print(f"[Correlation] {bias} (score={self.bias_score:.3f}) | {' | '.join(factors)}")

        except Exception as e:
            print(f"[Correlation] Update error: {e}")
            return self._neutral_result(str(e))

        return self.get_status()

    def get_vote(self, signal: str) -> dict:
        """
        Return correlation vote for a given signal direction.

        RISK_ON + LONG → confirms (vote weight +0.8)
        RISK_OFF + SHORT → confirms (vote weight +0.8)
        Conflicting → penalize (vote weight -0.5)
        """
        if self.bias == 'NEUTRAL':
            return {'confirmed': True, 'weight': 0.0, 'bias': self.bias}

        if signal == 'LONG' and self.bias == 'RISK_ON':
            return {'confirmed': True, 'weight': 0.8 * self.confidence, 'bias': self.bias}
        elif signal == 'SHORT' and self.bias == 'RISK_OFF':
            return {'confirmed': True, 'weight': 0.8 * self.confidence, 'bias': self.bias}
        elif signal == 'LONG' and self.bias == 'RISK_OFF':
            return {'confirmed': False, 'weight': -0.5 * self.confidence, 'bias': self.bias}
        elif signal == 'SHORT' and self.bias == 'RISK_ON':
            return {'confirmed': False, 'weight': -0.5 * self.confidence, 'bias': self.bias}

        return {'confirmed': True, 'weight': 0.0, 'bias': self.bias}

    def get_status(self) -> dict:
        """Return current cross-market status for dashboard."""
        with self._lock:
            return {
                'bias': self.bias,
                'score': self.bias_score,
                'confidence': self.confidence,
                'markets': self.market_data.copy(),
                'age_seconds': int(time.time() - self.last_update) if self.last_update else -1,
            }

    def _neutral_result(self, reason: str = "") -> dict:
        return {
            'bias': 'NEUTRAL',
            'score': 0.0,
            'confidence': 0.0,
            'markets': {},
            'reason': reason,
            'age_seconds': -1,
        }


if __name__ == '__main__':
    mc = MarketCorrelation()
    result = mc.update()
    print(f"\n🌍 Cross-Market Bias: {result['bias']} (score={result['score']:.3f})")
    for name, data in result.get('markets', {}).items():
        print(f"   {name}: {data.get('direction', '?')} {data.get('change', 0):+.3f}% (${data.get('price', 0):,.2f})")

    # Test voting
    for sig in ['LONG', 'SHORT']:
        vote = mc.get_vote(sig)
        print(f"   {sig} vote: confirmed={vote['confirmed']}, weight={vote['weight']:.3f}")
