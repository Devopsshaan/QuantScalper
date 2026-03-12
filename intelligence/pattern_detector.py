"""
pattern_detector.py — Candlestick pattern recognition + support/resistance detection.

Detects:
  Candlestick patterns: Bullish/Bearish Engulfing, Hammer, Inverted Hammer,
                        Doji, Pin Bar, Morning/Evening Star
  Chart patterns: Support/Resistance levels from price clusters

Provides pattern-based confirmation signals for the trading pipeline.
"""

import numpy as np
import pandas as pd


class PatternDetector:
    """
    Detects candlestick patterns and support/resistance levels.
    Returns pattern signals that feed into the strategy voting system.
    """

    def __init__(self):
        self.last_patterns: list[dict] = []
        self.support_levels: list[float] = []
        self.resistance_levels: list[float] = []

    def detect_all(self, df: pd.DataFrame) -> dict:
        """
        Run all pattern detection on the DataFrame.

        Returns:
            {
                'patterns': [{'name': str, 'direction': 'BULL'|'BEAR', 'strength': float}],
                'support': [float],
                'resistance': [float],
                'net_signal': 'BULL' | 'BEAR' | 'NEUTRAL',
                'pattern_score': float (-1 to +1),
                'near_support': bool,
                'near_resistance': bool,
            }
        """
        if df is None or len(df) < 10:
            return self._empty_result()

        patterns = []

        # Candlestick patterns (check last 3-5 candles)
        patterns.extend(self._detect_engulfing(df))
        patterns.extend(self._detect_hammer(df))
        patterns.extend(self._detect_doji(df))
        patterns.extend(self._detect_pin_bar(df))
        patterns.extend(self._detect_star_patterns(df))

        # Chart structure
        self.support_levels, self.resistance_levels = self._find_sr_levels(df)

        # Compute net signal
        bull_score = sum(p['strength'] for p in patterns if p['direction'] == 'BULL')
        bear_score = sum(p['strength'] for p in patterns if p['direction'] == 'BEAR')
        net = bull_score - bear_score

        if net > 0.3:
            net_signal = 'BULL'
        elif net < -0.3:
            net_signal = 'BEAR'
        else:
            net_signal = 'NEUTRAL'

        # Check if price is near S/R
        price = float(df.iloc[-1]['close'])
        near_support = any(abs(price - s) / price < 0.001 for s in self.support_levels)
        near_resistance = any(abs(price - r) / price < 0.001 for r in self.resistance_levels)

        self.last_patterns = patterns

        return {
            'patterns': patterns,
            'support': self.support_levels[:3],
            'resistance': self.resistance_levels[:3],
            'net_signal': net_signal,
            'pattern_score': round(np.clip(net, -1, 1), 3),
            'near_support': near_support,
            'near_resistance': near_resistance,
        }

    # ═════════════════════════════════════════════════════
    # CANDLESTICK PATTERN DETECTION
    # ═════════════════════════════════════════════════════

    def _body(self, row) -> float:
        return abs(float(row['close']) - float(row['open']))

    def _upper_wick(self, row) -> float:
        return float(row['high']) - max(float(row['close']), float(row['open']))

    def _lower_wick(self, row) -> float:
        return min(float(row['close']), float(row['open'])) - float(row['low'])

    def _range(self, row) -> float:
        return float(row['high']) - float(row['low'])

    def _is_bullish(self, row) -> bool:
        return float(row['close']) > float(row['open'])

    def _detect_engulfing(self, df: pd.DataFrame) -> list[dict]:
        """Detect bullish/bearish engulfing patterns."""
        patterns = []
        if len(df) < 2:
            return patterns

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        curr_body = self._body(curr)
        prev_body = self._body(prev)

        if prev_body == 0:
            return patterns

        # Bullish engulfing: prev bearish + curr bullish + curr body engulfs prev
        if (not self._is_bullish(prev) and self._is_bullish(curr)
                and curr_body > prev_body * 1.1
                and float(curr['close']) > float(prev['open'])
                and float(curr['open']) < float(prev['close'])):
            patterns.append({
                'name': 'Bullish Engulfing',
                'direction': 'BULL',
                'strength': 0.7,
            })

        # Bearish engulfing: prev bullish + curr bearish + curr body engulfs prev
        if (self._is_bullish(prev) and not self._is_bullish(curr)
                and curr_body > prev_body * 1.1
                and float(curr['close']) < float(prev['open'])
                and float(curr['open']) > float(prev['close'])):
            patterns.append({
                'name': 'Bearish Engulfing',
                'direction': 'BEAR',
                'strength': 0.7,
            })

        return patterns

    def _detect_hammer(self, df: pd.DataFrame) -> list[dict]:
        """Detect hammer and inverted hammer patterns."""
        patterns = []
        curr = df.iloc[-1]
        body = self._body(curr)
        lower = self._lower_wick(curr)
        upper = self._upper_wick(curr)
        total = self._range(curr)

        if total == 0 or body == 0:
            return patterns

        # Hammer: small body at top, long lower wick (2x+ body)
        if lower > body * 2 and upper < body * 0.5:
            patterns.append({
                'name': 'Hammer',
                'direction': 'BULL',
                'strength': 0.6,
            })

        # Inverted Hammer: small body at bottom, long upper wick
        if upper > body * 2 and lower < body * 0.5:
            patterns.append({
                'name': 'Inverted Hammer',
                'direction': 'BEAR',
                'strength': 0.5,
            })

        return patterns

    def _detect_doji(self, df: pd.DataFrame) -> list[dict]:
        """Detect doji patterns (indecision → potential reversal)."""
        patterns = []
        curr = df.iloc[-1]
        body = self._body(curr)
        total = self._range(curr)

        if total == 0:
            return patterns

        # Doji: body < 10% of total range
        if body / total < 0.10:
            # Determine if it's at a local high or low (last 10 candles)
            price = float(curr['close'])
            recent_highs = df['high'].iloc[-10:].values
            recent_lows = df['low'].iloc[-10:].values

            if price >= np.percentile(recent_highs, 80):
                patterns.append({
                    'name': 'Doji (at high)',
                    'direction': 'BEAR',
                    'strength': 0.4,
                })
            elif price <= np.percentile(recent_lows, 20):
                patterns.append({
                    'name': 'Doji (at low)',
                    'direction': 'BULL',
                    'strength': 0.4,
                })

        return patterns

    def _detect_pin_bar(self, df: pd.DataFrame) -> list[dict]:
        """Detect pin bars (long wick rejection)."""
        patterns = []
        curr = df.iloc[-1]
        body = self._body(curr)
        upper = self._upper_wick(curr)
        lower = self._lower_wick(curr)
        total = self._range(curr)

        if total == 0 or body == 0:
            return patterns

        # Bullish pin bar: very long lower wick (3x+ body), tiny upper wick
        if lower > body * 3 and upper < total * 0.15:
            patterns.append({
                'name': 'Bullish Pin Bar',
                'direction': 'BULL',
                'strength': 0.8,
            })

        # Bearish pin bar: very long upper wick (3x+ body), tiny lower wick
        if upper > body * 3 and lower < total * 0.15:
            patterns.append({
                'name': 'Bearish Pin Bar',
                'direction': 'BEAR',
                'strength': 0.8,
            })

        return patterns

    def _detect_star_patterns(self, df: pd.DataFrame) -> list[dict]:
        """Detect morning star / evening star (3-candle reversal)."""
        patterns = []
        if len(df) < 3:
            return patterns

        c1 = df.iloc[-3]  # first candle
        c2 = df.iloc[-2]  # star (small body)
        c3 = df.iloc[-1]  # third candle

        b1 = self._body(c1)
        b2 = self._body(c2)
        b3 = self._body(c3)

        if b1 == 0:
            return patterns

        # Morning Star: bearish → small body → bullish (reversal up)
        if (not self._is_bullish(c1) and b2 < b1 * 0.3 and self._is_bullish(c3)
                and b3 > b1 * 0.5):
            patterns.append({
                'name': 'Morning Star',
                'direction': 'BULL',
                'strength': 0.9,
            })

        # Evening Star: bullish → small body → bearish (reversal down)
        if (self._is_bullish(c1) and b2 < b1 * 0.3 and not self._is_bullish(c3)
                and b3 > b1 * 0.5):
            patterns.append({
                'name': 'Evening Star',
                'direction': 'BEAR',
                'strength': 0.9,
            })

        return patterns

    # ═════════════════════════════════════════════════════
    # SUPPORT / RESISTANCE DETECTION
    # ═════════════════════════════════════════════════════

    def _find_sr_levels(self, df: pd.DataFrame, window: int = 5, merge_pct: float = 0.002) -> tuple:
        """
        Find support and resistance levels from local highs/lows.
        Merges levels that are within merge_pct of each other.
        """
        if len(df) < window * 2:
            return [], []

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        price = float(closes[-1])

        # Find local maxima (resistance) and minima (support)
        resistance_raw = []
        support_raw = []

        for i in range(window, len(df) - window):
            # Local maximum
            if highs[i] == max(highs[i - window:i + window + 1]):
                resistance_raw.append(float(highs[i]))
            # Local minimum
            if lows[i] == min(lows[i - window:i + window + 1]):
                support_raw.append(float(lows[i]))

        # Merge nearby levels
        def merge_levels(levels):
            if not levels:
                return []
            levels = sorted(levels)
            merged = [levels[0]]
            for lvl in levels[1:]:
                if abs(lvl - merged[-1]) / merged[-1] < merge_pct:
                    merged[-1] = (merged[-1] + lvl) / 2  # average
                else:
                    merged.append(lvl)
            return merged

        support = merge_levels(support_raw)
        resistance = merge_levels(resistance_raw)

        # Filter: only keep levels within 2% of current price
        support = sorted([s for s in support if s < price and abs(s - price) / price < 0.02],
                         reverse=True)
        resistance = sorted([r for r in resistance if r > price and abs(r - price) / price < 0.02])

        return support[:5], resistance[:5]

    def get_vote(self, signal: str) -> dict:
        """
        Return pattern-based confirmation for a signal.
        """
        if not self.last_patterns:
            return {'confirmed': True, 'weight': 0.0, 'patterns': []}

        pattern_names = [p['name'] for p in self.last_patterns]

        bull_patterns = [p for p in self.last_patterns if p['direction'] == 'BULL']
        bear_patterns = [p for p in self.last_patterns if p['direction'] == 'BEAR']

        if signal == 'LONG' and bull_patterns:
            weight = sum(p['strength'] for p in bull_patterns) * 0.5
            return {'confirmed': True, 'weight': min(weight, 0.8), 'patterns': pattern_names}
        elif signal == 'SHORT' and bear_patterns:
            weight = sum(p['strength'] for p in bear_patterns) * 0.5
            return {'confirmed': True, 'weight': min(weight, 0.8), 'patterns': pattern_names}
        elif signal == 'LONG' and bear_patterns:
            return {'confirmed': False, 'weight': -0.3, 'patterns': pattern_names}
        elif signal == 'SHORT' and bull_patterns:
            return {'confirmed': False, 'weight': -0.3, 'patterns': pattern_names}

        return {'confirmed': True, 'weight': 0.0, 'patterns': pattern_names}

    def get_status(self) -> dict:
        """Return current pattern status for dashboard."""
        return {
            'patterns': self.last_patterns,
            'support': self.support_levels[:3],
            'resistance': self.resistance_levels[:3],
        }

    def _empty_result(self) -> dict:
        return {
            'patterns': [],
            'support': [],
            'resistance': [],
            'net_signal': 'NEUTRAL',
            'pattern_score': 0,
            'near_support': False,
            'near_resistance': False,
        }
