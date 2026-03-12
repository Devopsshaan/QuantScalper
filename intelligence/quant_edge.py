"""
quant_edge.py — Advanced Quantitative Edge Detection Engine.

Implements 5 mathematically-proven techniques from academic finance:

  1. HURST EXPONENT — detects whether market is trending or mean-reverting
     Source: H.E. Hurst (1951), Mandelbrot (1968)
     H > 0.55 → trending (use momentum strategies)
     H < 0.45 → mean-reverting (use mean-reversion strategies)
     H ≈ 0.50 → random walk (DON'T trade — no edge)

  2. SHANNON ENTROPY — measures market predictability
     Source: Claude Shannon (1948), Information Theory
     Low entropy → predictable patterns → high-probability trades
     High entropy → random noise → avoid trading

  3. VWAP MAGNET — institutional price attractor
     Source: Berkowitz et al. (1988), market microstructure theory
     Price reverts to VWAP. Distance from VWAP = trade opportunity.

  4. BAYESIAN WIN RATE — real-time probability updating
     Source: Bayes' Theorem (1763), applied to trading
     Updates win probability after each trade using Bayesian inference.
     Adjusts position sizing dynamically based on recent performance.

  5. MICROSTRUCTURE MOMENTUM — trade arrival rate analysis
     Source: Kyle (1985), Easley & O'Hara (1987)
     Volume-clock momentum: measures momentum in volume-space not time-space.
     Detects informed trading activity vs noise.
"""

import math
import numpy as np
from collections import deque
from logger import get_logger

_log = get_logger("quant_edge")


class QuantEdgeEngine:
    """
    Advanced quantitative edge detection that no retail bot has.

    Combines:
      - Hurst exponent for regime detection
      - Shannon entropy for predictability filtering
      - VWAP magnet for institutional level trading
      - Bayesian win rate for adaptive sizing
      - Microstructure momentum for informed trading detection
    """

    def __init__(self):
        self.hurst_cache = {}             # symbol -> last hurst value
        self.entropy_cache = {}           # symbol -> last entropy value
        self.vwap_cache = {}              # symbol -> VWAP data
        self.trade_history = deque(maxlen=100)  # last 100 trades for Bayesian
        self.bayesian_prior_wins = 5      # prior: assume 5 wins
        self.bayesian_prior_total = 10    # prior: assume 10 trades (50% start)

    # ═══════════════════════════════════════════════════════
    # 1. HURST EXPONENT — Trending vs Mean-Reverting Detection
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def hurst_exponent(prices: list, max_lag: int = 20) -> float:
        """
        Calculate the Hurst exponent using R/S analysis.

        Returns:
            H > 0.55: trending market (momentum strategies work)
            H ≈ 0.50: random walk (NO edge, don't trade)
            H < 0.45: mean-reverting (mean reversion strategies work)

        Mathematical basis:
            E[R(n)/S(n)] = C * n^H
            where R = range of cumulative deviations, S = std deviation
        """
        if len(prices) < max_lag * 2:
            return 0.5  # insufficient data

        prices = np.array(prices, dtype=float)
        returns = np.diff(np.log(prices))

        if len(returns) < max_lag:
            return 0.5

        lags = range(2, max_lag + 1)
        rs_values = []
        lag_values = []

        for lag in lags:
            # Split into sub-series
            n_subseries = len(returns) // lag
            if n_subseries < 1:
                continue

            rs_list = []
            for i in range(n_subseries):
                subset = returns[i * lag:(i + 1) * lag]
                if len(subset) < 2:
                    continue
                mean_val = np.mean(subset)
                cumdev = np.cumsum(subset - mean_val)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(subset, ddof=1)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.log(np.mean(rs_list)))
                lag_values.append(np.log(lag))

        if len(rs_values) < 3:
            return 0.5

        # Linear regression: log(R/S) = H * log(n) + C
        try:
            coeffs = np.polyfit(lag_values, rs_values, 1)
            hurst = float(coeffs[0])
            return max(0.0, min(1.0, hurst))
        except Exception:
            return 0.5

    # ═══════════════════════════════════════════════════════
    # 2. SHANNON ENTROPY — Market Predictability Score
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def shannon_entropy(prices: list, n_bins: int = 10) -> float:
        """
        Calculate Shannon entropy of price returns.

        Low entropy = predictable patterns = GOOD for trading
        High entropy = random noise = BAD for trading

        Returns:
            normalized_entropy in [0, 1]
            0 = perfectly predictable
            1 = completely random

        Mathematical basis:
            H = -Σ p(x) * log2(p(x))
            Normalized: H_norm = H / log2(n_bins)
        """
        if len(prices) < 20:
            return 1.0  # assume random if insufficient data

        returns = np.diff(np.log(np.array(prices, dtype=float)))
        if len(returns) < 10:
            return 1.0

        # Discretize returns into bins
        hist, _ = np.histogram(returns, bins=n_bins, density=True)
        # Normalize to probabilities
        total = hist.sum()
        if total == 0:
            return 1.0
        probs = hist / total
        probs = probs[probs > 0]  # remove zero bins

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_bins)
        normalized = entropy / max_entropy if max_entropy > 0 else 1.0

        return float(max(0.0, min(1.0, normalized)))

    # ═══════════════════════════════════════════════════════
    # 3. VWAP MAGNET — Institutional Price Level
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def vwap_analysis(df) -> dict:
        """
        Calculate VWAP and measure price deviation.

        VWAP acts as a "price magnet" — institutional traders
        execute around VWAP. Price tends to revert to it.

        Returns:
            vwap: the volume-weighted average price
            deviation_pct: how far current price is from VWAP (%)
            signal: 'buy' if below VWAP, 'sell' if above, 'neutral' if near
            strength: 0-1 signal strength based on deviation magnitude
        """
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            cumvol = df['volume'].cumsum()
            cum_tp_vol = (typical_price * df['volume']).cumsum()
            vwap = cum_tp_vol / cumvol.replace(0, 1)

            current_vwap = float(vwap.iloc[-1])
            current_price = float(df['close'].iloc[-1])

            deviation_pct = ((current_price - current_vwap) / current_vwap) * 100

            # Signal: price below VWAP = buy (expect reversion up)
            # price above VWAP = sell (expect reversion down)
            abs_dev = abs(deviation_pct)

            if abs_dev < 0.05:
                signal = 'neutral'
                strength = 0.0
            elif deviation_pct < 0:
                signal = 'buy'
                strength = min(1.0, abs_dev / 0.5)  # max strength at 0.5% deviation
            else:
                signal = 'sell'
                strength = min(1.0, abs_dev / 0.5)

            return {
                'vwap': round(current_vwap, 2),
                'price': round(current_price, 2),
                'deviation_pct': round(deviation_pct, 4),
                'signal': signal,
                'strength': round(strength, 3),
            }
        except Exception:
            return {'vwap': 0, 'price': 0, 'deviation_pct': 0,
                    'signal': 'neutral', 'strength': 0}

    # ═══════════════════════════════════════════════════════
    # 4. BAYESIAN WIN RATE — Real-time Probability Updating
    # ═══════════════════════════════════════════════════════
    def bayesian_win_rate(self) -> dict:
        """
        Bayesian posterior estimate of win probability.

        Uses Beta-Binomial conjugate prior:
            Prior: Beta(α=5, β=5)  → 50% prior belief
            After each win: α += 1
            After each loss: β += 1
            Posterior mean = α / (α + β)

        This automatically weights recent performance while
        maintaining a stable long-term estimate.

        Returns:
            win_rate: posterior mean win probability
            confidence: how certain we are (0-1)
            suggested_kelly: optimal Kelly fraction for this win rate
            sample_size: number of trades analyzed
        """
        wins = self.bayesian_prior_wins
        total = self.bayesian_prior_total

        # Count recent trades
        for trade in self.trade_history:
            total += 1
            if trade.get('pnl', 0) > 0:
                wins += 1

        alpha = wins
        beta_param = total - wins
        posterior_mean = alpha / (alpha + beta_param) if (alpha + beta_param) > 0 else 0.5

        # Confidence: based on sample size (more trades = more confident)
        sample_size = len(self.trade_history)
        confidence = min(1.0, sample_size / 50)  # full confidence at 50 trades

        # Kelly criterion with Bayesian win rate
        # Assuming average RR of 2.0
        rr = 2.0
        kelly_f = posterior_mean - (1 - posterior_mean) / rr
        kelly_f = max(0, kelly_f) * 0.5  # half-Kelly for safety

        return {
            'win_rate': round(posterior_mean, 4),
            'confidence': round(confidence, 3),
            'suggested_kelly': round(kelly_f, 4),
            'sample_size': sample_size,
            'alpha': round(alpha, 1),
            'beta': round(beta_param, 1),
        }

    def record_trade(self, pnl: float, symbol: str = ''):
        """Record a completed trade for Bayesian updating."""
        self.trade_history.append({'pnl': pnl, 'symbol': symbol})

    # ═══════════════════════════════════════════════════════
    # 5. MICROSTRUCTURE MOMENTUM — Volume-Clock Analysis
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def volume_clock_momentum(df, lookback: int = 20) -> dict:
        """
        Momentum measured in volume-space, not time-space.

        Traditional momentum uses fixed time intervals (misleading).
        Volume-clock measures momentum per unit of volume traded,
        revealing true buying/selling pressure.

        Higher volume-momentum = stronger institutional conviction.

        Returns:
            vol_momentum: momentum per unit volume (-1 to +1)
            informed_ratio: estimated proportion of informed trading
            signal: 'buy', 'sell', or 'neutral'
        """
        try:
            if len(df) < lookback + 5:
                return {'vol_momentum': 0, 'informed_ratio': 0, 'signal': 'neutral'}

            recent = df.tail(lookback)
            price_change = float(recent['close'].iloc[-1]) - float(recent['close'].iloc[0])
            total_volume = float(recent['volume'].sum())

            if total_volume == 0:
                return {'vol_momentum': 0, 'informed_ratio': 0, 'signal': 'neutral'}

            # Volume-weighted price change per unit volume
            vol_momentum = price_change / (total_volume + 1e-9)

            # Normalize to [-1, 1] range using recent price
            price_level = float(recent['close'].mean())
            normalized = (vol_momentum / price_level) * 1e6  # scale factor
            normalized = max(-1.0, min(1.0, normalized))

            # Informed trading ratio (VPIN-inspired)
            # Compare volume of up-bars vs down-bars
            up_vol = float(recent[recent['close'] > recent['open']]['volume'].sum())
            down_vol = float(recent[recent['close'] <= recent['open']]['volume'].sum())
            total_bar_vol = up_vol + down_vol
            if total_bar_vol > 0:
                order_imbalance = abs(up_vol - down_vol) / total_bar_vol
            else:
                order_imbalance = 0

            signal = 'neutral'
            if normalized > 0.15:
                signal = 'buy'
            elif normalized < -0.15:
                signal = 'sell'

            return {
                'vol_momentum': round(normalized, 4),
                'informed_ratio': round(order_imbalance, 4),
                'signal': signal,
            }
        except Exception:
            return {'vol_momentum': 0, 'informed_ratio': 0, 'signal': 'neutral'}

    # ═══════════════════════════════════════════════════════
    # COMPOSITE EDGE SCORE — Combines All 5 Techniques
    # ═══════════════════════════════════════════════════════
    def compute_edge(self, symbol: str, df, signal_direction: str = 'LONG') -> dict:
        """
        Compute composite quantitative edge score.

        Combines all 5 techniques into a single actionable score.

        Returns:
            edge_score: 0-1 (higher = stronger edge)
            trade_recommended: bool
            edge_breakdown: detailed scores from each technique
            edge_boost: quality score boost to add
        """
        try:
            prices = df['close'].tolist()[-60:]

            # 1. Hurst exponent
            hurst = self.hurst_exponent(prices)
            self.hurst_cache[symbol] = hurst

            # Hurst score: best when clearly trending (>0.55) or mean-reverting (<0.45)
            # Worst near 0.5 (random walk — no edge)
            hurst_edge = abs(hurst - 0.5) * 2  # 0 at random, 1 at extreme

            # Strategy alignment: if trending, momentum signals are better
            if signal_direction == 'LONG' and hurst > 0.55:
                hurst_edge *= 1.3  # bonus for aligned trending
            elif signal_direction == 'SHORT' and hurst > 0.55:
                hurst_edge *= 1.3
            elif hurst < 0.45:
                hurst_edge *= 1.2  # mean-reversion also has edge

            hurst_edge = min(1.0, hurst_edge)

            # 2. Shannon entropy
            entropy = self.shannon_entropy(prices)
            self.entropy_cache[symbol] = entropy

            # Lower entropy = more predictable = better edge
            entropy_edge = 1.0 - entropy  # invert: high score = low entropy

            # 3. VWAP analysis
            vwap = self.vwap_analysis(df)
            vwap_aligned = False
            if signal_direction == 'LONG' and vwap['signal'] == 'buy':
                vwap_aligned = True
            elif signal_direction == 'SHORT' and vwap['signal'] == 'sell':
                vwap_aligned = True
            vwap_edge = vwap['strength'] if vwap_aligned else vwap['strength'] * 0.3

            # 4. Bayesian win rate
            bayesian = self.bayesian_win_rate()
            bayesian_edge = bayesian['win_rate'] * bayesian['confidence']

            # 5. Volume-clock momentum
            vcm = self.volume_clock_momentum(df)
            vcm_aligned = False
            if signal_direction == 'LONG' and vcm['signal'] == 'buy':
                vcm_aligned = True
            elif signal_direction == 'SHORT' and vcm['signal'] == 'sell':
                vcm_aligned = True
            vcm_edge = abs(vcm['vol_momentum']) if vcm_aligned else abs(vcm['vol_momentum']) * 0.2

            # Composite edge score with weights
            weights = {
                'hurst': 0.25,      # 25% — regime detection
                'entropy': 0.20,    # 20% — predictability
                'vwap': 0.20,       # 20% — institutional level
                'bayesian': 0.15,   # 15% — adapted win rate
                'vcm': 0.20,        # 20% — microstructure
            }

            edge_score = (
                weights['hurst'] * hurst_edge +
                weights['entropy'] * entropy_edge +
                weights['vwap'] * vwap_edge +
                weights['bayesian'] * bayesian_edge +
                weights['vcm'] * vcm_edge
            )
            edge_score = round(max(0, min(1.0, edge_score)), 4)

            # Trade recommendation: only if edge > 0.35 AND entropy < 0.85
            trade_recommended = edge_score > 0.35 and entropy < 0.85

            # Edge boost to add to quality score
            edge_boost = edge_score * 0.12  # max 12% boost

            result = {
                'edge_score': edge_score,
                'trade_recommended': trade_recommended,
                'edge_boost': round(edge_boost, 4),
                'breakdown': {
                    'hurst': round(hurst, 4),
                    'hurst_edge': round(hurst_edge, 4),
                    'entropy': round(entropy, 4),
                    'entropy_edge': round(entropy_edge, 4),
                    'vwap_deviation': vwap['deviation_pct'],
                    'vwap_signal': vwap['signal'],
                    'vwap_edge': round(vwap_edge, 4),
                    'bayesian_win_rate': bayesian['win_rate'],
                    'bayesian_edge': round(bayesian_edge, 4),
                    'vol_momentum': vcm['vol_momentum'],
                    'informed_ratio': vcm['informed_ratio'],
                    'vcm_edge': round(vcm_edge, 4),
                },
            }

            _log.info(
                f"[EDGE] {symbol} {signal_direction} score={edge_score:.3f} "
                f"H={hurst:.3f} S={entropy:.3f} VWAP={vwap['deviation_pct']:.3f}% "
                f"Bayes={bayesian['win_rate']:.2f} VCM={vcm['vol_momentum']:.3f} "
                f"→ {'TRADE' if trade_recommended else 'SKIP'}"
            )

            return result

        except Exception as e:
            _log.warning(f"[EDGE] {symbol} computation failed: {e}")
            return {
                'edge_score': 0.3,
                'trade_recommended': True,  # don't block on failure
                'edge_boost': 0.03,
                'breakdown': {},
            }
