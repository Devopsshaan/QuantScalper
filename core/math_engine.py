"""
math_engine.py — Ornstein-Uhlenbeck Mean-Reversion + VPIN Engine

MATHEMATICAL FOUNDATION:

  1. Ornstein-Uhlenbeck (O-U) Process:
     dX(t) = θ(μ - X(t))dt + σ dW(t)
     
     Where:
       θ = speed of mean reversion (higher = faster snap-back)
       μ = long-term equilibrium price (estimated via OLS)
       σ = volatility of the process
     
     Half-life = ln(2) / θ
     
     Trade signal: when price deviates > 2σ from μ AND half-life < 30 bars,
     enter mean-reversion trade expecting snap-back.

  2. VPIN (Volume-Synchronized Probability of Informed Trading):
     Estimates the probability that current order flow is driven by
     informed traders (whales, news front-runners).
     High VPIN = toxic flow = DON'T TRADE (you'll get run over).
     Low VPIN = normal flow = safe to enter.

  3. Integration: O-U provides the SIGNAL, VPIN provides the SAFETY CHECK.
     Combined, they create institutional-grade mean-reversion entries
     that avoid toxic flow periods.

WHY THIS IS UNIQUE:
  - No retail bot uses O-U process for crypto signal generation
  - VPIN is a market microstructure metric from academic literature
  - The half-life filter ensures we only trade FAST mean-reversion
    (slow mean-reversion = capital locked up too long)
"""

import numpy as np
from logger import get_logger

_log = get_logger("math_engine")


class OrnsteinUhlenbeckEngine:
    """
    Ornstein-Uhlenbeck mean-reversion signal generator.
    
    Fits an O-U process to recent price data, then checks if:
    1. The process IS mean-reverting (θ > 0)
    2. The half-life is short enough for scalping (< 30 bars)
    3. Price has deviated beyond 2σ from equilibrium
    4. Order flow (VPIN) is not toxic
    
    If all conditions met → signal mean-reversion trade.
    """

    def __init__(self, lookback: int = 60, deviation_threshold: float = 2.0,
                 max_half_life: int = 30, vpin_threshold: float = 0.7):
        self.lookback = lookback
        self.deviation_threshold = deviation_threshold
        self.max_half_life = max_half_life
        self.vpin_threshold = vpin_threshold
        self.last_result: dict = {}

    def analyze(self, prices: list, volumes: list = None) -> dict:
        """
        Run O-U analysis on recent prices.
        
        Args:
            prices: list of recent close prices (min 30)
            volumes: list of recent volumes (for VPIN calculation)
        
        Returns:
            {signal, theta, mu, sigma, half_life, deviation_z, vpin, is_safe}
        """
        if len(prices) < 30:
            return self._no_signal("insufficient_data")

        prices_arr = np.array(prices[-self.lookback:], dtype=float)
        log_prices = np.log(prices_arr)

        # ═══ Step 1: Fit O-U parameters via OLS regression ═══
        # dX = θ(μ - X)dt → regress X(t) on X(t-1)
        # X(t) = a + b*X(t-1) + ε
        # θ = -ln(b), μ = a/(1-b), σ = std(ε) * sqrt(-2*ln(b)/((1-b²)))
        
        y = log_prices[1:]    # X(t)
        x = log_prices[:-1]   # X(t-1)
        
        n = len(x)
        if n < 10:
            return self._no_signal("too_few_observations")

        # OLS: y = a + b*x
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        
        if ss_xx < 1e-15:
            return self._no_signal("zero_variance")

        b = ss_xy / ss_xx
        a = y_mean - b * x_mean

        # Check if mean-reverting: b must be < 1 (and > 0 for stability)
        if b >= 1.0 or b <= 0:
            return self._no_signal("not_mean_reverting")

        # O-U parameters
        theta = -np.log(b)           # speed of mean reversion
        mu = a / (1 - b)             # equilibrium (in log-price space)
        
        residuals = y - (a + b * x)
        sigma_resid = np.std(residuals)
        
        # σ of O-U process (annualized)
        if (1 - b**2) > 0:
            sigma = sigma_resid * np.sqrt(-2 * np.log(b) / (1 - b**2))
        else:
            sigma = sigma_resid

        # Half-life in bars
        half_life = np.log(2) / theta if theta > 0 else float('inf')

        # ═══ Step 2: Check deviation from equilibrium ═══
        current_log_price = log_prices[-1]
        deviation = current_log_price - mu
        
        # Z-score: how many σ away from equilibrium
        eq_std = sigma / np.sqrt(2 * theta) if theta > 0 else sigma
        if eq_std < 1e-10:
            eq_std = sigma_resid
        
        z_score = deviation / eq_std if eq_std > 0 else 0

        # ═══ Step 3: VPIN check ═══
        vpin = 0.0
        vpin_safe = True
        if volumes is not None and len(volumes) >= 20:
            vpin = self._compute_vpin(prices_arr[-20:], np.array(volumes[-20:], dtype=float))
            vpin_safe = vpin < self.vpin_threshold

        # ═══ Step 4: Generate signal ═══
        signal = None
        reason = "no_deviation"

        if half_life > self.max_half_life:
            reason = f"half_life_too_slow ({half_life:.1f} bars)"
        elif not vpin_safe:
            reason = f"vpin_toxic ({vpin:.3f})"
        elif z_score < -self.deviation_threshold:
            # Price is FAR BELOW equilibrium → BUY (mean-revert UP)
            signal = "LONG"
            reason = f"ou_mean_revert_up (z={z_score:.2f})"
        elif z_score > self.deviation_threshold:
            # Price is FAR ABOVE equilibrium → SELL (mean-revert DOWN)
            signal = "SHORT"
            reason = f"ou_mean_revert_down (z={z_score:.2f})"

        self.last_result = {
            'signal': signal,
            'theta': round(float(theta), 6),
            'mu': round(float(np.exp(mu)), 4),   # convert back from log
            'sigma': round(float(sigma), 6),
            'half_life': round(float(half_life), 1),
            'z_score': round(float(z_score), 3),
            'vpin': round(float(vpin), 4),
            'vpin_safe': vpin_safe,
            'reason': reason,
            'is_mean_reverting': theta > 0 and b < 1,
        }

        if signal:
            _log.info(
                f"O-U SIGNAL: {signal} | θ={theta:.4f} μ={np.exp(mu):.2f} "
                f"half_life={half_life:.1f} z={z_score:.2f} vpin={vpin:.3f}"
            )

        return self.last_result

    def _compute_vpin(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """
        Volume-Synchronized Probability of Informed Trading.
        
        Estimates order flow toxicity by classifying volume as
        buy-initiated or sell-initiated using the tick rule,
        then measuring the imbalance.
        
        High VPIN (>0.7) = informed traders dominating = DANGER
        Low VPIN (<0.4) = normal flow = SAFE
        """
        if len(prices) < 5 or len(volumes) < 5:
            return 0.0

        returns = np.diff(prices) / prices[:-1]
        
        # Classify each bar's volume as buy or sell using tick direction
        buy_vol = np.zeros(len(returns))
        sell_vol = np.zeros(len(returns))
        
        for i in range(len(returns)):
            vol = volumes[i + 1]  # volume of the bar
            if returns[i] > 0:
                buy_vol[i] = vol
            elif returns[i] < 0:
                sell_vol[i] = vol
            else:
                buy_vol[i] = vol / 2
                sell_vol[i] = vol / 2

        # VPIN = |buy_vol - sell_vol| / total_vol over buckets
        total_vol = np.sum(volumes[1:])
        if total_vol < 1e-10:
            return 0.0

        # Use 5 equal-volume buckets
        n_buckets = 5
        bucket_vol = total_vol / n_buckets
        
        imbalances = []
        cum_buy = 0
        cum_sell = 0
        cum_total = 0
        
        for i in range(len(returns)):
            cum_buy += buy_vol[i]
            cum_sell += sell_vol[i]
            cum_total += volumes[i + 1]
            
            if cum_total >= bucket_vol:
                imb = abs(cum_buy - cum_sell) / max(cum_total, 1e-10)
                imbalances.append(imb)
                cum_buy = 0
                cum_sell = 0
                cum_total = 0

        if not imbalances:
            return 0.0

        vpin = float(np.mean(imbalances))
        return min(1.0, max(0.0, vpin))

    def _no_signal(self, reason: str) -> dict:
        self.last_result = {
            'signal': None,
            'theta': 0, 'mu': 0, 'sigma': 0,
            'half_life': 0, 'z_score': 0,
            'vpin': 0, 'vpin_safe': True,
            'reason': reason,
            'is_mean_reverting': False,
        }
        return self.last_result
