"""
probability_engine.py — Calibrated probability fusion engine.

AUDIT UPGRADES (Task 5):
  - Added Platt scaling (logistic calibration) for raw model outputs
  - Added isotonic regression calibration option
  - Probability calibration ensures predicted probabilities reflect
    real empirical frequencies (e.g., 70% predicted = ~70% observed)
  - Tracks calibration error for monitoring
"""

import numpy as np
from collections import deque

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ProbabilityCalibrator:
    """
    Calibrates raw probabilities using Platt scaling or isotonic regression.

    Platt scaling fits a logistic function: P(y=1|f) = 1 / (1 + exp(A*f + B))
    Isotonic regression fits a non-parametric monotone function.
    """

    def __init__(self, method: str = 'platt'):
        self.method = method
        self.is_fitted = False
        self._platt_a = 0.0
        self._platt_b = 0.0
        self._isotonic = None

        # History for calibration fitting
        self._raw_probs = deque(maxlen=500)
        self._outcomes = deque(maxlen=500)
        self.calibration_error = 0.0

    def record(self, raw_prob: float, actual_outcome: int):
        """Record a prediction and its outcome for calibration."""
        self._raw_probs.append(raw_prob)
        self._outcomes.append(actual_outcome)

    def fit(self) -> bool:
        """Fit calibration model on accumulated history."""
        if not HAS_SKLEARN or len(self._raw_probs) < 30:
            return False

        X = np.array(self._raw_probs).reshape(-1, 1)
        y = np.array(self._outcomes)

        if len(np.unique(y)) < 2:
            return False

        try:
            if self.method == 'platt':
                lr = LogisticRegression(C=1.0, solver='lbfgs')
                lr.fit(X, y)
                self._platt_a = float(lr.coef_[0][0])
                self._platt_b = float(lr.intercept_[0])
            elif self.method == 'isotonic':
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(X.ravel(), y)
                self._isotonic = iso

            self.is_fitted = True
            self._compute_calibration_error()
            return True
        except Exception:
            return False

    def calibrate(self, raw_prob: float) -> float:
        """Apply calibration to a raw probability."""
        if not self.is_fitted:
            return raw_prob

        try:
            if self.method == 'platt':
                logit = self._platt_a * raw_prob + self._platt_b
                calibrated = 1.0 / (1.0 + np.exp(-logit))
                return float(np.clip(calibrated, 0.01, 0.99))
            elif self.method == 'isotonic' and self._isotonic is not None:
                calibrated = self._isotonic.predict([raw_prob])[0]
                return float(np.clip(calibrated, 0.01, 0.99))
        except Exception:
            pass
        return raw_prob

    def _compute_calibration_error(self):
        """Expected Calibration Error (ECE) across bins."""
        if len(self._raw_probs) < 20:
            return

        probs = np.array(self._raw_probs)
        outcomes = np.array(self._outcomes)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = outcomes[mask].mean()
            bin_conf = probs[mask].mean()
            ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)

        self.calibration_error = round(float(ece), 4)


class ProbabilityEngine:
    """
    Bayesian-inspired signal fusion with calibration support.

    Fuses multiple signal sources into calibrated probability.
    """

    DEFAULT_WEIGHTS = {
        'strategies': 0.30,
        'ml_prediction': 0.25,
        'order_flow': 0.15,
        'volatility': 0.10,
        'cross_exchange': 0.10,
        'pattern': 0.05,
        'correlation': 0.05,
    }

    def __init__(self, threshold: float = 0.65, calibration_method: str = 'platt'):
        self.threshold = threshold
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.calibrator = ProbabilityCalibrator(method=calibration_method)
        self.last_result: dict = {}

    def fuse(self, signals: dict) -> dict:
        """
        Fuse signal sources into a single calibrated probability.

        Args:
            signals: {source_name: {'direction': str, 'probability': float}}
        """
        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0

        for source, signal in signals.items():
            if not signal:
                continue
            direction = signal.get('direction', 'NEUTRAL')
            prob = signal.get('probability', 0.5)
            weight = self.weights.get(source, 0.1)

            if direction == 'LONG':
                long_score += prob * weight
            elif direction == 'SHORT':
                short_score += prob * weight
            total_weight += weight

        if total_weight > 0:
            long_prob = long_score / total_weight
            short_prob = short_score / total_weight
        else:
            long_prob = short_prob = 0.0

        # Apply calibration
        long_prob = self.calibrator.calibrate(long_prob)
        short_prob = self.calibrator.calibrate(short_prob)

        if long_prob > short_prob and long_prob > self.threshold:
            direction = 'LONG'
            probability = long_prob
        elif short_prob > long_prob and short_prob > self.threshold:
            direction = 'SHORT'
            probability = short_prob
        else:
            direction = None
            probability = max(long_prob, short_prob)

        self.last_result = {
            'direction': direction,
            'probability': round(probability, 4),
            'long_prob': round(long_prob, 4),
            'short_prob': round(short_prob, 4),
            'passed_threshold': direction is not None,
            'calibration_error': self.calibrator.calibration_error,
        }
        return self.last_result

    def record_outcome(self, predicted_prob: float, outcome: int):
        """Record prediction outcome for calibration learning."""
        self.calibrator.record(predicted_prob, outcome)
        if len(self.calibrator._raw_probs) % 50 == 0:
            self.calibrator.fit()
