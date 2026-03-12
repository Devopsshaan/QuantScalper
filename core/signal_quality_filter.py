"""
signal_quality_filter.py — Multi-factor signal quality scoring.

Combines independent signal sources into a calibrated quality score:

  Score = 0.25 * technical
        + 0.35 * ML_confidence
        + 0.20 * order_flow
        + 0.20 * volatility

Trades execute only if signal_score > threshold (default 0.70).

AUDIT FIXES:
  - Replaces ad-hoc grading in ai_scalper with proper composite scoring
  - Normalizes all inputs to [0, 1] before fusion
  - Provides rejection reasons for dashboard transparency
"""

from logger import log_signal


class SignalQualityFilter:
    """
    Filters weak signals before they reach the execution layer.

    Each factor is normalized to [0, 1]:
      technical:    vote_count / max_votes, adjusted by HTF/regime
      ml_confidence: raw model probability if aligned, else 1 - prob
      order_flow:   abs(imbalance) if direction matches, else penalty
      volatility:   regime suitability score
    """

    WEIGHTS = {
        'technical':    0.25,
        'ml_confidence': 0.35,
        'order_flow':   0.20,
        'volatility':   0.20,
    }

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold
        self.last_result: dict = {}

    def score(self, signal: str, context: dict) -> dict:
        """
        Compute composite signal quality score.

        Args:
            signal: 'LONG' or 'SHORT'
            context: Dict with keys:
                vote_count (int), max_votes (int),
                htf_confirmed (bool), regime_confidence (float),
                ml_prediction (dict), ml_accuracy (float),
                order_flow_imbalance (float), order_flow_pressure (str),
                atr_pct (float), regime (str)

        Returns:
            {score, grade, passed, breakdown, reject_reason}
        """
        # ── Technical factor ──
        vote_count = context.get('vote_count', 0)
        max_votes = context.get('max_votes', 4)
        htf_ok = context.get('htf_confirmed', False)
        regime_conf = context.get('regime_confidence', 0.5)

        # Base tech: even a weak technical signal gets a baseline score so it can pass if other factors are strong
        base_tech = 0.5 + (vote_count / max(max_votes, 1)) * 0.5
        htf_bonus = 0.15 if htf_ok else -0.10
        regime_bonus = (regime_conf - 0.5) * 0.2
        technical = max(0, min(1, base_tech + htf_bonus + regime_bonus))

        # ── ML confidence factor ──
        ml_pred = context.get('ml_prediction', {})
        ml_dir = ml_pred.get('direction', 'NEUTRAL')
        ml_conf = ml_pred.get('confidence', 0.5)
        ml_acc = context.get('ml_accuracy', 0.5)

        if ml_acc < 0.52:
            # Model not reliable — neutral
            ml_score = 0.5
        elif ml_dir == 'NEUTRAL':
            ml_score = 0.5
        elif (signal == 'LONG' and ml_dir == 'UP') or (signal == 'SHORT' and ml_dir == 'DOWN'):
            # Aligned
            ml_score = 0.5 + (ml_conf - 0.5) * ml_acc
        else:
            # Conflicting
            ml_score = 0.5 - (ml_conf - 0.5) * ml_acc * 0.8

        ml_score = max(0, min(1, ml_score))

        # ── Order flow factor ──
        imbalance = context.get('order_flow_imbalance', 0)
        pressure = context.get('order_flow_pressure', 'NEUTRAL')

        if signal == 'LONG':
            of_score = 0.5 + imbalance * 0.5   # imbalance +1 = score 1.0
        else:
            of_score = 0.5 - imbalance * 0.5   # imbalance -1 = score 1.0 for SHORT

        of_score = max(0, min(1, of_score))

        # ── Volatility environment factor ──
        regime = context.get('regime', 'QUIET')
        atr_pct = context.get('atr_pct', 0)

        vol_table = {
            'TRENDING_UP':   0.85 if signal == 'LONG' else 0.5,
            'TRENDING_DOWN': 0.85 if signal == 'SHORT' else 0.5,
            'RANGING':       0.70,
            'VOLATILE':      0.75,   # raised — volatility IS the opportunity for scalping
            'QUIET':         0.75,   # raised — allow quiet regime trades when config permits
        }
        vol_score = vol_table.get(regime, 0.55)

        # Extra penalty for very low ATR (no edge)
        if atr_pct < 0.01:
            vol_score *= 0.7

        vol_score = max(0, min(1, vol_score))

        # ── Composite score ──
        w = self.WEIGHTS
        composite = (
            w['technical'] * technical +
            w['ml_confidence'] * ml_score +
            w['order_flow'] * of_score +
            w['volatility'] * vol_score
        )
        composite = round(composite, 4)

        # ── Grade assignment ──
        if composite >= 0.85:
            grade, size_pct = 'A', 1.0
        elif composite >= 0.75:
            grade, size_pct = 'B', 0.75
        elif composite >= self.threshold:
            grade, size_pct = 'C', 0.50
        else:
            grade, size_pct = 'D', 0.0

        passed = composite >= self.threshold
        reject_reason = None if passed else f"score {composite:.3f} < threshold {self.threshold}"

        self.last_result = {
            'score': composite,
            'grade': grade,
            'size_pct': size_pct,
            'passed': passed,
            'reject_reason': reject_reason,
            'breakdown': {
                'technical': round(technical, 3),
                'ml_confidence': round(ml_score, 3),
                'order_flow': round(of_score, 3),
                'volatility': round(vol_score, 3),
            },
        }

        # Structured logging
        log_signal(
            symbol=context.get('symbol', '?'),
            direction=signal,
            grade=grade,
            score=composite,
            quality=composite,
            accepted=passed,
            breakdown=self.last_result['breakdown'],
        )

        return self.last_result


def regrade(quality: dict, threshold: float = 0.60) -> dict:
    """
    Re-assign grade after external score modifications (boosts).
    
    FIXES the bug where score gets boosted by vote/funding/edge
    but grade stays at 'D' from the original lower score.
    Must be called after any quality['score'] modification.
    """
    score = quality.get('score', 0)
    if score >= 0.85:
        quality['grade'] = 'A'
        quality['size_pct'] = 1.0
    elif score >= 0.75:
        quality['grade'] = 'B'
        quality['size_pct'] = 0.75
    elif score >= threshold:
        quality['grade'] = 'C'
        quality['size_pct'] = 0.50
    else:
        quality['grade'] = 'D'
        quality['size_pct'] = 0.0
    quality['passed'] = score >= threshold
    return quality
