"""
strategy_voting_engine.py — Quant Scalper v6

Multi-strategy ensemble voting: collect signals from registered strategies,
aggregate votes, and produce a confidence-weighted final trade decision.

Signal schema per strategy:
    {"direction": "buy" | "sell" | "hold", "confidence": float}

Output schema:
    {
        "decision": "buy" | "sell" | "hold",
        "confidence": float,          # winning_votes / total_votes
        "votes": {strategy_name: direction, ...},
        "raw_votes": {strategy_name: {direction, confidence}, ...},
    }

Safety guarantees:
  • Any strategy that raises or returns bad data → defaults to "hold" with 0.0 confidence
  • Never crashes the trading loop
  • Works with any subset of strategies registered
"""

import logging
from typing import Any, Dict

_log = logging.getLogger("trading")


# ── Direction constants ──────────────────────────────────────────────────────
BUY  = "buy"
SELL = "sell"
HOLD = "hold"

_VALID_DIRECTIONS = {BUY, SELL, HOLD}

# Default signal when a strategy is missing or fails
_DEFAULT_SIGNAL: Dict[str, Any] = {"direction": HOLD, "confidence": 0.0}


class StrategyVotingEngine:
    """
    Ensemble voting engine for Quant Scalper v6.

    Usage:
        engine = StrategyVotingEngine()

        signals = {
            "trend":      {"direction": "buy",  "confidence": 0.8},
            "liquidity":  {"direction": "buy",  "confidence": 0.7},
            "orderbook":  {"direction": "hold", "confidence": 0.5},
            "volatility": {"direction": "sell", "confidence": 0.6},
            "funding":    {"direction": "buy",  "confidence": 0.65},
        }

        result = engine.vote(signals)
        # result["decision"]   → "buy"
        # result["confidence"] → 0.6   (3/5 votes)
        # result["votes"]      → {trend: buy, liquidity: buy, ...}
    """

    # Registered strategy keys (determines vote weight and display order)
    STRATEGY_KEYS = [
        "trend",
        "liquidity",
        "orderbook",
        "volatility",
        "funding",
        "imbalance_detector",
        "liquidation_cascade",
        "vol_breakout"
    ]

    def __init__(self):
        self._last_result: Dict[str, Any] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def vote(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate strategy signals and return a voted decision.

        Parameters
        ----------
        signals : dict
            Keys are strategy names (see STRATEGY_KEYS).
            Values are dicts with at least {"direction": str, "confidence": float}.
            Missing strategies → defaulted to "hold".

        Returns
        -------
        dict with keys:
            decision    : "buy" | "sell" | "hold"
            confidence  : float  (winning_vote_count / total_strategies)
            votes       : {strategy: direction}
            raw_votes   : {strategy: {direction, confidence}}
        """
        votes: Dict[str, str]        = {}
        raw_votes: Dict[str, Dict]   = {}

        # ── 1. Collect and normalise signals ─────────────────────────────
        for key in self.STRATEGY_KEYS:
            raw = signals.get(key)
            normalised = self._normalise_signal(key, raw)
            votes[key]     = normalised["direction"]
            raw_votes[key] = normalised

        # ── 2. Count buy / sell votes ─────────────────────────────────────
        total_votes = len(self.STRATEGY_KEYS)            # always 5
        buy_count   = sum(1 for d in votes.values() if d == BUY)
        sell_count  = sum(1 for d in votes.values() if d == SELL)

        # ── 3. Determine dominant direction with Strict Consensus ──
        # Fix: Need at LEAST 2 independent strategies to signal BUY or SELL
        MIN_VOTES = 2
        
        if buy_count >= MIN_VOTES and buy_count > sell_count:
            decision   = BUY
            confidence = round(buy_count / total_votes, 4)
        elif sell_count >= MIN_VOTES and sell_count > buy_count:
            decision   = SELL
            confidence = round(sell_count / total_votes, 4)
        else:
            # Tie (or all hold) → hold
            decision   = HOLD
            confidence = round(max(buy_count, sell_count) / total_votes, 4)

        result = {
            "decision":   decision,
            "confidence": confidence,
            "votes":      votes,
            "raw_votes":  raw_votes,
            "buy_count":  buy_count,
            "sell_count": sell_count,
            "total":      total_votes,
        }
        self._last_result = result
        return result

    def get_last_result(self) -> Dict[str, Any]:
        """Return the most recent voting result (empty dict if not yet voted)."""
        return self._last_result.copy()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _normalise_signal(self, key: str, raw: Any) -> Dict[str, Any]:
        """
        Validate and normalise a single strategy signal.
        Any error → default "hold" (never raises).
        """
        if raw is None:
            return _DEFAULT_SIGNAL.copy()

        try:
            if not isinstance(raw, dict):
                _log.debug(f"[VOTE] {key}: non-dict signal → hold default")
                return _DEFAULT_SIGNAL.copy()

            direction = str(raw.get("direction", HOLD)).lower().strip()
            if direction not in _VALID_DIRECTIONS:
                _log.debug(f"[VOTE] {key}: invalid direction '{direction}' → hold")
                direction = HOLD

            try:
                confidence = float(raw.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.0

            return {"direction": direction, "confidence": confidence}

        except Exception as exc:  # absolute safety net
            _log.warning(f"[VOTE] {key}: signal normalisation failed ({exc}) → hold")
            return _DEFAULT_SIGNAL.copy()

    # ── Adapter helpers ───────────────────────────────────────────────────────

    @staticmethod
    def signal_from_trend(trend_long: bool, trend_short: bool, confidence: float = 0.65) -> Dict:
        """Build a signal dict from boolean trend flags."""
        if trend_long:
            return {"direction": BUY,  "confidence": confidence}
        if trend_short:
            return {"direction": SELL, "confidence": confidence}
        return {"direction": HOLD, "confidence": 0.0}

    @staticmethod
    def signal_from_liquidity(liquidity_result: Dict) -> Dict:
        """Convert LiquiditySweepScalp.detect() output → voting signal."""
        if not isinstance(liquidity_result, dict):
            return _DEFAULT_SIGNAL.copy()
        raw_sig  = str(liquidity_result.get("signal") or "").upper()
        conf     = float(liquidity_result.get("confidence", 0.0))
        direction = BUY if raw_sig == "LONG" else (SELL if raw_sig == "SHORT" else HOLD)
        return {"direction": direction, "confidence": conf}

    @staticmethod
    def signal_from_orderbook(ob: Dict) -> Dict:
        """Convert order book imbalance dict → voting signal."""
        if not isinstance(ob, dict):
            return _DEFAULT_SIGNAL.copy()
        pressure = str(ob.get("pressure", "NEUTRAL")).upper()
        imbalance = float(ob.get("imbalance", 0.0))
        confidence = min(0.9, abs(imbalance))
        if pressure == "BUY":
            return {"direction": BUY,  "confidence": confidence}
        if pressure == "SELL":
            return {"direction": SELL, "confidence": confidence}
        return {"direction": HOLD, "confidence": 0.0}

    @staticmethod
    def signal_from_volatility(regime: Dict) -> Dict:
        """
        Volatility expansion signal:
          - VOLATILE  + UP   → buy
          - VOLATILE  + DOWN → sell
          - Otherwise        → hold
        """
        if not isinstance(regime, dict):
            return _DEFAULT_SIGNAL.copy()
        r     = str(regime.get("regime", "")).upper()
        bias  = str(regime.get("bias",   "NEUTRAL")).upper()
        conf  = float(regime.get("confidence", 0.5))
        if r == "VOLATILE":
            if bias == "UP":
                return {"direction": BUY,  "confidence": conf}
            if bias == "DOWN":
                return {"direction": SELL, "confidence": conf}
        return {"direction": HOLD, "confidence": 0.0}

    @staticmethod
    def signal_from_funding(funding: Dict, base_side: str) -> Dict:
        """Convert funding rate boost info → voting signal that aligns direction."""
        if not isinstance(funding, dict):
            return _DEFAULT_SIGNAL.copy()
        boost = float(funding.get("boost", 0.0))
        rate  = float(funding.get("rate",  0.0))
        if boost > 0.0:
            # boost > 0 means funding aligned with base_side
            direction = BUY if base_side in ("LONG", "buy") else SELL
            return {"direction": direction, "confidence": min(0.9, 0.5 + boost * 5)}
        return {"direction": HOLD, "confidence": 0.0}

    @staticmethod
    def signal_from_dict(strat_result: Dict) -> Dict:
        """Generic adapter for modular bot strategies returning {'signal', 'confidence'}"""
        if not isinstance(strat_result, dict):
            return _DEFAULT_SIGNAL.copy()
            
        raw_sig = str(strat_result.get("signal") or "").upper()
        conf = float(strat_result.get("confidence", 0.0))
        
        if raw_sig == "LONG":
            direction = BUY
        elif raw_sig == "SHORT":
            direction = SELL
        else:
            direction = HOLD
            
        return {"direction": direction, "confidence": conf}
