"""
trade_graveyard.py — Anti-Pattern Memory

THIRD INNOVATION: Most bots learn from wins. This learns from LOSSES.

After every losing trade, the graveyard records the exact market
conditions that led to the loss (regime, volatility, entropy, 
order flow, time of day, etc.) as a "tombstone."

Before every new trade, it checks whether current conditions
match any tombstone. If similarity > threshold, it blocks the trade.

This is a form of NEGATIVE REINFORCEMENT LEARNING:
  - Traditional: "do more of what worked"  
  - This: "never repeat what failed"

The mathematical trick: market conditions that cause losses tend
to CLUSTER in specific regimes. A loss during low-entropy + 
high-vol + negative funding in Asian session is likely to repeat 
under the same conditions. By blocking those exact conditions,
you eliminate the most predictable source of losses.

Over time, the graveyard becomes a highly specific "don't trade" map
of dangerous market states, unique to this account's history.
"""

import time
import json
import numpy as np
from collections import deque
from pathlib import Path
from logger import get_logger

_log = get_logger("graveyard")

GRAVEYARD_FILE = Path(__file__).parent.parent / "data" / "trade_graveyard.json"


class TradeGraveyard:
    """
    Remembers losing conditions and blocks similar future trades.
    
    Each "tombstone" is a feature vector of the market state
    when a loss occurred. Before trading, we compute similarity
    between current state and all tombstones.
    """

    FEATURE_KEYS = [
        'regime',           # TRENDING/RANGING/VOLATILE/QUIET
        'entropy',          # market predictability (0-1)
        'hurst',            # trending tendency (0-1)  
        'atr_pct',          # volatility as % of price
        'vol_ratio',        # volume vs average
        'rsi',              # 0-100
        'imbalance',        # order book imbalance (-1 to 1)
        'hour_utc',         # 0-23
        'bb_width',         # bollinger band width
    ]

    def __init__(self, similarity_threshold: float = 0.75, max_tombstones: int = 100):
        self.similarity_threshold = similarity_threshold
        self.max_tombstones = max_tombstones
        self.tombstones: list[dict] = []
        self._load()

    def bury(self, conditions: dict, loss_amount: float, symbol: str = ""):
        """
        Record a losing trade's conditions as a tombstone.
        
        Args:
            conditions: dict with keys from FEATURE_KEYS
            loss_amount: negative PnL value
            symbol: which asset
        """
        features = self._extract_features(conditions)
        tombstone = {
            'features': features,
            'loss': round(loss_amount, 4),
            'symbol': symbol,
            'timestamp': time.time(),
            'count': 1,  # how many times this pattern has lost
        }

        # Check if similar tombstone already exists — strengthen it
        for existing in self.tombstones:
            sim = self._similarity(features, existing['features'])
            if sim > 0.85:
                existing['count'] += 1
                existing['loss'] += loss_amount
                existing['timestamp'] = time.time()
                _log.info(f"Strengthened tombstone (count={existing['count']}, total_loss=${existing['loss']:.2f})")
                self._save()
                return

        self.tombstones.append(tombstone)
        if len(self.tombstones) > self.max_tombstones:
            # Remove oldest/weakest tombstone
            self.tombstones.sort(key=lambda t: t['count'], reverse=True)
            self.tombstones = self.tombstones[:self.max_tombstones]

        _log.info(f"New tombstone buried | loss=${loss_amount:.2f} | total={len(self.tombstones)}")
        self._save()

    def should_block(self, conditions: dict) -> dict:
        """
        Check if current conditions match any tombstone.
        
        Returns:
            {blocked: bool, similarity: float, matching_tombstone: dict or None, reason: str}
        """
        if not self.tombstones:
            return {'blocked': False, 'similarity': 0, 'matching_tombstone': None, 'reason': 'no_history'}

        features = self._extract_features(conditions)
        max_sim = 0.0
        worst_match = None

        for tombstone in self.tombstones:
            sim = self._similarity(features, tombstone['features'])
            # Weight by loss count (repeated losses in same conditions = stronger block)
            weighted_sim = sim * min(1.0 + tombstone['count'] * 0.1, 2.0)
            if weighted_sim > max_sim:
                max_sim = weighted_sim
                worst_match = tombstone

        blocked = max_sim > self.similarity_threshold

        if blocked and worst_match:
            _log.warning(
                f"BLOCKED by tombstone | similarity={max_sim:.2f} | "
                f"pattern_losses={worst_match['count']} | total_lost=${worst_match['loss']:.2f}")

        return {
            'blocked': blocked,
            'similarity': round(max_sim, 3),
            'matching_tombstone': worst_match if blocked else None,
            'reason': f'matches_loss_pattern (sim={max_sim:.2f})' if blocked else 'clear',
        }

    def _extract_features(self, conditions: dict) -> list:
        """Convert conditions dict to normalized feature vector."""
        features = []
        for key in self.FEATURE_KEYS:
            val = conditions.get(key, 0)
            if isinstance(val, str):
                # Encode regime as number
                regime_map = {'TRENDING': 1.0, 'RANGING': 0.5, 'VOLATILE': 0.8, 'QUIET': 0.2}
                val = regime_map.get(val, 0.3)
            features.append(float(val))

        # Normalize to [0, 1] range
        normalizers = [1.0, 1.0, 1.0, 1.0, 3.0, 100.0, 2.0, 24.0, 0.1]
        normalized = []
        for i, (f, n) in enumerate(zip(features, normalizers)):
            normalized.append(min(1.0, max(0.0, abs(f) / max(n, 1e-9))))

        return normalized

    def _similarity(self, a: list, b: list) -> float:
        """
        Normalized Euclidean distance converted to similarity [0,1].
        FIX: cosine similarity between positive vectors is always >0.85.
        Euclidean distance properly measures actual difference.
        sim = 1 / (1 + euclidean_distance)
        """
        if len(a) != len(b) or not a:
            return 0.0
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        dist = np.sqrt(np.sum((a - b) ** 2))
        max_dist = np.sqrt(len(a))  # max possible distance is sqrt(N) since features are [0,1]
        
        # Scale dist from [0, max_dist] to similarity [1, 0] linearly
        # A perfectly identical state gets 1.0, extreme opposite gets 0.0
        similarity = max(0.0, 1.0 - (dist / max_dist))
        
        # Add an exponent to penalize distant points even more and reward close ones
        # This makes it harder to hit the 0.75 threshold by accident
        similarity = similarity ** 2
        
        return float(similarity)

    def _save(self):
        try:
            GRAVEYARD_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(GRAVEYARD_FILE, 'w') as f:
                json.dump(self.tombstones, f, indent=2, default=str)
        except Exception as e:
            _log.warning(f"Failed to save graveyard: {e}")

    def _load(self):
        try:
            if GRAVEYARD_FILE.exists():
                with open(GRAVEYARD_FILE) as f:
                    self.tombstones = json.load(f)
                    _log.info(f"Loaded {len(self.tombstones)} tombstones from graveyard")
        except Exception:
            self.tombstones = []

    def get_status(self) -> dict:
        return {
            'tombstone_count': len(self.tombstones),
            'total_losses_recorded': sum(t.get('count', 1) for t in self.tombstones),
            'total_loss_amount': round(sum(t.get('loss', 0) for t in self.tombstones), 2),
            'similarity_threshold': self.similarity_threshold,
        }
