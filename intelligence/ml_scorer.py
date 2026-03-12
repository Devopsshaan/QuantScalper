"""
ml_scorer.py — ML signal scorer with proper feature preprocessing.

TASK 6 UPGRADES:
  - Z-score normalization for all features
  - Rolling normalization (avoids look-ahead bias)
  - Robust feature scaling (handles outliers)
  - NaN/inf guards at every stage
  - Retraining with validation split for honest accuracy

EXISTING FEATURES PRESERVED:
  - GradientBoosting classifier
  - Auto-retrain on interval
  - Confidence boost/penalty for signal pipeline
"""

import time
import threading
import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from logger import ml_log


FEATURE_COLS = [
    'rsi_z', 'macd_z', 'macd_hist_z', 'atr_norm',
    'bb_position', 'bb_width_z', 'vol_ratio_z',
    'ema9_slope', 'ema21_slope', 'ema50_slope',
    'roc_5', 'roc_10', 'hour_sin', 'hour_cos',
    'price_vs_ema50',
]


class MLScorer:
    """ML signal scorer with robust feature preprocessing."""

    def __init__(self, retrain_interval: int = 21600):
        self.model = None
        self.scaler = RobustScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.accuracy = 0.0
        self.val_accuracy = 0.0
        self.feature_importance: dict[str, float] = {}
        self.last_prediction: dict = {}
        self.train_samples = 0
        self.retrain_interval = retrain_interval
        self.last_train_time = 0
        self._lock = threading.Lock()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ML features with z-score and rolling normalization."""
        feat = df.copy()

        # Raw indicators
        feat['rsi'] = RSIIndicator(close=feat['close'], window=14).rsi()

        macd = MACD(close=feat['close'], window_slow=26, window_fast=12, window_sign=9)
        feat['macd'] = macd.macd()
        feat['macd_hist'] = macd.macd_diff()

        atr = AverageTrueRange(high=feat['high'], low=feat['low'],
                                close=feat['close'], window=14)
        feat['atr_norm'] = atr.average_true_range() / feat['close'] * 100

        bb = BollingerBands(close=feat['close'], window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_range = (bb_upper - bb_lower).replace(0, 1)
        feat['bb_width'] = bb_range / feat['close'] * 100
        feat['bb_position'] = (feat['close'] - bb_lower) / bb_range

        vol_ma = feat['volume'].rolling(20).mean().replace(0, 1)
        feat['vol_ratio'] = feat['volume'] / vol_ma

        # EMA slopes
        feat['ema9'] = EMAIndicator(close=feat['close'], window=9).ema_indicator()
        feat['ema21'] = EMAIndicator(close=feat['close'], window=21).ema_indicator()
        feat['ema50'] = EMAIndicator(close=feat['close'], window=50).ema_indicator()
        feat['ema9_slope'] = feat['ema9'].pct_change(3) * 100
        feat['ema21_slope'] = feat['ema21'].pct_change(3) * 100
        feat['ema50_slope'] = feat['ema50'].pct_change(5) * 100

        # Momentum
        feat['roc_5'] = feat['close'].pct_change(5) * 100
        feat['roc_10'] = feat['close'].pct_change(10) * 100

        # Time cyclical
        if 'timestamp' in feat.columns:
            hours = pd.to_datetime(feat['timestamp']).dt.hour
        else:
            hours = pd.Series(range(len(feat))) % 24
        feat['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        feat['price_vs_ema50'] = (feat['close'] - feat['ema50']) / feat['ema50'].replace(0, 1) * 100

        # ── TASK 6: Rolling z-score normalization ──
        # Prevents look-ahead bias by normalizing within a rolling window
        window = 50
        for col, z_col in [('rsi', 'rsi_z'), ('macd', 'macd_z'),
                           ('macd_hist', 'macd_hist_z'),
                           ('bb_width', 'bb_width_z'),
                           ('vol_ratio', 'vol_ratio_z')]:
            roll_mean = feat[col].rolling(window).mean()
            roll_std = feat[col].rolling(window).std().replace(0, 1)
            feat[z_col] = (feat[col] - roll_mean) / roll_std

        feat.dropna(inplace=True)

        # Guard: replace inf
        feat.replace([np.inf, -np.inf], 0, inplace=True)

        return feat

    def _compute_labels(self, df: pd.DataFrame, lookahead: int = 5,
                         threshold: float = 0.08) -> pd.Series:
        future_return = df['close'].shift(-lookahead) / df['close'] - 1
        future_pct = future_return * 100
        labels = pd.Series(0, index=df.index)
        labels[future_pct > threshold] = 1
        labels[future_pct < -threshold] = -1
        return labels

    def train(self, df: pd.DataFrame) -> dict:
        """Train with TimeSeriesSplit for honest out-of-sample accuracy."""
        if not HAS_SKLEARN:
            return {'trained': False, 'reason': 'scikit-learn not installed'}

        try:
            ml_log.info("Computing features for training...")
            feat_df = self.compute_features(df)
            if len(feat_df) < 100:
                return {'trained': False, 'reason': f'insufficient_data ({len(feat_df)})'}

            labels = self._compute_labels(feat_df)
            valid = labels != 0
            feat_df = feat_df[valid].iloc[:-5]
            labels = labels[valid].iloc[:-5]

            if len(feat_df) < 50:
                return {'trained': False, 'reason': f'insufficient_labeled ({len(feat_df)})'}

            X = feat_df[FEATURE_COLS].values
            y = labels.values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

            # ── TASK 6: RobustScaler (handles outliers better than StandardScaler) ──
            X_scaled = self.scaler.fit_transform(X)

            # ── TASK 7 spirit: TimeSeriesSplit for honest validation ──
            tscv = TimeSeriesSplit(n_splits=3)
            val_scores = []

            for train_idx, val_idx in tscv.split(X_scaled):
                fold_model = GradientBoostingClassifier(
                    n_estimators=80, max_depth=3, learning_rate=0.1,
                    subsample=0.8, random_state=42)
                fold_model.fit(X_scaled[train_idx], y[train_idx])
                preds = fold_model.predict(X_scaled[val_idx])
                val_scores.append(accuracy_score(y[val_idx], preds))

            val_accuracy = np.mean(val_scores) if val_scores else 0

            # Final model on all data
            model = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42)
            model.fit(X_scaled, y)

            importance = dict(zip(FEATURE_COLS, model.feature_importances_))
            sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            with self._lock:
                self.model = model
                self.is_trained = True
                self.accuracy = round(float(val_accuracy), 4)
                self.val_accuracy = round(float(val_accuracy), 4)
                self.feature_importance = {k: round(v, 4) for k, v in sorted_imp.items()}
                self.train_samples = len(X)
                self.last_train_time = time.time()

            ml_log.info(
                f"Trained: val_accuracy={val_accuracy:.1%} samples={len(X)} "
                f"top_features={list(sorted_imp.keys())[:3]}")
            return {'trained': True, 'accuracy': self.accuracy, 'samples': len(X)}

        except Exception as e:
            ml_log.error(f"Training error: {e}")
            return {'trained': False, 'reason': str(e)}

    def predict(self, df: pd.DataFrame) -> dict:
        if not self.is_trained or self.model is None:
            return {'direction': 'NEUTRAL', 'up_prob': 0.5, 'down_prob': 0.5, 'confidence': 0.0}

        try:
            feat_df = self.compute_features(df)
            if len(feat_df) < 1:
                return {'direction': 'NEUTRAL', 'up_prob': 0.5, 'down_prob': 0.5, 'confidence': 0.0}

            X = feat_df[FEATURE_COLS].iloc[-1:].values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            X_scaled = self.scaler.transform(X)

            proba = self.model.predict_proba(X_scaled)[0]
            classes = list(self.model.classes_)

            up_prob = float(proba[classes.index(1)]) if 1 in classes else 0.0
            down_prob = float(proba[classes.index(-1)]) if -1 in classes else 0.0

            if up_prob > 0.6:
                direction, confidence = 'UP', up_prob
            elif down_prob > 0.6:
                direction, confidence = 'DOWN', down_prob
            else:
                direction, confidence = 'NEUTRAL', max(up_prob, down_prob)

            result = {
                'direction': direction,
                'up_prob': round(up_prob, 3),
                'down_prob': round(down_prob, 3),
                'confidence': round(confidence, 3),
            }
            with self._lock:
                self.last_prediction = result
            return result

        except Exception as e:
            ml_log.error(f"Prediction error: {e}")
            return {'direction': 'NEUTRAL', 'up_prob': 0.5, 'down_prob': 0.5, 'confidence': 0.0}

    def should_retrain(self) -> bool:
        if not self.is_trained:
            return True
        return time.time() - self.last_train_time > self.retrain_interval

    def get_confidence_boost(self, signal: str, prediction: dict) -> float:
        direction = prediction.get('direction', 'NEUTRAL')
        if direction == 'NEUTRAL':
            return 1.0
        if (signal == 'LONG' and direction == 'UP') or (signal == 'SHORT' and direction == 'DOWN'):
            return 1.0 + (prediction.get('confidence', 0.5) - 0.5) * 0.6
        else:
            return 1.0 - (prediction.get('confidence', 0.5) - 0.5) * 0.8

    def get_status(self) -> dict:
        with self._lock:
            return {
                'trained': self.is_trained,
                'accuracy': self.accuracy,
                'val_accuracy': self.val_accuracy,
                'samples': self.train_samples,
                'top_features': list(self.feature_importance.keys())[:5],
                'last_prediction': self.last_prediction.copy(),
                'model_age_s': int(time.time() - self.last_train_time) if self.last_train_time else -1,
            }
