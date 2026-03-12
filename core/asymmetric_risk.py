"""
asymmetric_risk.py — Dynamic Asymmetric Risk Engine

SECOND INNOVATION: Instead of fixed 2:1 or 3:1 R:R ratios,
compute the OPTIMAL TP/SL from actual market structure.

The insight: Support/resistance levels, ATR, and recent price
behavior tell you EXACTLY where the market is likely to stop
and reverse. Place TP just BEFORE the next barrier. Place SL
just BEYOND the nearest noise level.

This creates genuinely asymmetric trades where the TP has a
clear catalyst (price magnet) and the SL is placed where
the thesis is definitively wrong.

MATHEMATICAL BASIS:

  1. ATR Noise Floor: The minimum SL must exceed random noise.
     On 1-minute candles, noise ≈ 0.7-1.0 × ATR.
     SL below this = stopped out by noise = guaranteed losses.

  2. VWAP Magnet: Institutional traders execute around VWAP.
     Price reverts to VWAP more reliably than to any
     indicator-based level. Use as TP anchor.

  3. Swing Point Clustering: Recent highs/lows where price
     reversed create natural barriers. TP should target
     the nearest swing point in trade direction.

  4. Expected Value Optimization: Given the distance to TP
     and SL, plus the estimated win probability, compute
     whether the trade has positive expected value.
     ONLY allow trades where EV > 0.
"""

import numpy as np
from logger import get_logger

_log = get_logger("asym_risk")


class AsymmetricRiskEngine:
    """
    Computes optimal TP/SL from market structure.
    Ensures every trade has positive expected value.
    """

    def __init__(self, min_rr: float = 1.8, max_sl_pct: float = 0.012):
        self.min_rr = min_rr          # minimum acceptable R:R
        self.max_sl_pct = max_sl_pct  # max SL as % of entry (1.2%)

    def compute(self, side: str, entry: float, atr: float,
                df=None, win_probability: float = 0.55) -> dict:
        """
        Compute optimal TP/SL for a trade.

        Args:
            side: 'LONG' or 'SHORT'
            entry: entry price
            atr: current ATR value
            df: OHLCV DataFrame for swing point detection
            win_probability: estimated P(win) from quality score

        Returns:
            {tp, sl, sl_pct, rr_ratio, expected_value, is_positive_ev}
        """
        # Step 1: Compute noise-based minimum SL
        # SL must be > noise floor to avoid random stopouts
        noise_floor = atr * 1.0  # 1.0 ATR = typical 1-min noise range

        # Step 2: Find structural SL (beyond nearest swing point)
        structural_sl = noise_floor  # default
        if df is not None and len(df) > 10:
            structural_sl = self._find_structural_sl(side, entry, df, atr)

        # Step 3: Use the LARGER of noise floor and structural SL
        sl_dist = max(noise_floor, structural_sl)

        # Step 4: Cap SL to protect capital
        max_sl_dist = entry * self.max_sl_pct
        sl_dist = min(sl_dist, max_sl_dist)

        # Ensure minimum SL (spread + slippage buffer)
        min_sl_dist = entry * 0.002  # 0.2% minimum
        sl_dist = max(min_sl_dist, sl_dist)

        # Step 5: Find optimal TP using market structure
        tp_dist = sl_dist * self.min_rr  # default: minimum R:R
        if df is not None and len(df) > 10:
            structural_tp = self._find_structural_tp(side, entry, df, atr)
            if structural_tp > tp_dist:
                tp_dist = structural_tp  # use market structure if it's better

        # Step 6: Compute expected value
        # EV = P(win) × TP_dist - P(loss) × SL_dist
        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0
        ev = (win_probability * tp_dist) - ((1 - win_probability) * sl_dist)
        is_positive_ev = ev > 0

        # FIX: If EV is negative, DO NOT auto-adjust TP.
        # Wider TP with low win probability just means TP gets hit less often.
        # Negative EV = no trade. Period.
        if not is_positive_ev:
            _log.info(f"NEGATIVE EV: {ev:.4f} — trade blocked (p={win_probability:.2f} RR={rr_ratio:.1f})")

        # Compute actual prices
        if side == 'LONG':
            tp_price = entry + tp_dist
            sl_price = entry - sl_dist
        else:
            tp_price = entry - tp_dist
            sl_price = entry + sl_dist

        sl_pct = sl_dist / entry

        result = {
            'tp': round(tp_price, 2),
            'sl': round(sl_price, 2),
            'tp_dist': round(tp_dist, 4),
            'sl_dist': round(sl_dist, 4),
            'sl_pct': round(sl_pct, 6),
            'rr_ratio': round(rr_ratio, 2),
            'expected_value': round(ev, 4),
            'is_positive_ev': is_positive_ev,
            'noise_floor': round(noise_floor, 4),
            'win_probability': round(win_probability, 3),
        }

        _log.info(
            f"{side} entry={entry:.2f} | SL={sl_price:.2f}({sl_pct*100:.2f}%) "
            f"TP={tp_price:.2f} | RR={rr_ratio:.1f} EV={ev:.4f} "
            f"{'✓ +EV' if is_positive_ev else '✗ -EV BLOCKED'}"
        )

        return result

    def _find_structural_sl(self, side: str, entry: float,
                             df, atr: float) -> float:
        """
        Find the nearest swing point beyond which our thesis is wrong.
        SL goes just beyond this level.
        """
        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]

        if side == 'LONG':
            # For longs, SL below nearest swing low
            recent_lows = []
            for i in range(2, len(lows) - 2):
                if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and \
                   lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
                    recent_lows.append(float(lows[i]))

            if recent_lows:
                below_entry = [l for l in recent_lows if l < entry]
                if below_entry:
                    nearest_low = max(below_entry)  # highest low below entry
                    sl_dist = entry - nearest_low + atr * 0.2  # buffer beyond swing
                    return sl_dist
        else:
            # For shorts, SL above nearest swing high
            recent_highs = []
            for i in range(2, len(highs) - 2):
                if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and \
                   highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
                    recent_highs.append(float(highs[i]))

            if recent_highs:
                above_entry = [h for h in recent_highs if h > entry]
                if above_entry:
                    nearest_high = min(above_entry)
                    sl_dist = nearest_high - entry + atr * 0.2
                    return sl_dist

        return atr * 1.2  # fallback

    def _find_structural_tp(self, side: str, entry: float,
                             df, atr: float) -> float:
        """
        Find the nearest barrier in the profit direction.
        TP targets just before this barrier.
        """
        highs = df['high'].values[-30:]
        lows = df['low'].values[-30:]

        # Also compute VWAP as a price magnet
        try:
            typical = (df['high'] + df['low'] + df['close']) / 3
            cumvol = df['volume'].cumsum()
            vwap = float((typical * df['volume']).cumsum().iloc[-1] / cumvol.iloc[-1])
        except Exception:
            vwap = entry

        if side == 'LONG':
            # Target nearest resistance / swing high above entry
            targets = []
            for i in range(2, len(highs) - 2):
                if highs[i] >= highs[i-1] and highs[i] >= highs[i-2]:
                    h = float(highs[i])
                    if h > entry * 1.001:  # must be meaningfully above
                        targets.append(h)

            if vwap > entry * 1.001:
                targets.append(vwap)

            if targets:
                nearest = min(targets)
                tp_dist = nearest - entry - atr * 0.1  # slightly before barrier
                if tp_dist > atr * 0.5:
                    return tp_dist
        else:
            targets = []
            for i in range(2, len(lows) - 2):
                if lows[i] <= lows[i-1] and lows[i] <= lows[i-2]:
                    l = float(lows[i])
                    if l < entry * 0.999:
                        targets.append(l)

            if vwap < entry * 0.999:
                targets.append(vwap)

            if targets:
                nearest = max(targets)
                tp_dist = entry - nearest - atr * 0.1
                if tp_dist > atr * 0.5:
                    return tp_dist

        return atr * 2.5  # fallback
