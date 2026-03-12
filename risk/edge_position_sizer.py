"""
edge_position_sizer.py — Adaptive Kelly-based position sizing.

TASK 3: Size positions proportional to signal probability and edge.

Kelly formula:
    f = p - (1 - p) / reward_risk

Uses half-Kelly for safety. Stronger signals get larger allocation.

AUDIT FIXES:
  - Replaces fixed percentage sizing with edge-aware sizing
  - Scales with signal quality score (stronger signals → larger size)
  - Caps at configurable maximum to prevent over-leverage
  - Degrades gracefully when win rate data is insufficient
"""

from logger import risk_log


class EdgePositionSizer:
    """
    Kelly-criterion position sizer that scales with edge.

    Uses historical win rate + signal quality to compute optimal size.
    Half-Kelly provides a safety margin against estimation error.
    """

    def __init__(self, leverage: int = 10, max_risk_pct: float = 0.05,
                 min_risk_pct: float = 0.005, kelly_fraction: float = 0.5):
        self.leverage = leverage
        self.max_risk_pct = max_risk_pct     # absolute cap
        self.min_risk_pct = min_risk_pct     # minimum viable trade
        self.kelly_fraction = kelly_fraction  # half-Kelly by default
        self.default_risk_pct = 0.02         # fallback when no data

    def calculate(self, equity: float, sl_distance_pct: float,
                  signal_probability: float = 0.5,
                  reward_risk_ratio: float = 2.0,
                  signal_quality_score: float = 0.5,
                  win_rate: float = None,
                  avg_win: float = None,
                  avg_loss: float = None) -> dict:
        """
        Compute position size using adaptive Kelly sizing.

        Args:
            equity: Current account equity
            sl_distance_pct: Stop loss distance as fraction of price
            signal_probability: Estimated probability of winning
            reward_risk_ratio: Target R:R ratio
            signal_quality_score: Composite quality score (0-1)
            win_rate: Historical win rate (overrides probability if provided)
            avg_win / avg_loss: Historical averages for Kelly calc
        """
        if equity <= 0 or sl_distance_pct <= 0:
            return self._result(0, 0, 0, 'zero_equity_or_sl')

        # ── Determine optimal risk fraction ──
        # Use historical data if available, else use signal probability
        p = win_rate if win_rate and win_rate > 0 else signal_probability
        rr = reward_risk_ratio

        # If we have actual win/loss data, compute Kelly from that
        if avg_win and avg_loss and abs(avg_loss) > 0:
            b = avg_win / abs(avg_loss)  # ratio of average win to average loss
            kelly_f = (b * p - (1 - p)) / b
        else:
            # Kelly using R:R ratio
            kelly_f = p - (1 - p) / rr

        # Apply fraction (half-Kelly for safety)
        risk_pct = kelly_f * self.kelly_fraction

        # Scale by signal quality (strong signals get full allocation)
        quality_mult = 0.3 + 0.7 * signal_quality_score  # range 0.3 to 1.0
        risk_pct *= quality_mult

        # Clamp to bounds
        if risk_pct <= 0:
            # Negative Kelly = no edge — use minimum or skip
            risk_pct = self.min_risk_pct * signal_quality_score
            if risk_pct < self.min_risk_pct * 0.5:
                return self._result(0, 0, 0, 'no_edge')

        risk_pct = max(self.min_risk_pct, min(self.max_risk_pct, risk_pct))

        # ── Convert to position size ──
        risk_usd = equity * risk_pct
        position_size_usd = risk_usd / sl_distance_pct
        max_notional = equity * self.leverage * 0.95
        position_size_usd = min(position_size_usd, max_notional)

        margin_usd = position_size_usd / self.leverage

        risk_log.info(
            f"Kelly sizing: p={p:.2f} rr={rr:.1f} kelly_f={kelly_f:.4f} "
            f"risk%={risk_pct:.4f} quality={signal_quality_score:.2f} "
            f"size=${position_size_usd:.0f}"
        )

        return self._result(
            position_size_usd, margin_usd, risk_usd,
            method='kelly',
            risk_pct=risk_pct,
            kelly_raw=kelly_f,
            quality_mult=quality_mult,
            signal_probability=p,
        )

    def _result(self, size_usd, margin_usd, risk_usd, method='none', **extra):
        return {
            'size_usd': round(size_usd, 2),
            'margin_usd': round(margin_usd, 2),
            'risk_usd': round(risk_usd, 2),
            'method': method,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in extra.items()},
        }
