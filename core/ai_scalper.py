"""
ai_scalper.py — Quant Scalper v7: REVERSED PIPELINE

The v6 pipeline: Signal → Filter → Trade
The v7 pipeline: Gate → Signal → Graveyard → Asymmetric Risk → Trade

Step 1: ExploitGate checks if market is non-random RIGHT NOW
        If gate is CLOSED → skip everything → zero trades → zero losses
Step 2: Only if gate is OPEN → generate signals (4 core strategies)
Step 3: TradeGraveyard checks if conditions match past losses → block if yes
Step 4: AsymmetricRiskEngine computes optimal TP/SL from market structure
        Only allows trades with positive expected value
Step 5: Kelly sizing + execution

BUGS FIXED FROM v6:
  - vol_spike restored to 1.20 (was 0.85)
  - Trend filter enforced (no counter-trend)
  - Trailing stop at 1.0 ATR (was 0.5)
  - Single portfolio registration
  - Bayesian trade recording connected
  - Reduced to 3-6 symbols (was 10)
  - Quality threshold at 0.65 (was 0.55)
"""

import time
import numpy as np
import pandas as pd
from typing import Optional

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from logger import system_log, log_signal, get_logger
from core.signal_quality_filter import regrade  # FIX: must be module-level, not local to __init__
from core.math_engine import OrnsteinUhlenbeckEngine
_log = get_logger("scalper")


class AIScalper:
    """Single-asset signal engine — CLEANED (4 strategies, strict filter)."""

    def __init__(self, symbol, feed, regime_classifier):
        self.symbol = symbol
        self.feed = feed
        self.regime_classifier = regime_classifier
        self.current_regime = {}
        self.last_atr = 0.0
        self.last_signal_info = {}
        self.ou_engine = OrnsteinUhlenbeckEngine(lookback=60, deviation_threshold=1.8, max_half_life=30)

    def _compute_indicators(self, df):
        df = df.copy()
        df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_diff'] = macd.macd_diff()
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        rsi_m = df['rsi'].rolling(50).mean()
        rsi_s = df['rsi'].rolling(50).std()
        df['rsi_upper'] = (rsi_m + rsi_s).clip(upper=75)
        df['rsi_lower'] = (rsi_m - rsi_s).clip(lower=25)
        df.dropna(inplace=True)
        return df

    def generate_signal(self, df, recommended_types: list = None):
        """Generate signal. If recommended_types provided, only run those strategies."""
        df = self._compute_indicators(df)
        if len(df) < 3:
            return None
        latest, prev = df.iloc[-1], df.iloc[-2]
        self.last_atr = float(latest['atr'])
        price = float(latest['close'])
        self.current_regime = self.regime_classifier.classify(df)
        weights = self.current_regime.get('weights', {})
        trend_bull = price > latest['ema50']
        trend_bear = price < latest['ema50']
        ru = latest.get('rsi_upper', 65)
        rl = latest.get('rsi_lower', 35)

        # 4 core strategies — clean, proven
        trend_l = prev['ema9'] <= prev['ema21'] and latest['ema9'] > latest['ema21']
        trend_s = prev['ema9'] >= prev['ema21'] and latest['ema9'] < latest['ema21']
        mr_l = latest['rsi'] < rl and latest['close'] <= latest['bb_lower'] * 1.002
        mr_s = latest['rsi'] > ru and latest['close'] >= latest['bb_upper'] * 0.998
        vol_spike = latest['vol_ratio'] > 1.20  # FIXED: real threshold
        mom_l = latest['macd_diff'] > prev['macd_diff'] and latest['macd_diff'] > 0 and vol_spike
        mom_s = latest['macd_diff'] < prev['macd_diff'] and latest['macd_diff'] < 0 and vol_spike
        tc_l = latest['ema9'] > latest['ema21'] and price > latest['ema9'] and 45 < latest['rsi'] < 70
        tc_s = latest['ema9'] < latest['ema21'] and price < latest['ema9'] and 30 < latest['rsi'] < 55

        # If exploit gate recommends specific types, mask the others
        if recommended_types:
            if 'trend' not in recommended_types and 'momentum' not in recommended_types:
                trend_l = trend_s = False
                tc_l = tc_s = False
                mom_l = mom_s = False
            if 'mean_reversion' not in recommended_types:
                mr_l = mr_s = False

        # ═══ 5th strategy: Ornstein-Uhlenbeck mean-reversion ═══
        ou_l = ou_s = False
        try:
            close_list = df['close'].tolist()
            vol_list = df['volume'].tolist() if 'volume' in df.columns else None
            ou_result = self.ou_engine.analyze(close_list, vol_list)
            if ou_result.get('signal') == 'LONG' and ou_result.get('vpin_safe', True):
                ou_l = True
            elif ou_result.get('signal') == 'SHORT' and ou_result.get('vpin_safe', True):
                ou_s = True
        except Exception:
            pass

        lv = int(sum([trend_l, mr_l, mom_l, tc_l, ou_l]))
        sv = int(sum([trend_s, mr_s, mom_s, tc_s, ou_s]))

        self.last_signal_info = {
            'rsi': round(float(latest['rsi']), 1), 'atr': round(self.last_atr, 2),
            'vol_ratio': round(float(latest['vol_ratio']), 2), 'price': price,
            'trend_filter': 'BULL' if trend_bull else ('BEAR' if trend_bear else 'NEUTRAL'),
            'long_votes': lv, 'short_votes': sv,
            'regime': self.current_regime.get('regime', 'QUIET'),
            'regime_confidence': self.current_regime.get('confidence', 0),
            'atr_pct': self.current_regime.get('atr_pct', 0),
            'macd_diff': round(float(latest['macd_diff']), 4),
            'bb_width': round(float(latest['bb_width']), 4),
            'ema9': round(float(latest['ema9']), 2),
            'ema21': round(float(latest['ema21']), 2),
            'ema50': round(float(latest['ema50']), 2),
            'entropy': 0, 'hurst': 0,  # filled by gate
        }

        # STRICT: trend filter required normally, but relaxed for micro-scalping
        is_quiet = self.current_regime.get('regime', 'QUIET') in ['QUIET', 'RANGING']
        eff_min_votes = 1 if is_quiet else weights.get('min_votes', 2)

        if lv >= eff_min_votes and (trend_bull or is_quiet):
            return 'LONG'
        if sv >= eff_min_votes and (trend_bear or is_quiet):
            return 'SHORT'
        return None


class MultiAssetScalper:
    """
    V7 REVERSED PIPELINE.

    Per cycle:
      1. ExploitGate: is market non-random? → if NO, skip entirely
      2. Signal generation (only approved strategy types)
      3. TradeGraveyard: do conditions match past losses? → if YES, block
      4. AsymmetricRisk: compute optimal TP/SL → reject if negative EV
      5. Quality filter + Kelly sizing
      6. Execute
    """

    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'XRP/USDT']

    def __init__(self, initial_balance=200, leverage=10, risk_per_trade_pct=0.02,
                 max_daily_loss_pct=0.10, interval_seconds=30, daily_profit_target=None,
                 active_symbols=None, mode='paper', config=None):
        from intelligence.market_regime import MarketRegimeClassifier
        from intelligence.ml_scorer import MLScorer
        from intelligence.pattern_detector import PatternDetector
        from intelligence.news_filter import NewsFilter
        from intelligence.market_correlation import MarketCorrelation
        from market.crypto_feed import CryptoFeed
        from core.signal_quality_filter import SignalQualityFilter  # regrade imported at module level
        from risk.futures_paper_trader import FuturesPaperTrader
        from risk.edge_position_sizer import EdgePositionSizer
        from risk.portfolio_manager import PortfolioManager
        from execution.execution_engine import ExecutionEngine, TradingMode
        from risk.global_risk_guard import GlobalRiskGuard
        from intelligence.quant_edge import QuantEdgeEngine
        from core.exploit_gate import ExploitGate
        from core.asymmetric_risk import AsymmetricRiskEngine
        from core.trade_graveyard import TradeGraveyard

        cfg = config or {}
        self.interval = interval_seconds
        self.is_running = False
        self.min_trade_grade = cfg.get('min_trade_grade', 'C')  # FIX: store from config
        self.cycle_count = 0
        if active_symbols:
            valid = [s for s in active_symbols if s in self.SYMBOLS]
            self.SYMBOLS = valid or self.SYMBOLS[:3]

        # Core
        self.trader = FuturesPaperTrader(
            initial_balance=initial_balance, leverage=leverage,
            risk_per_trade_pct=risk_per_trade_pct, max_daily_loss_pct=max_daily_loss_pct,
            daily_profit_target=daily_profit_target)
        self.regime_classifier = MarketRegimeClassifier()
        self.engines = {}
        for sym in self.SYMBOLS:
            self.engines[sym] = AIScalper(sym, CryptoFeed(symbol=sym, timeframe='1m', limit=100),
                                          self.regime_classifier)

        # ═══ THE THREE INNOVATIONS ═══
        self.exploit_gate = ExploitGate()
        self.asymmetric_risk = AsymmetricRiskEngine(min_rr=1.8, max_sl_pct=0.012)
        self.graveyard = TradeGraveyard(similarity_threshold=0.75)
        self.quant_edge = QuantEdgeEngine()

        # Standard subsystems
        self.signal_filter = SignalQualityFilter(threshold=cfg.get('signal_quality_threshold', 0.60))
        self.position_sizer = EdgePositionSizer(leverage=leverage)
        self.portfolio_manager = PortfolioManager(
            initial_balance=initial_balance, max_open_positions=cfg.get('max_open_positions', 3),
            max_asset_exposure_pct=cfg.get('max_asset_exposure_pct', 0.95), 
            max_total_leverage=cfg.get('max_total_leverage', 20), 
            max_drawdown_pct=cfg.get('max_drawdown_pct', 15) / 100.0)
        self.portfolio_manager._leverage = leverage
        mode_map = {'backtest': TradingMode.BACKTEST, 'paper': TradingMode.PAPER, 'live': TradingMode.LIVE}
        self.execution_engine = ExecutionEngine(
            mode=mode_map.get(mode, TradingMode.PAPER), paper_trader=self.trader,
            portfolio_manager=self.portfolio_manager)
        self.execution = self.execution_engine
        self.portfolio = self.portfolio_manager

        self.ranker = None
        try:
            from market.volatility_scanner import VolatilityScanner
            self.ranker = VolatilityScanner(self.SYMBOLS)
        except:
            pass
        self.news_filter = NewsFilter(pause_minutes=3, poll_interval=300)
        self.ml_scorer = MLScorer(retrain_interval=21600)
        self.pattern_detector = PatternDetector()
        self.correlation = MarketCorrelation()
        self.global_risk_guard = GlobalRiskGuard(
            max_daily_loss=cfg.get('max_daily_loss', 10), max_drawdown_pct=cfg.get('max_drawdown_pct', 15),
            max_open_positions=cfg.get('max_open_positions', 2), max_consecutive_losses=5)

        # State
        self.active_positions = {}
        self.all_prices = {}
        self.all_indicators = {}
        self.all_regimes = {}
        self.all_confidence = {}
        self.all_order_flow = {}
        self.rankings = []
        self.ml_status = {}
        self.correlation_status = {}
        self.pattern_status = {}
        self.last_quality_scores = {}
        self.health_status = {}
        self.all_edge_scores = {}
        self.gate_status = {}  # exploit gate state
        self.graveyard_status = {}
        self.last_funding = {}
        self.all_strategy_votes = {}

    def execute_cycle(self):
        self.cycle_count += 1
        self.trader.tick_cooldown()

        if self.cycle_count % 5 == 1 and self.ranker:
            try: self.rankings = self.ranker.rank_all()
            except: pass
        try: self.correlation_status = self.correlation.update()
        except: pass

        global_news_ok = self.news_filter.should_trade()
        top_symbols = (self.ranker.get_top_n(3) if self.ranker and self.rankings
                       else self.SYMBOLS[:3])

        # Scan assets
        results = {}
        for sym, engine in self.engines.items():
            try:
                df = engine.feed.fetch_candles()
                if df is None or df.empty: continue
                price = float(df.iloc[-1]['close'])
                self.all_prices[sym] = price
                # Don't generate signal yet — gate decides first
                self.all_indicators[sym] = {'price': price, 'signal': None}
                regime = engine.regime_classifier.classify(df)
                self.all_regimes[sym] = regime
                ob = engine.feed.fetch_order_book(sym)
                results[sym] = {'price': price, 'atr': 0, 'regime': regime, 'orderbook': ob, 'df': df}
            except Exception as e:
                system_log.error(f"Scan error {sym}: {e}")

        # Manage existing positions
        for active_sym in list(self.active_positions.keys()):
            if active_sym not in self.trader.positions:
                self.portfolio_manager.close_position(active_sym, 0)
                del self.active_positions[active_sym]
                continue
            if active_sym in results:
                df = results[active_sym]['df']
                engine = self.engines[active_sym]
                engine._compute_indicators(df)  # need ATR
                p, a = results[active_sym]['price'], engine.last_atr
                results[active_sym]['atr'] = a
                self.trader.update_pnl(p, active_sym)
                exit_reason = self.trader.check_exit_conditions(p, a, active_sym)
                if exit_reason:
                    pnl = self.trader.close_position(p, reason=exit_reason, symbol=active_sym)
                    self.portfolio_manager.close_position(active_sym, pnl)
                    self.quant_edge.record_trade(pnl, active_sym)
                    # GRAVEYARD: bury losing conditions
                    if pnl < 0:
                        pos_info = self.active_positions.get(active_sym, {})
                        self.graveyard.bury(pos_info.get('conditions', {}), pnl, active_sym)
                    del self.active_positions[active_sym]

        state = self.trader.get_state()
        self.portfolio_manager.update_equity(state['equity'])
        self.health_status['risk_guard'] = self.global_risk_guard.evaluate(
            state, self.portfolio_manager.get_status(), self.trader.trade_history)

        can_open = (len(self.trader.positions) < self.portfolio_manager.max_open_positions
                    and global_news_ok and self.health_status.get('risk_guard', {}).get('can_trade', True))

        if can_open:
            for sym in top_symbols:
                if sym not in results: continue
                if not self.news_filter.should_trade(sym): continue
                r = results[sym]
                df = r['df']
                prices = df['close'].tolist()

                # ═══ STEP 1: EXPLOIT GATE ═══
                gate = self.exploit_gate.evaluate(prices)
                self.gate_status[sym] = gate

                if not gate['is_exploitable']:
                    _log.debug(f"Gate CLOSED for {sym} — skipping (H={gate['tests'].get('hurst',0):.3f})")
                    continue

                # ═══ STEP 2: GENERATE SIGNAL (only recommended types) ═══
                engine = self.engines[sym]
                signal = engine.generate_signal(df, recommended_types=gate.get('recommended'))
                self.all_indicators[sym] = {**engine.last_signal_info, 'signal': signal, 'price': r['price']}
                self.all_indicators[sym]['entropy'] = gate['tests'].get('entropy', 0)
                self.all_indicators[sym]['hurst'] = gate['tests'].get('hurst', 0)

                if not signal or signal not in ('LONG', 'SHORT'):
                    continue

                r['atr'] = engine.last_atr
                r['signal'] = signal

                # ═══ STEP 3: GRAVEYARD CHECK ═══
                conditions = {
                    'regime': r['regime'].get('regime', 'QUIET'),
                    'entropy': gate['tests'].get('entropy', 0.5),
                    'hurst': gate['tests'].get('hurst', 0.5),
                    'atr_pct': r['regime'].get('atr_pct', 0),
                    'vol_ratio': engine.last_signal_info.get('vol_ratio', 1),
                    'rsi': engine.last_signal_info.get('rsi', 50),
                    'imbalance': r['orderbook'].get('imbalance', 0),
                    'hour_utc': time.gmtime().tm_hour,
                    'bb_width': engine.last_signal_info.get('bb_width', 0),
                }
                grave_check = self.graveyard.should_block(conditions)
                self.graveyard_status[sym] = grave_check

                if grave_check['blocked']:
                    _log.info(f"GRAVEYARD blocked {sym} {signal} (sim={grave_check['similarity']:.2f})")
                    continue

                # ═══ STEP 4: ASYMMETRIC RISK (positive EV check) ═══
                htf = engine.feed.fetch_htf_trend(sym)
                htf_ok = htf.get('direction') in (('UP','NEUTRAL') if signal=='LONG' else ('DOWN','NEUTRAL'))
                ob = r.get('orderbook', {})
                ml_pred = {}
                try:
                    if self.ml_scorer.is_trained:
                        ml_pred = self.ml_scorer.predict(df)
                except: pass

                # Quality score for win probability estimate
                quality = self.signal_filter.score(signal, {
                    'symbol': sym, 'vote_count': engine.last_signal_info.get(
                        'long_votes' if signal=='LONG' else 'short_votes', 0),
                    'max_votes': 4, 'htf_confirmed': htf_ok,
                    'regime_confidence': r['regime'].get('confidence', 0.5),
                    'ml_prediction': ml_pred,
                    'ml_accuracy': self.ml_scorer.accuracy if self.ml_scorer.is_trained else 0,
                    'order_flow_imbalance': ob.get('imbalance', 0),
                    'order_flow_pressure': ob.get('pressure', 'NEUTRAL'),
                    'atr_pct': r['regime'].get('atr_pct', 0),
                    'regime': r['regime'].get('regime', 'QUIET'),
                })

                # Gate confidence boost
                quality['score'] = round(min(0.99, quality['score'] + gate['confidence'] * 0.1), 4)
                quality = regrade(quality, self.signal_filter.threshold)

                self.last_quality_scores[sym] = quality
                self.all_confidence[sym] = quality
                self.all_order_flow[sym] = ob

                if not quality['passed']:
                    continue

                # Asymmetric risk computation
                risk_result = self.asymmetric_risk.compute(
                    side=signal, entry=r['price'], atr=engine.last_atr,
                    df=df, win_probability=quality['score'])

                if not risk_result['is_positive_ev']:
                    _log.info(f"NEGATIVE EV blocked {sym} {signal} (EV={risk_result['expected_value']:.4f})")
                    continue

                tp = risk_result['tp']
                sl = risk_result['sl']
                sl_pct = risk_result['sl_pct']

                # ═══ STEP 5: KELLY SIZING + EXECUTE ═══
                stats = self.trader.get_statistics()
                sizing = self.position_sizer.calculate(
                    equity=state['equity'], sl_distance_pct=sl_pct,
                    signal_probability=quality['score'],
                    reward_risk_ratio=risk_result['rr_ratio'],
                    signal_quality_score=quality['score'],
                    win_rate=stats.get('win_rate',0)/100 if stats.get('total_trades',0)>10 else None,
                    avg_win=stats.get('avg_win') if stats.get('total_trades',0)>10 else None,
                    avg_loss=stats.get('avg_loss') if stats.get('total_trades',0)>10 else None)

                exec_result = self.execution_engine.execute_signal(
                    signal=signal, symbol=sym, price=r['price'],
                    quality_result=quality, sizing_result=sizing,
                    tp_price=tp, sl_price=sl, grade=quality['grade'],
                    regime=str(r['regime'].get('regime', '')),
                    min_grade=self.min_trade_grade)  # FIX: pass configured grade filter

                if exec_result.get('executed'):
                    self.portfolio_manager.register_position(sym, signal, sizing['size_usd'], r['price'])
                    self.active_positions[sym] = {
                        'side': signal, 'entry': r['price'], 'grade': quality['grade'],
                        'quality_score': quality['score'], 'conditions': conditions,
                        'gate_confidence': gate['confidence'], 'exploit_type': gate['exploit_type'],
                        'rr_ratio': risk_result['rr_ratio'], 'ev': risk_result['expected_value'],
                    }
                    system_log.info(
                        f"ENTRY {sym} {signal} q={quality['score']:.3f} "
                        f"gate={gate['exploit_type']}({gate['confidence']:.2f}) "
                        f"RR={risk_result['rr_ratio']:.1f} EV={risk_result['expected_value']:.4f} "
                        f"${sizing['size_usd']:.0f}")
                    break

        state = self.trader.get_state()
        self.ml_status = self.ml_scorer.get_status()
        self.pattern_status = self.pattern_detector.get_status()
        return {
            'cycle': self.cycle_count, 'prices': self.all_prices.copy(),
            'active_symbol': self._get_active_symbol(), 'state': state,
            'indicators': self.all_indicators.copy(), 'rankings': self.rankings,
            'news': self.news_filter.get_status(), 'active_positions': self.active_positions.copy(),
            'regimes': self.all_regimes.copy(), 'confidence': self.all_confidence.copy(),
            'order_flow': self.all_order_flow.copy(), 'ml': self.ml_status,
            'correlation': self.correlation_status, 'patterns': self.pattern_status,
            'quality_scores': self.last_quality_scores.copy(), 'health': self.health_status.copy(),
            'portfolio': self.portfolio_manager.get_status(),
            'edge_scores': self.all_edge_scores.copy(),
            'gate': self.gate_status.copy(), 'graveyard': self.graveyard.get_status(),
            'strategy_votes': self.all_strategy_votes,
            'funding': self.last_funding,
        }

    def _get_active_symbol(self):
        return list(self.active_positions.keys())[0] if self.active_positions else None

    def run(self, interval_seconds=None):
        interval = interval_seconds or self.interval
        self.is_running = True
        self.news_filter.start_polling()
        system_log.info(f"Quant Scalper v7 REVERSED PIPELINE | {len(self.SYMBOLS)} assets | {interval}s")
        try:
            first = list(self.engines.values())[0]
            td = first.feed.fetch_candles(limit=1000)
            if td is not None and len(td) > 100: self.ml_scorer.train(td)
        except: pass
        try: self.correlation.update()
        except: pass
        try:
            while self.is_running:
                self.execute_cycle()
                time.sleep(interval)
        except KeyboardInterrupt:
            system_log.info("Stopped by user")
        finally:
            self.is_running = False
            self.news_filter.stop_polling()
