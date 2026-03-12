import pandas as pd 
from ta.trend import ADXIndicator, EMAIndicator 
from ta.volatility import BollingerBands, AverageTrueRange 
 
class MarketRegime: 
    TRENDING = 'TRENDING' 
    RANGING = 'RANGING' 
    VOLATILE = 'VOLATILE' 
    QUIET = 'QUIET' 
 
REGIME_WEIGHTS = { 
    MarketRegime.TRENDING: {'rr_ratio': 2.4, 'size_mult': 1.0, 'min_votes': 2}, 
    MarketRegime.RANGING: {'rr_ratio': 1.8, 'size_mult': 0.8, 'min_votes': 2}, 
    MarketRegime.VOLATILE: {'rr_ratio': 1.6, 'size_mult': 0.7, 'min_votes': 1}, 
    MarketRegime.QUIET: {'rr_ratio': 1.4, 'size_mult': 0.5, 'min_votes': 1}, 
} 
 
class MarketRegimeClassifier: 
    def __init__(self, adx_period=14, bb_period=20, lookback=50): 
        self.adx_period = adx_period 
        self.bb_period = bb_period 
        self.lookback = lookback 
        self.last_regime = MarketRegime.QUIET 
 
    def classify(self, df): 
        if df is None or len(df) < max(self.lookback, 60): 
            return self._default_result() 
        df = df.copy() 
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=self.adx_period) 
        df['adx'] = adx.adx() 
        df['di_plus'] = adx.adx_pos() 
        df['di_minus'] = adx.adx_neg() 
        bb = BollingerBands(close=df['close'], window=self.bb_period, window_dev=2) 
        df['bb_upper'] = bb.bollinger_hband() 
        df['bb_lower'] = bb.bollinger_lband() 
        df['bb_mid'] = bb.bollinger_mavg() 
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, 1) 
        df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator() 
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range() 
        df['atr_pct'] = df['atr'] / df['close'].replace(0, 1) * 100 
        df.dropna(inplace=True) 
        if len(df) < 3: 
            return self._default_result() 
        latest = df.iloc[-1] 
        bb_baseline = float(df['bb_width'].tail(self.lookback).mean()) 
        bb_width = float(latest['bb_width']) 
        atr_pct = float(latest['atr_pct']) 
        adx_value = float(latest['adx']) 
        price = float(latest['close']) 
        ema50 = float(latest['ema50']) 
        bias = 'UP' if price > ema50 and float(latest['di_plus']) >= float(latest['di_minus']) else ('DOWN' if price < ema50 and float(latest['di_minus']) > float(latest['di_plus']) else 'NEUTRAL') 
        if atr_pct < 0.08 and bb_width < bb_baseline * 0.8: 
            regime, confidence = MarketRegime.QUIET, 0.65 
        elif adx_value >= 25 and bias != 'NEUTRAL': 
            regime, confidence = MarketRegime.TRENDING, min(0.95, 0.6 + (adx_value - 25) / 40) 
        elif atr_pct >= 0.6 or bb_width > bb_baseline * 1.6: 
            regime, confidence = MarketRegime.VOLATILE, 0.7 
        else: 
            regime, confidence = MarketRegime.RANGING, 0.62 
        weights = REGIME_WEIGHTS[regime] 
        self.last_regime = regime 
        return {'regime': regime, 'bias': bias, 'confidence': round(confidence, 3), 'adx': round(adx_value, 2), 'bb_width': round(bb_width, 4), 'atr_pct': round(atr_pct, 4), 'weights': weights, 'rr_ratio': weights['rr_ratio'], 'size_mult': weights['size_mult'], 'min_votes': weights['min_votes']} 
 
    def _default_result(self): 
        weights = REGIME_WEIGHTS[MarketRegime.QUIET] 
        return {'regime': MarketRegime.QUIET, 'bias': 'NEUTRAL', 'confidence': 0.3, 'adx': 0, 'bb_width': 0, 'atr_pct': 0, 'weights': weights, 'rr_ratio': weights['rr_ratio'], 'size_mult': weights['size_mult'], 'min_votes': weights['min_votes']}
