"""
crypto_feed.py — Stabilized market data feed.

TASK 8 UPGRADES:
  - Enhanced retry with exponential backoff + jitter
  - Data validation (reject stale/corrupt candles)
  - Connection health monitoring
  - Graceful degradation on API failures
  - Rate-limit aware caching
  - Exchange reconnection on repeated failures
"""

import time
import random
import ccxt
import pandas as pd
from ta.trend import EMAIndicator

from logger import data_log, error_log


class CryptoFeed:
    """Stabilized market data feed with retry, validation, and reconnection."""

    SUPPORTED_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'XRP/USDT', 'ADA/USDT', 'LINK/USDT', 'AVAX/USDT', 'MATIC/USDT']
    CACHE_TTL = 30
    MAX_CONSECUTIVE_FAILURES = 10

    def __init__(self, symbol='BTC/USDT', timeframe='1m', limit=100, retries=3):
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.retries = retries
        self._cache: dict[str, tuple[float, object]] = {}
        self._consecutive_failures = 0
        self._total_requests = 0
        self._total_failures = 0
        self._last_success_time = time.time()
        self._exchange = None
        self._init_exchange()

    def _init_exchange(self):
        """Initialize or reinitialize exchange connection."""
        try:
            self._exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
                'timeout': 15000,
            })
            data_log.info(f"Exchange initialized for {self.symbol}")
        except Exception as e:
            error_log.error(f"Exchange init failed: {e}")
            raise RuntimeError(f"Failed to initialise exchange: {e}") from e

    @property
    def exchange(self):
        # Auto-reconnect after too many failures
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            data_log.warning("Too many failures — reconnecting exchange...")
            self._init_exchange()
            self._consecutive_failures = 0
        return self._exchange

    def _get_cached(self, key: str):
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self.CACHE_TTL:
                return data
        return None

    def _set_cached(self, key: str, data):
        self._cache[key] = (time.time(), data)

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data."""
        if df is None or df.empty:
            return df

        original_len = len(df)

        # Remove rows with zero/negative prices
        for col in ['open', 'high', 'low', 'close']:
            df = df[df[col] > 0]

        # Ensure high >= low
        df = df[df['high'] >= df['low']]

        # Ensure high >= open, close and low <= open, close
        df = df[
            (df['high'] >= df['open']) & (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & (df['low'] <= df['close'])
        ]

        # Remove duplicate timestamps
        df = df.drop_duplicates(subset='timestamp', keep='last')

        # Sort by time
        df = df.sort_values('timestamp').reset_index(drop=True)

        removed = original_len - len(df)
        if removed > 0:
            data_log.warning(f"Validation removed {removed}/{original_len} invalid candles")

        return df

    def fetch_candles(self, symbol=None, timeframe=None, limit=None) -> pd.DataFrame:
        """Fetch OHLCV with enhanced retry + jitter + validation."""
        symbol = symbol or self.symbol
        timeframe = timeframe or self.timeframe
        limit = limit or self.limit

        cache_key = f"candles_{symbol}_{timeframe}_{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._total_requests += 1
        last_error = None

        for attempt in range(1, self.retries + 1):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv:
                    raise ValueError("Empty OHLCV response")

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(inplace=True)

                # TASK 8: Validate data quality
                df = self._validate_ohlcv(df)
                if df.empty:
                    raise ValueError("All candles failed validation")

                self._set_cached(cache_key, df)
                self._consecutive_failures = 0
                self._last_success_time = time.time()
                return df

            except Exception as e:
                last_error = e
                self._consecutive_failures += 1
                self._total_failures += 1
                # Exponential backoff with jitter
                base_wait = 2 ** attempt
                jitter = random.uniform(0, base_wait * 0.3)
                wait = base_wait + jitter
                data_log.warning(
                    f"Fetch attempt {attempt}/{self.retries} failed "
                    f"({symbol} {timeframe}): {e}. Retry in {wait:.1f}s"
                )
                time.sleep(wait)

        error_log.error(f"All {self.retries} attempts failed for {symbol}. Last: {last_error}")
        return None

    def fetch_order_book(self, symbol=None, depth: int = 10) -> dict:
        symbol = symbol or self.symbol
        cache_key = f"ob_{symbol}_{depth}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            ob = self.exchange.fetch_order_book(symbol, limit=depth)
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            bid_vol = sum(b[1] for b in bids[:depth]) if bids else 0
            ask_vol = sum(a[1] for a in asks[:depth]) if asks else 0
            total = bid_vol + ask_vol
            imbalance = (bid_vol - ask_vol) / total if total > 0 else 0
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread_bps = ((best_ask - best_bid) / mid * 10000) if mid > 0 else 0
            pressure = 'BUY' if imbalance > 0.10 else ('SELL' if imbalance < -0.10 else 'NEUTRAL')

            result = {
                'bid_volume': round(bid_vol, 4), 'ask_volume': round(ask_vol, 4),
                'imbalance': round(imbalance, 4), 'pressure': pressure,
                'best_bid': best_bid, 'best_ask': best_ask,
                'spread_bps': round(spread_bps, 2),
            }
            self._set_cached(cache_key, result)
            return result
        except Exception as e:
            data_log.warning(f"Order book error {symbol}: {e}")
            return {
                'bid_volume': 0, 'ask_volume': 0, 'imbalance': 0,
                'pressure': 'UNKNOWN', 'best_bid': 0, 'best_ask': 0, 'spread_bps': 0,
            }

    def fetch_htf_trend(self, symbol=None, timeframes=('5m', '15m')) -> dict:
        symbol = symbol or self.symbol
        cache_key = f"htf_{symbol}_{'_'.join(timeframes)}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        trends = {}
        directions = []
        for tf in timeframes:
            df = self.fetch_candles(symbol=symbol, timeframe=tf, limit=50)
            if df is None or len(df) < 25:
                trends[tf] = {'trend': 'UNKNOWN', 'ema9': 0, 'ema21': 0}
                continue
            ema9 = EMAIndicator(close=df['close'], window=9).ema_indicator()
            ema21 = EMAIndicator(close=df['close'], window=21).ema_indicator()
            e9, e21 = float(ema9.iloc[-1]), float(ema21.iloc[-1])
            price = float(df.iloc[-1]['close'])
            trend = 'UP' if e9 > e21 and price > e9 else ('DOWN' if e9 < e21 and price < e9 else 'NEUTRAL')
            trends[tf] = {'trend': trend, 'ema9': round(e9, 2), 'ema21': round(e21, 2), 'price': round(price, 2)}
            directions.append(trend)

        non_neutral = [d for d in directions if d != 'NEUTRAL']
        if non_neutral and all(d == non_neutral[0] for d in non_neutral):
            aligned, direction = True, non_neutral[0]
        elif not non_neutral:
            aligned, direction = True, 'NEUTRAL'
        else:
            aligned, direction = False, 'MIXED'

        result = {'timeframes': trends, 'aligned': aligned, 'direction': direction}
        self._set_cached(cache_key, result)
        return result

    def get_health(self) -> dict:
        """Return feed health metrics for monitoring."""
        uptime = time.time() - self._last_success_time
        failure_rate = self._total_failures / max(self._total_requests, 1) * 100
        return {
            'total_requests': self._total_requests,
            'total_failures': self._total_failures,
            'failure_rate_pct': round(failure_rate, 1),
            'consecutive_failures': self._consecutive_failures,
            'seconds_since_success': round(uptime, 0),
            'status': 'HEALTHY' if self._consecutive_failures < 3 else 'DEGRADED',
        }
