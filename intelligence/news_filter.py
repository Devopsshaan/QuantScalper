"""
news_filter.py — Crypto news sentiment filter.

Polls free CryptoCompare news API for high-impact events.
When dangerous keywords are detected, trading is paused to avoid whipsaws.
"""

import time
import threading
from datetime import datetime

try:
    import urllib.request
    import json

    def _fetch_json(url):
        req = urllib.request.Request(url, headers={'User-Agent': 'CryptoBot/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
except ImportError:
    def _fetch_json(url):
        return {}


# High-impact keywords that cause market whipsaws
# Must be very specific to avoid false positives (e.g. "breach" in "price breached support")
DANGER_KEYWORDS = [
    'hacked', 'exploit',
    'flash crash',
    'fomc', 'rate hike', 'rate cut',
    'delisted', 'insolvent', 'bankruptcy',
]

# Keywords that affect ALL crypto (global pause)
GLOBAL_KEYWORDS = ['fomc', 'flash crash']

# Map news keywords to specific assets
ASSET_KEYWORDS = {
    'BTC/USDT': ['bitcoin', 'btc'],
    'ETH/USDT': ['ethereum', 'eth', 'vitalik'],
    'SOL/USDT': ['solana', 'sol'],
    'BNB/USDT': ['binance', 'bnb'],
    'DOGE/USDT': ['doge', 'dogecoin', 'elon'],
    'XRP/USDT': ['xrp', 'ripple', 'sec vs ripple'],
}

CRYPTO_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest"


class NewsFilter:
    def __init__(self, pause_minutes: int = 3, poll_interval: int = 300):
        self.pause_minutes = pause_minutes
        self.poll_interval = poll_interval  # seconds between polls
        self.is_paused = False
        self.pause_until = 0
        self.last_alert = ""
        self.last_alert_time = ""
        self.recent_headlines: list[dict] = []
        self._running = False
        self._thread = None
        # v5: Per-asset pause tracking
        self.paused_assets: dict[str, float] = {}  # symbol -> pause_until timestamp
        self.asset_alerts: dict[str, str] = {}  # symbol -> alert message

    def start_polling(self):
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print("[NewsFilter] Started polling for market-moving news.")

    def stop_polling(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            try:
                self._check_news()
            except Exception as e:
                print(f"[NewsFilter] Poll error: {e}")
            time.sleep(self.poll_interval)

    def _check_news(self):
        """Fetch latest crypto news and scan for danger keywords."""
        try:
            data = _fetch_json(CRYPTO_NEWS_URL)
            articles = data.get('Data', [])

            # Only check articles from last 30 minutes
            cutoff = time.time() - 1800
            recent = []

            # Must contain BOTH a danger keyword AND a crypto keyword
            crypto_terms = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol',
                           'binance', 'bnb', 'doge', 'dogecoin', 'xrp', 'ripple', 'defi',
                           'stablecoin', 'usdt', 'exchange']

            for article in articles[:20]:
                pub_time = article.get('published_on', 0)
                title = article.get('title', '')
                body = article.get('body', '')[:200]
                source = article.get('source', '')

                recent.append({
                    'title': title[:100],
                    'source': source,
                    'time': datetime.fromtimestamp(pub_time).strftime('%H:%M') if pub_time else '',
                    'danger': False,
                })

                if pub_time < cutoff:
                    continue

                # Check for danger keywords + must be crypto related
                text = (title + ' ' + body).lower()
                is_crypto = any(ct in text for ct in crypto_terms)
                if not is_crypto:
                    continue

                for keyword in DANGER_KEYWORDS:
                    if keyword in text:
                        self._trigger_pause(title, keyword, body)
                        recent[-1]['danger'] = True
                        break

            self.recent_headlines = recent[:10]

        except Exception as e:
            print(f"[NewsFilter] Error fetching news: {e}")

    def _trigger_pause(self, headline: str, keyword: str, text: str = ""):
        """Pause trading for affected assets (or all if global keyword)."""
        # Check if this is a global event (affects all crypto)
        is_global = any(gk in keyword.lower() for gk in GLOBAL_KEYWORDS)

        if is_global:
            # Global pause
            self.is_paused = True
            self.pause_until = time.time() + (self.pause_minutes * 60)
            self.last_alert = f"⚠️ {headline[:80]}... [{keyword}]"
            self.last_alert_time = datetime.now().strftime('%H:%M:%S')
            print(f"[NewsFilter] ⚠️ GLOBAL PAUSE — detected '{keyword}' in: {headline[:60]}...")
        else:
            # Asset-specific pause: find which assets are mentioned
            affected = []
            combined = (headline + ' ' + text).lower()
            for sym, keywords in ASSET_KEYWORDS.items():
                if any(kw in combined for kw in keywords):
                    affected.append(sym)
                    self.paused_assets[sym] = time.time() + (self.pause_minutes * 60)
                    self.asset_alerts[sym] = f"⚠️ {headline[:60]}... [{keyword}]"

            if affected:
                # Don't overwrite global alert message if a global pause is active
                if not self.is_paused:
                    self.last_alert = f"⚠️ {headline[:80]}... [{keyword}] → {', '.join(a.replace('/USDT','') for a in affected)}"
                    self.last_alert_time = datetime.now().strftime('%H:%M:%S')
                print(f"[NewsFilter] ⚠️ ASSET PAUSE {affected} — detected '{keyword}' in: {headline[:60]}...")
            else:
                # Can't determine asset → just log, don't pause everything
                print(f"[NewsFilter] ℹ️ Unrelated danger keyword '{keyword}' in: {headline[:60]}... (ignored)")

    def should_trade(self, symbol: str = None) -> bool:
        """
        Check if trading is allowed for a specific asset.

        Args:
            symbol: The asset symbol (e.g., 'XRP/USDT'). If None, checks global pause only.

        Returns:
            True if trading is allowed for this symbol.
        """
        # Check global pause first
        if self.is_paused:
            if time.time() > self.pause_until:
                self.is_paused = False
                self.last_alert = ""
                print("[NewsFilter] Global pause expired. Trading resumed.")
            else:
                return False

        # Check per-asset pause
        if symbol and symbol in self.paused_assets:
            if time.time() > self.paused_assets[symbol]:
                del self.paused_assets[symbol]
                if symbol in self.asset_alerts:
                    del self.asset_alerts[symbol]
                print(f"[NewsFilter] {symbol} pause expired. Trading resumed.")
                return True
            return False

        return True

    def get_status(self) -> dict:
        """Return current news filter status for dashboard."""
        remaining = max(0, int(self.pause_until - time.time())) if self.is_paused else 0

        # Clean up expired per-asset pauses
        now = time.time()
        active_asset_pauses = {}
        for sym, until in self.paused_assets.items():
            if until > now:
                active_asset_pauses[sym] = {
                    'remaining_s': int(until - now),
                    'alert': self.asset_alerts.get(sym, ''),
                }

        return {
            'active': self._running,
            'paused': self.is_paused or bool(active_asset_pauses),
            'global_paused': self.is_paused,
            'pause_remaining_s': remaining,
            'last_alert': self.last_alert,
            'last_alert_time': self.last_alert_time,
            'recent_headlines': self.recent_headlines,
            'paused_assets': active_asset_pauses,
        }


if __name__ == '__main__':
    nf = NewsFilter(pause_minutes=5, poll_interval=30)
    nf._check_news()
    status = nf.get_status()
    print(f"\nPaused: {status['paused']}")
    print(f"Alert: {status['last_alert']}")
    print(f"\nRecent headlines:")
    for h in status['recent_headlines']:
        danger = "🔴" if h['danger'] else "🟢"
        print(f"  {danger} [{h['time']}] {h['title']}")
