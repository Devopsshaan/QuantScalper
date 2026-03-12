"""
server.py — FastAPI backend for Quant Scalper v6.

TASK 10: Serves dashboard with mode selection (backtest/paper/live).
TASK 11: API exposes signal quality scores, portfolio status, ML status.
"""

import os, sys, asyncio, json, queue, threading, time, tempfile
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path

if not os.getenv('GIT_DIR'):
    _t = tempfile.mkdtemp(prefix="nogit_")
    os.environ['GIT_DIR'] = _t
    os.environ['GIT_WORK_TREE'] = _t

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ccxt
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from core.ai_scalper import MultiAssetScalper
from core.scalping_patch import apply_scalping_patch
from logger import system_log

bot: MultiAssetScalper = None
bot_thread: threading.Thread = None
event_queue: queue.Queue = queue.Queue(maxsize=100)
lock = threading.Lock()

CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"
DASHBOARD_FILE = PROJECT_ROOT / "dashboard" / "dashboard.html"

DEFAULT_CONFIG = {
    "mode": "paper", "initial_balance": 200, "leverage": 10,
    "risk_per_trade_pct": 0.02, "max_daily_loss_pct": 0.08,
    "circuit_breaker_losses": 3, "circuit_breaker_cooldown_cycles": 15,
    "interval_seconds": 30, "daily_profit_target": 30.0,
    "min_trade_grade": "C",  "allow_quiet_regime_c_trades": True,   # FIX: was "B"/False — grade C is the realistic scalp floor
    "signal_quality_threshold": 0.65,   # FIX: was 0.70 — matches SignalQualityFilter internal default
    "max_open_positions": 2, "max_asset_exposure_pct": 0.95,
    "max_total_leverage": 20, "max_drawdown_pct": 15,
    "max_daily_loss": 10, "max_consecutive_losses": 5,
    "max_trades_per_hour": 6, "scalp_mode": True,
    "active_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT", "MATIC/USDT"],
}

ALLOWED_MODES = {'backtest', 'paper', 'live'}

def _safe_json(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isinf(obj) or np.isnan(obj):
            return "Infinity" if obj > 0 else ("-Infinity" if obj < 0 else "NaN")
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
NUMERIC_FIELDS = (
    'initial_balance', 'leverage', 'risk_per_trade_pct', 'max_daily_loss_pct',
    'interval_seconds', 'signal_quality_threshold', 'max_open_positions',
    'max_asset_exposure_pct', 'max_total_leverage', 'max_drawdown_pct',
    'max_daily_loss', 'max_consecutive_losses', 'max_trades_per_hour',
)

def _json_error(message, status_code=500):
    system_log.error(message)
    return JSONResponse({'status': 'error', 'message': str(message)}, status_code=status_code)

def _normalize_symbols(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return DEFAULT_CONFIG['active_symbols'][:]

def _validate_config(cfg):
    errors = []
    if cfg.get('mode') not in ALLOWED_MODES:
        errors.append('mode must be one of backtest, paper, live')
    for field in NUMERIC_FIELDS:
        try:
            value = float(cfg[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f'{field} must be numeric')
            continue
        if value <= 0:
            errors.append(f'{field} must be greater than zero')
    cfg['active_symbols'] = _normalize_symbols(cfg.get('active_symbols'))
    if not cfg['active_symbols']:
        errors.append('active_symbols cannot be empty')
    cfg['max_open_positions'] = int(cfg['max_open_positions'])
    cfg['leverage'] = int(cfg['leverage'])
    cfg['interval_seconds'] = max(1, int(cfg['interval_seconds']))
    return errors

def _stop_running_bot():
    global bot, bot_thread
    if bot:
        bot.is_running = False
        if hasattr(bot, 'news_filter'):
            bot.news_filter.stop_polling()
    if bot_thread and bot_thread.is_alive():
        bot_thread.join(timeout=5)

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding='utf-8') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception as exc:
            system_log.error(f'Config load error: {exc}')
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

def _bot_loop(bot_inst):
    bot_inst.is_running = True
    bot_inst.news_filter.start_polling()
    interval = bot_inst.interval
    try:
        first = list(bot_inst.engines.values())[0]
        train_df = first.feed.fetch_candles(limit=1000)
        if train_df is not None and len(train_df) > 100:
            bot_inst.ml_scorer.train(train_df)
    except: pass
    try: bot_inst.correlation.update()
    except: pass

    while bot_inst.is_running:
        try:
            result = bot_inst.execute_cycle()
            with lock:
                pass  # latest stored in bot itself
            try: event_queue.put_nowait(result)
            except queue.Full:
                try: event_queue.get_nowait()
                except: pass
                event_queue.put_nowait(result)
        except Exception as e:
            system_log.error(f"BotLoop error: {e}")
            import traceback; traceback.print_exc()
        time.sleep(interval)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
    yield
    if bot and bot.is_running:
        bot.is_running = False
        bot.news_filter.stop_polling()

app = FastAPI(title="Quant Scalper v6", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    if DASHBOARD_FILE.exists():
        return DASHBOARD_FILE.read_text(encoding='utf-8')
    return "<h1>Dashboard not found</h1>"

@app.get("/api/stream")
async def sse_stream(request: Request):
    async def gen():
        while True:
            if await request.is_disconnected(): break
            try:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: event_queue.get(timeout=2.0))
                safe_data = _safe_json(data)
                yield f"data: {json.dumps(safe_data, default=str)}\n\n"
            except: yield ": heartbeat\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/state")
async def get_state():
    try:
        if bot is None:
            return JSONResponse({'status': 'ok', 'running': False})
        with lock:
            state = bot.trader.get_state()
            state['status'] = 'ok'
            state['running'] = bot.is_running
            state['cycle'] = bot.cycle_count
            state['prices'] = bot.all_prices
            state['indicators'] = bot.all_indicators
            state['rankings'] = bot.rankings
            state['news'] = bot.news_filter.get_status()
            state['active_positions'] = bot.active_positions
            state['active_symbol'] = bot._get_active_symbol()
            state['regimes'] = bot.all_regimes
            state['confidence'] = bot.all_confidence
            state['order_flow'] = bot.all_order_flow
            state['ml'] = bot.ml_status
            state['correlation'] = bot.correlation_status
            state['patterns'] = bot.pattern_status
            state['quality_scores'] = bot.last_quality_scores
            state['funding'] = getattr(bot, 'last_funding', {})
            state['health'] = getattr(bot, 'health_status', {})
            state['portfolio'] = bot.portfolio_manager.get_status() if hasattr(bot, 'portfolio_manager') else {}
            state['strategy_votes'] = getattr(bot, 'all_strategy_votes', {})
            state['edge_scores'] = getattr(bot, 'all_edge_scores', {})
            if hasattr(bot, 'quant_edge'):
                state['bayesian'] = bot.quant_edge.bayesian_win_rate()
        return JSONResponse(_safe_json(state))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return _json_error(f'/api/state failed: {exc}')

@app.get("/api/trades")
async def get_trades():
    try:
        return JSONResponse(_safe_json(bot.trader.trade_history) if bot else [])
    except Exception as exc:
        return _json_error(f'/api/trades failed: {exc}')

@app.get("/api/ml")
async def get_ml():
    return JSONResponse(bot.ml_status if bot else {'trained': False})

@app.get("/api/portfolio")
async def get_portfolio():
    try:
        return JSONResponse(bot.portfolio_manager.get_status() if bot else {})
    except Exception as exc:
        return _json_error(f'/api/portfolio failed: {exc}')

@app.get("/api/candles/{symbol}")
async def get_candles(symbol: str, timeframe: str = '1m', limit: int = 100):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        sym = symbol.replace('-', '/')
        ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=limit)
        return JSONResponse([{
            'time': c[0] // 1000, 'open': c[1], 'high': c[2],
            'low': c[3], 'close': c[4], 'volume': c[5]
        } for c in ohlcv])
    except Exception as exc:
        return _json_error(f'/api/candles failed: {exc}')

@app.post("/api/start")
async def start_bot(request: Request):
    global bot, bot_thread
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    cfg = load_config()
    cfg.update({k: v for k, v in body.items() if k in DEFAULT_CONFIG})
    errors = _validate_config(cfg)
    if errors:
        return JSONResponse({'status': 'error', 'message': '; '.join(errors)}, status_code=400)
    try:
        save_config(cfg)
        _stop_running_bot()
        system_log.info(f'Starting bot mode={cfg.get("mode")} symbols={cfg.get("active_symbols")}')
        bot = MultiAssetScalper(
            initial_balance=cfg['initial_balance'],
            leverage=cfg['leverage'],
            risk_per_trade_pct=cfg['risk_per_trade_pct'],
            max_daily_loss_pct=cfg['max_daily_loss_pct'],
            interval_seconds=cfg['interval_seconds'],
            daily_profit_target=cfg.get('daily_profit_target'),
            active_symbols=cfg.get('active_symbols'),
            mode=cfg.get('mode', 'paper'),
            config=cfg,
        )
        apply_scalping_patch(bot, cfg)
        bot_thread = threading.Thread(target=_bot_loop, args=(bot,), daemon=True)
        bot_thread.start()
        return JSONResponse({'status': 'started', 'mode': cfg.get('mode'), 'config': cfg})
    except Exception as exc:
        _stop_running_bot()
        return _json_error(f'Bot startup failed: {exc}')

@app.post("/api/stop")
async def stop_bot():
    try:
        _stop_running_bot()
        return JSONResponse({'status': 'stopped'})
    except Exception as exc:
        return _json_error(f'Bot stop failed: {exc}')


@app.get('/api/pnl')
async def get_pnl():
    try:
        if bot is None:
            return JSONResponse({'status': 'ok', 'pnl': 0.0, 'equity': 0.0, 'roi_pct': 0.0})
        state = bot.trader.get_state()
        return JSONResponse({'status': 'ok', 'pnl': state.get('daily_pnl', 0.0), 'equity': state.get('equity', 0.0), 'roi_pct': state.get('roi_pct', 0.0)})
    except Exception as exc:
        return _json_error(f'/api/pnl failed: {exc}')

@app.get('/api/health')
async def get_health():
    try:
        if bot is None:
            return JSONResponse({'status': 'ok', 'running': False, 'bot': 'idle'})
        return JSONResponse({'status': 'ok', 'running': bot.is_running, 'cycle': bot.cycle_count, 'symbols': bot.SYMBOLS, 'portfolio_halted': bot.portfolio_manager.get_status().get('is_halted', False), 'risk_guard': getattr(bot, 'health_status', {}).get('risk_guard', {}), 'thread_alive': bot_thread.is_alive() if bot_thread else False})
    except Exception as exc:
        return _json_error(f'/api/health failed: {exc}')

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', '8000'))
    system_log.info(f'Starting Quant Scalper v6 on http://localhost:{port}')
    uvicorn.run(app, host='0.0.0.0', port=port)
