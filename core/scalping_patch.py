from intelligence.market_regime import MarketRegime

DEFAULT_ALLOWED_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

def _allowed_grades(min_trade_grade):
    return ('A', 'B') if str(min_trade_grade).upper() == 'B' else ('A', 'B', 'C')

def _patch_engine(engine):
    if getattr(engine, '_scalping_grade_patch_applied', False):
        return
    original = getattr(engine, '_grade_signal', None)
    if original is None:
        engine._scalping_grade_patch_applied = True
        return

    def patched(*args, **kwargs):
        confidence = original(*args, **kwargs)
        min_trade_grade = getattr(engine, 'min_trade_grade', 'B')
        allow_quiet = getattr(engine, 'allow_quiet_regime_c_trades', False)
        grade = confidence.get('grade', 'D')
        regime_info = kwargs.get('regime_info')
        if regime_info is None and len(args) not in (0, 1, 2, 3):
            regime_info = args[3]
        regime_name = str((regime_info or {}).get('regime', ''))
        allowed = _allowed_grades(min_trade_grade)
        if grade not in allowed:
            confidence['passed'] = False
            confidence['reject_reason'] = 'grade_below_' + str(min_trade_grade)
        elif regime_name in ('QUIET', MarketRegime.QUIET) and grade == 'C' and not allow_quiet:
            confidence['passed'] = False
            confidence['reject_reason'] = 'quiet_regime_requires_B_or_higher'
        else:
            confidence.pop('reject_reason', None)
        return confidence

    engine._grade_signal = patched
    engine._scalping_grade_patch_applied = True

def _patch_trader(trader):
    if getattr(trader, '_scalping_open_patch_applied', False):
        return
    original = trader.open_position

    def patched(side, price, size_usd, tp_price=0.0, sl_price=0.0, grade='', regime='', symbol=''):
        from logger import system_log
        min_trade_grade = getattr(trader, 'min_trade_grade', 'B')
        allow_quiet = getattr(trader, 'allow_quiet_regime_c_trades', False)
        allowed_symbols = getattr(trader, 'allowed_symbols', DEFAULT_ALLOWED_SYMBOLS)
        allowed = _allowed_grades(min_trade_grade)
        if symbol and allowed_symbols and symbol not in allowed_symbols:
            system_log.warning(f'[Patch] Rejected {symbol} outside focus list')
            return False
        if grade and grade not in allowed:
            system_log.warning(f'[Patch] Rejected {symbol or "trade"} grade {grade} below {min_trade_grade}')
            return False
        if regime in ('QUIET', MarketRegime.QUIET) and grade == 'C' and not allow_quiet:
            system_log.warning(f'[Patch] Rejected quiet-regime {symbol or "trade"} grade C')
            return False
        return original(side, price, size_usd, tp_price=tp_price, sl_price=sl_price, grade=grade, regime=regime, symbol=symbol)

    trader.open_position = patched
    trader._scalping_open_patch_applied = True

def _trim_bot_symbols(bot, allowed_symbols):
    allowed_symbols = [sym for sym in allowed_symbols if sym in bot.engines]
    if not allowed_symbols:
        allowed_symbols = [sym for sym in DEFAULT_ALLOWED_SYMBOLS if sym in bot.engines]
    bot.engines = {sym: bot.engines[sym] for sym in allowed_symbols}
    bot.SYMBOLS = allowed_symbols
    if hasattr(bot, 'ranker'):
        bot.ranker.assets = allowed_symbols
    return allowed_symbols

def apply_scalping_patch(bot, cfg):
    min_trade_grade = cfg.get('min_trade_grade', 'B')
    allow_quiet = cfg.get('allow_quiet_regime_c_trades', False)
    cb_losses = cfg.get('circuit_breaker_losses', 3)
    cb_cooldown = cfg.get('circuit_breaker_cooldown_cycles', 6)
    time_stop_candles = cfg.get('time_stop_candles', 20)
    allowed_symbols = cfg.get('active_symbols', DEFAULT_ALLOWED_SYMBOLS)

    if isinstance(allowed_symbols, str):
        allowed_symbols = [s.strip() for s in allowed_symbols.split(',') if s.strip()]

    allowed_symbols = _trim_bot_symbols(bot, allowed_symbols)

    bot.trader.min_trade_grade = min_trade_grade
    bot.trader.allow_quiet_regime_c_trades = allow_quiet
    bot.trader.allowed_symbols = allowed_symbols
    bot.trader.circuit_breaker_losses = cb_losses
    bot.trader.circuit_breaker_cooldown = cb_cooldown
    bot.trader.time_stop_candles = time_stop_candles
    _patch_trader(bot.trader)

    for engine in bot.engines.values():
        engine.min_trade_grade = min_trade_grade
        engine.allow_quiet_regime_c_trades = allow_quiet
        if hasattr(engine, 'trader'):
            engine.trader.min_trade_grade = min_trade_grade
            engine.trader.allow_quiet_regime_c_trades = allow_quiet
            engine.trader.allowed_symbols = allowed_symbols
            engine.trader.circuit_breaker_losses = cb_losses
            engine.trader.circuit_breaker_cooldown = cb_cooldown
            engine.trader.time_stop_candles = time_stop_candles
        else:
            engine.min_trade_grade = min_trade_grade
        _patch_engine(engine)
