import json, sys, time 
from pathlib import Path 
 
ROOT = Path(__file__).parent.parent 
sys.path.insert(0, str(ROOT)) 
 
from core.ai_scalper import MultiAssetScalper 
from core.scalping_patch import apply_scalping_patch 
from logger import system_log 
 
RESULTS_FILE = ROOT / 'data' / 'paper_run_results.json' 
STATE_FILE = ROOT / 'data' / 'paper_runner_state.json' 
 
def load_config(): 
    cfg_file = ROOT / 'config' / 'config.json' 
    if cfg_file.exists(): 
        try: 
            with open(cfg_file, encoding='utf-8') as f: 
                return json.load(f) 
        except Exception as exc: 
            system_log.error(f'Config load failed: {exc}') 
            return {} 
    return {}
def persist_state(scalper, results, status='running', error_message=''): 
    RESULTS_FILE.parent.mkdir(exist_ok=True) 
    payload = { 
        'status': status, 
        'error_message': error_message, 
        'timestamp': time.time(), 
        'results_count': len(results), 
        'latest_result': results[-1] if results else None, 
        'trader_state': scalper.trader.get_state() if scalper else {}, 
    } 
    STATE_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8') 
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str), encoding='utf-8') 
 
def main(): 
    cfg = load_config() 
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 100 
    mode = sys.argv[2] if len(sys.argv) > 2 else 'paper' 
    results = [] 
    scalper = None 
    exit_code = 0
 
    system_log.info(f'Starting {mode} run for {cycles} cycles') 
 
    try: 
        scalper = MultiAssetScalper( 
            initial_balance=cfg.get('initial_balance', 200), 
            leverage=cfg.get('leverage', 10), 
            risk_per_trade_pct=cfg.get('risk_per_trade_pct', 0.02), 
            max_daily_loss_pct=cfg.get('max_daily_loss_pct', 0.10), 
            interval_seconds=cfg.get('interval_seconds', 30), 
            daily_profit_target=cfg.get('daily_profit_target'), 
            active_symbols=cfg.get('active_symbols'), 
            mode=mode, 
            config=cfg, 
        ) 
        apply_scalping_patch(scalper, cfg) 
        if hasattr(scalper, 'news_filter'): 
            scalper.news_filter.start_polling() 
        persist_state(scalper, results, status='started')
 
        for i in range(cycles): 
            try: 
                system_log.info(f'Cycle {i + 1}/{cycles}') 
                res = scalper.execute_cycle() 
                results.append(res) 
                persist_state(scalper, results) 
            except KeyboardInterrupt: 
                system_log.info('Interrupted by user') 
                persist_state(scalper, results, status='interrupted') 
                break 
            except Exception as exc: 
                exit_code = 1
                system_log.error(f'Cycle {i + 1} failed: {exc}') 
                persist_state(scalper, results, status='error', error_message=str(exc)) 
                time.sleep(max(1, int(getattr(scalper, 'interval', 1)))) 
                continue 
            time.sleep(scalper.interval) 
    except KeyboardInterrupt: 
        system_log.info('Interrupted during startup') 
        exit_code = 130
    except Exception as exc: 
        exit_code = 1
        system_log.error(f'Paper runner failed: {exc}') 
        if scalper: 
            persist_state(scalper, results, status='error', error_message=str(exc)) 
    finally: 
        if scalper and hasattr(scalper, 'news_filter'): 
            scalper.news_filter.stop_polling() 
        if scalper: 
            final_state = scalper.trader.get_state() 
            persist_state(scalper, results, status='stopped' if exit_code == 0 else 'error') 
            system_log.info(f"Final equity=${final_state['equity']:.2f} trades={final_state['trade_count']}") 
            print(json.dumps(final_state, indent=2, default=str)) 
 
    return exit_code 
 
if __name__ == '__main__': 
    raise SystemExit(main())
