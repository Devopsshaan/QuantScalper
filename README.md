# Quant Scalper v7 — Reversed Pipeline Architecture

## HOW TO RUN

### Step 1: Install dependencies
```bash
cd quant_scalper_v7
pip install ccxt pandas ta numpy scikit-learn fastapi uvicorn yfinance
```

### Step 2: Run with Dashboard (RECOMMENDED)
```bash
python -m api.server
```
Then open **http://localhost:8000** in your browser.
Click **START** to begin paper trading.

### Step 3: Run headless (no browser)
```bash
python -m runner.paper_runner 200 paper
```
This runs 200 cycles (about 100 minutes at 30s intervals).
Results saved to `data/paper_run_results.json`.

### Step 4: Monitor logs
```bash
tail -f logs/trading.log
```
Look for:
- `Gate OPEN` / `Gate CLOSED` — is the market exploitable?
- `ENTRY` — trade opened with quality score, RR, EV
- `GRAVEYARD blocked` — past loss pattern detected
- `NEGATIVE EV` — trade blocked for bad math
- `CLOSE` — trade closed with PnL

## WHAT TO EXPECT

**First 1-2 hours:** Bot will mostly say "Gate CLOSED" — this is GOOD.
It means the market is random noise and there's no edge to trade.

**When gate opens:** You'll see trades with quality scores, R:R ratios,
and expected values logged. Wins should be ~2x the size of losses.

**Daily target:** $5-15 on good days. Some days will be $0 (gate closed).
That's the bot protecting you from noise.

**Check the graveyard:** After a few losing trades, the bot builds a
"danger map" of conditions that caused losses. It will refuse to
repeat those conditions. This gets smarter over time.

## CONFIGURATION

Edit `config/config.json`:

| Setting | Default | Purpose |
|---------|---------|---------|
| initial_balance | 200 | Starting capital |
| leverage | 10 | Futures leverage |
| signal_quality_threshold | 0.65 | Min score to trade (higher = fewer but better trades) |
| max_open_positions | 2 | Max simultaneous trades |
| daily_profit_target | 15 | Stop after this daily profit |
| active_symbols | BTC,ETH,SOL | Which coins to trade |

## THE REVERSED PIPELINE

```
Traditional bot:  Signal → Filter → Trade (hope for the best)
This bot:         Gate → Signal → Graveyard → EV Check → Trade

Step 1: ExploitGate — Is the market non-random RIGHT NOW?
        Uses Hurst exponent, Shannon entropy, autocorrelation,
        variance ratio. If market is random walk → NO TRADES.

Step 2: Signal Generation — Only runs if gate is OPEN.
        Uses only strategy types recommended by the gate.

Step 3: TradeGraveyard — Do current conditions match past losses?
        If yes → BLOCKED. Learns from every loss.

Step 4: AsymmetricRisk — Is expected value positive?
        EV = P(win)×TP - P(loss)×SL. If negative → BLOCKED.

Step 5: Kelly sizing → Execute only proven +EV trades.
```
