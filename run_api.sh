#!/bin/bash
cd /home/ubuntu/quant_scalper_v6
pip install -r requirements.txt
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
