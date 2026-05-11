#!/usr/bin/env bash
set -euo pipefail
python3 run.py \
  --truth-noise \
  --sensor-noise \
  --controller lqr \
  --duration 80.0 \
  --dt 0.03 \
  --animate \
  --animation-speed 3
