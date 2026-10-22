#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Tina
source reproduction/env.sh
exec /root/autodl-tmp/envs/tina/bin/python -u -m reproduction.mechanism_queue >> logs/mechanism_queue.log 2>&1
