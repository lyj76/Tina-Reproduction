#!/usr/bin/env bash
cd /root/autodl-tmp/Tina || exit 1
exec /root/autodl-tmp/envs/tina/bin/python -u -m reproduction.core_tui "$@"
