#!/usr/bin/env bash
set -e

cd /opt/binary-bot/meta-bot
source venv/bin/activate

export POLYMARKET_MODE=live
export POLYMARKET_MARKET_LIMIT=1
export POLYMARKET_DEBUG_RAW=false

exec python3 -m binary_bot.app