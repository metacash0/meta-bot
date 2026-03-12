#!/usr/bin/env bash
set -e

cd /opt/binary-bot/meta-bot
source venv/bin/activate

export POLYMARKET_MODE=live
export POLYMARKET_MARKET_LIMIT=1
export POLYMARKET_DEBUG_RAW=false
export APIFOOTBALL_API_KEY="2a0966f08995b9ea7235261cdf039e76"

echo "[startup] rebuilding market map"
python3 -m binary_bot.build_market_map

echo "[startup] launching bot app"
exec python3 -m binary_bot.app