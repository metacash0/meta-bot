from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    import websocket
except ImportError:
    websocket = None  # type: ignore[assignment]

from binary_bot.models import MarketSnapshot


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        # Debug logging must never break feed processing.
        return


def _iter_candidate_payloads(message: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    data = message.get("data")
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item

    payload = message.get("payload")
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item

    yield message


def _extract_price_size(level: Any) -> Tuple[Optional[float], float]:
    if isinstance(level, dict):
        price = _as_float(level.get("price"))
        if price is None:
            price = _as_float(level.get("p"))
        size = _as_float(level.get("size"))
        if size is None:
            size = _as_float(level.get("quantity"))
        if size is None:
            size = _as_float(level.get("amount"))
        if size is None:
            size = 0.0
        return price, float(size)

    if isinstance(level, (list, tuple)) and level:
        price = _as_float(level[0])
        size = _as_float(level[1]) if len(level) > 1 else 0.0
        if size is None:
            size = 0.0
        return price, float(size)

    return None, 0.0


def _extract_top_bid(payload: Dict[str, Any]) -> Tuple[Optional[float], float]:
    for candidate in _iter_candidate_payloads(payload):
        direct = _as_float(candidate.get("bid"))
        if direct is None:
            direct = _as_float(candidate.get("best_bid"))
        if direct is None:
            direct = _as_float(candidate.get("bestBid"))
        if direct is not None:
            size = _as_float(candidate.get("bid_size"))
            if size is None:
                size = _as_float(candidate.get("best_bid_size"))
            if size is None:
                size = _as_float(candidate.get("bidSize"))
            return direct, float(size if size is not None else 0.0)

        bids = candidate.get("bids")
        if isinstance(bids, list) and bids:
            for level in bids:
                price, size = _extract_price_size(level)
                if price is not None and 0.0 <= float(price) <= 1.0:
                    return float(price), float(size)

        book = candidate.get("book")
        if isinstance(book, dict):
            book_bids = book.get("bids")
            if isinstance(book_bids, list) and book_bids:
                for level in book_bids:
                    price, size = _extract_price_size(level)
                    if price is not None and 0.0 <= float(price) <= 1.0:
                        return float(price), float(size)

    return None, 0.0


def _extract_top_ask(payload: Dict[str, Any]) -> Tuple[Optional[float], float]:
    for candidate in _iter_candidate_payloads(payload):
        direct = _as_float(candidate.get("ask"))
        if direct is None:
            direct = _as_float(candidate.get("best_ask"))
        if direct is None:
            direct = _as_float(candidate.get("bestAsk"))
        if direct is not None:
            size = _as_float(candidate.get("ask_size"))
            if size is None:
                size = _as_float(candidate.get("best_ask_size"))
            if size is None:
                size = _as_float(candidate.get("askSize"))
            return direct, float(size if size is not None else 0.0)

        asks = candidate.get("asks")
        if isinstance(asks, list) and asks:
            for level in asks:
                price, size = _extract_price_size(level)
                if price is not None and 0.0 <= float(price) <= 1.0:
                    return float(price), float(size)

        book = candidate.get("book")
        if isinstance(book, dict):
            book_asks = book.get("asks")
            if isinstance(book_asks, list) and book_asks:
                for level in book_asks:
                    price, size = _extract_price_size(level)
                    if price is not None and 0.0 <= float(price) <= 1.0:
                        return float(price), float(size)

    return None, 0.0


def _extract_last(payload: Dict[str, Any]) -> Optional[float]:
    for candidate in _iter_candidate_payloads(payload):
        for key in ("last", "last_price", "lastPrice", "price", "trade_price", "fill_price"):
            value = _as_float(candidate.get(key))
            if value is not None:
                return value
    return None


def _extract_market_id(payload: Dict[str, Any]) -> Optional[str]:
    for candidate in _iter_candidate_payloads(payload):
        for key in ("asset_id", "assetId", "token_id", "tokenId", "market_id", "marketId"):
            value = candidate.get(key)
            if value is not None:
                return str(value)
    return None


def _extract_timestamp(payload: Dict[str, Any]) -> float:
    for candidate in _iter_candidate_payloads(payload):
        for key in ("ts", "timestamp", "time"):
            value = _as_float(candidate.get(key))
            if value is not None:
                if value > 1e12:
                    return float(value / 1000.0)
                return float(value)
    return float(time.time())


def _extract_event_type(payload: Dict[str, Any]) -> str:
    top_level = payload.get("event_type")
    if isinstance(top_level, str):
        return top_level

    for candidate in _iter_candidate_payloads(payload):
        value = candidate.get("event_type")
        if isinstance(value, str):
            return value

    return ""


def _is_best_bid_ask_event(payload: Dict[str, Any]) -> bool:
    return _extract_event_type(payload) == "best_bid_ask"


def _normalize_message_to_snapshot(payload: Dict[str, Any], default_market_id: str) -> Optional[MarketSnapshot]:
    market_id = _extract_market_id(payload) or default_market_id

    bid, bid_size = _extract_top_bid(payload)
    ask, ask_size = _extract_top_ask(payload)
    last = _extract_last(payload)

    if bid is None and ask is None and last is None:
        return None

    if bid is None and ask is not None:
        bid = float(max(0.01, ask - 0.01))
    if ask is None and bid is not None:
        ask = float(min(0.99, bid + 0.01))

    if bid is None or ask is None:
        return None

    if float(bid) < 0.0 or float(bid) > 1.0:
        return None
    if float(ask) < 0.0 or float(ask) > 1.0:
        return None
    if float(bid) >= float(ask):
        return None
    if (float(ask) - float(bid)) > 0.20:
        return None

    mid = (float(bid) + float(ask)) / 2.0
    if last is None or float(last) < 0.0 or float(last) > 1.0:
        last = mid

    return MarketSnapshot(
        ts=float(_extract_timestamp(payload)),
        market_id=str(market_id),
        bid=float(bid),
        ask=float(ask),
        last=float(last),
        bid_size=float(bid_size),
        ask_size=float(ask_size),
    )


def _extract_token_ids_from_market_row(row: Dict[str, Any]) -> List[str]:
    token_ids: List[str] = []

    clob_token_ids = row.get("clobTokenIds")
    if isinstance(clob_token_ids, str):
        try:
            decoded = json.loads(clob_token_ids)
            if isinstance(decoded, list):
                token_ids.extend(str(v) for v in decoded if v is not None)
        except (TypeError, ValueError):
            pass
    elif isinstance(clob_token_ids, list):
        token_ids.extend(str(v) for v in clob_token_ids if v is not None)

    tokens = row.get("tokens")
    if isinstance(tokens, list):
        for token in tokens:
            if isinstance(token, dict):
                for key in ("token_id", "tokenId", "asset_id", "assetId", "id"):
                    value = token.get(key)
                    if value is not None:
                        token_ids.append(str(value))
                        break
            elif token is not None:
                token_ids.append(str(token))

    for key in ("token_id", "tokenId", "asset_id", "assetId", "clob_token_id", "clobTokenId"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            token_ids.extend(str(v) for v in value if v is not None)
        else:
            token_ids.append(str(value))

    # Preserve order and drop duplicates.
    deduped: List[str] = []
    seen = set()
    for token_id in token_ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        deduped.append(token_id)
    return deduped


def discover_polymarket_asset_ids(
    gamma_url: str,
    market_limit: int,
    active_only: bool,
    closed: bool,
    timeout_sec: float,
) -> List[str]:
    if requests is None:
        raise RuntimeError("requests is required for POLYMARKET_MODE=live")

    params: Dict[str, Any] = {
        "limit": int(max(1, market_limit)),
    }
    if active_only:
        params["active"] = "true"
    params["closed"] = "true" if closed else "false"

    resp = requests.get(gamma_url, params=params, timeout=float(timeout_sec))
    resp.raise_for_status()

    payload = resp.json()
    markets: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        markets = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("markets"), list):
            markets = [item for item in payload["markets"] if isinstance(item, dict)]
        elif isinstance(payload.get("data"), list):
            markets = [item for item in payload["data"] if isinstance(item, dict)]

    if not markets:
        raise RuntimeError("Gamma market discovery returned no market rows")

    asset_ids: List[str] = []
    valid_markets_collected = 0
    for market in markets:
        if market.get("enableOrderBook") is False:
            continue

        ids = _extract_token_ids_from_market_row(market)
        if ids:
            asset_ids.extend(ids)
            valid_markets_collected += 1
            if valid_markets_collected >= max(1, market_limit):
                break

    # Preserve order and drop duplicates.
    deduped: List[str] = []
    seen = set()
    for asset_id in asset_ids:
        if asset_id in seen:
            continue
        seen.add(asset_id)
        deduped.append(asset_id)

    if not deduped:
        raise RuntimeError("Gamma market discovery found no token/asset IDs")

    return deduped


class MockDataFeed:
    """
    Deterministic mock data feed for phase-1 testing.

    Produces a smooth oscillating market with small spread changes and
    simple depth changes so the rest of the paper bot can be exercised
    end-to-end before plugging in a real websocket feed.
    """

    def __init__(
        self,
        market_id: str = "mock_market_1",
        start_mid: float = 0.50,
        base_spread: float = 0.02,
        tick_interval_sec: float = 1.0,
        total_ticks: int = 300,
    ):
        self.market_id = market_id
        self.start_mid = float(start_mid)
        self.base_spread = float(base_spread)
        self.tick_interval_sec = float(tick_interval_sec)
        self.total_ticks = int(total_ticks)
        self.reconnect_count = 0
        self.last_message_wallclock: Optional[float] = None
        self.last_snapshot_wallclock: Optional[float] = None
        self.last_snapshot_ts: Optional[float] = None
        self.message_count = 0
        self.snapshot_count = 0
        self.started_wallclock = float(time.time())

    def snapshots(self) -> Iterator[MarketSnapshot]:
        for t in range(self.total_ticks):
            ts = time.time()

            wave = 0.10 * math.sin(t / 8.0)
            drift = 0.0005 * t / max(self.total_ticks, 1)
            mid = min(0.95, max(0.05, self.start_mid + wave + drift))

            spread_bump = 0.01 * (0.5 + 0.5 * math.sin(t / 17.0))
            spread = max(0.01, self.base_spread + spread_bump)

            bid = max(0.01, mid - spread / 2.0)
            ask = min(0.99, mid + spread / 2.0)

            bid_size = 100.0 + 20.0 * (1.0 + math.sin(t / 7.0))
            ask_size = 100.0 + 20.0 * (1.0 + math.cos(t / 9.0))
            last = mid

            yield MarketSnapshot(
                ts=float(ts),
                market_id=self.market_id,
                bid=float(bid),
                ask=float(ask),
                last=float(last),
                bid_size=float(bid_size),
                ask_size=float(ask_size),
            )

            time.sleep(self.tick_interval_sec)

    def monitoring_state(self) -> Dict[str, Any]:
        return {
            "reconnect_count": 0,
            "last_message_wallclock": None,
            "last_snapshot_wallclock": None,
            "last_snapshot_ts": None,
            "message_count": 0,
            "snapshot_count": 0,
            "started_wallclock": self.started_wallclock,
        }


class PolymarketLiveDataFeed:
    def __init__(self):
        if requests is None or websocket is None:
            raise RuntimeError(
                "POLYMARKET_MODE=live requires dependencies: requests and websocket-client"
            )

        self.ws_url = os.getenv(
            "POLYMARKET_WS_URL",
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        )
        self.gamma_url = os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com/markets")
        self.market_limit = _env_int("POLYMARKET_MARKET_LIMIT", 5)
        self.active_only = _env_bool("POLYMARKET_ACTIVE_ONLY", True)
        self.closed = _env_bool("POLYMARKET_CLOSED", False)
        self.feed_timeout_sec = _env_float("POLYMARKET_FEED_TIMEOUT_SEC", 30.0)
        self.ping_interval_sec = _env_float("POLYMARKET_PING_INTERVAL_SEC", 15.0)
        self.heartbeat_sec = _env_float("POLYMARKET_HEARTBEAT_SEC", 10.0)
        self.stale_feed_sec = _env_float("POLYMARKET_STALE_FEED_SEC", 15.0)
        self.debug_raw = _env_bool("POLYMARKET_DEBUG_RAW", False)
        self.debug_raw_limit = _env_int("POLYMARKET_DEBUG_RAW_LIMIT", 20)
        self.debug_log_path = os.getenv(
            "POLYMARKET_DEBUG_LOG_PATH",
            "binary_bot/logs/polymarket_debug.jsonl",
        )
        self.reconnect_count = 0
        self.last_snapshot_ts: Optional[float] = None
        self.last_message_wallclock: Optional[float] = None
        self.last_snapshot_wallclock: Optional[float] = None
        self.message_count = 0
        self.snapshot_count = 0
        self.started_wallclock = float(time.time())

        self.asset_ids = discover_polymarket_asset_ids(
            gamma_url=self.gamma_url,
            market_limit=self.market_limit,
            active_only=self.active_only,
            closed=self.closed,
            timeout_sec=self.feed_timeout_sec,
        )

    def _connect(self) -> websocket.WebSocket:
        ws = websocket.create_connection(self.ws_url, timeout=float(self.feed_timeout_sec))
        sub_payload = {
            "assets_ids": self.asset_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }
        ws.send(json.dumps(sub_payload))
        return ws

    def snapshots(self) -> Iterator[MarketSnapshot]:
        print("[datafeed] mode=live")
        print("[datafeed] subscribing asset_ids=%d" % len(self.asset_ids))

        reconnect_attempt = 0
        ws: Optional[websocket.WebSocket] = None
        last_ping = time.time()
        raw_logged = 0
        normalized_logged = 0

        while True:
            try:
                if ws is None:
                    ws = self._connect()
                    reconnect_attempt = 0
                    last_ping = time.time()
                    print("[datafeed] websocket connected")

                if (time.time() - last_ping) >= self.ping_interval_sec:
                    ws.ping()
                    last_ping = time.time()

                raw = ws.recv()
                if raw is None:
                    continue
                self.message_count += 1
                self.last_message_wallclock = float(time.time())

                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="ignore")

                if self.debug_raw and raw_logged < self.debug_raw_limit:
                    _safe_append_jsonl(
                        self.debug_log_path,
                        {
                            "kind": "raw_message",
                            "ts": float(time.time()),
                            "payload": raw,
                        },
                    )
                    raw_logged += 1

                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    continue

                messages: List[Dict[str, Any]] = []
                if isinstance(payload, dict):
                    messages = [payload]
                elif isinstance(payload, list):
                    messages = [m for m in payload if isinstance(m, dict)]

                for message in messages:
                    if not _is_best_bid_ask_event(message):
                        continue
                    snapshot = _normalize_message_to_snapshot(message, default_market_id="polymarket")
                    if snapshot is not None:
                        receive_wallclock = float(time.time())
                        self.snapshot_count += 1
                        self.last_snapshot_wallclock = receive_wallclock
                        self.last_snapshot_ts = float(snapshot.ts)
                        if self.debug_raw and normalized_logged < self.debug_raw_limit:
                            _safe_append_jsonl(
                                self.debug_log_path,
                                {
                                    "kind": "normalized_snapshot",
                                    "ts": float(time.time()),
                                    "payload": snapshot.__dict__,
                                },
                            )
                            normalized_logged += 1
                        yield snapshot

            except KeyboardInterrupt:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                raise
            except Exception:
                reconnect_attempt += 1
                self.reconnect_count += 1
                backoff = min(30.0, 1.0 + reconnect_attempt * 2.0)
                print("[datafeed] websocket reconnect attempt=%d sleep=%.1fs" % (reconnect_attempt, backoff))
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                    ws = None
                time.sleep(backoff)

    def monitoring_state(self) -> Dict[str, Any]:
        return {
            "reconnect_count": int(self.reconnect_count),
            "last_message_wallclock": self.last_message_wallclock,
            "last_snapshot_wallclock": self.last_snapshot_wallclock,
            "last_snapshot_ts": self.last_snapshot_ts,
            "message_count": int(self.message_count),
            "snapshot_count": int(self.snapshot_count),
            "started_wallclock": self.started_wallclock,
        }


def get_default_feed(
    market_id: str = "mock_market_1",
    tick_interval_sec: float = 0.25,
    total_ticks: int = 120,
):
    mode = os.getenv("POLYMARKET_MODE", "mock").strip().lower()

    if mode == "live":
        return PolymarketLiveDataFeed()

    print("[datafeed] mode=mock")
    return MockDataFeed(
        market_id=market_id,
        start_mid=0.50,
        base_spread=0.02,
        tick_interval_sec=tick_interval_sec,
        total_ticks=total_ticks,
    )
