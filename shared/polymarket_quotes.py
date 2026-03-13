from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests


POLYMARKET_CLOB_BASE_URL = os.getenv("POLYMARKET_CLOB_BASE_URL", "https://clob.polymarket.com").rstrip("/")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_price(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _extract_price_size(level: Any) -> Tuple[float | None, float | None]:
    if isinstance(level, dict):
        price = _safe_float(
            level.get("price", level.get("px", level.get("rate", level.get("value"))))
        )
        size = _safe_float(
            level.get("size", level.get("sz", level.get("quantity", level.get("amount"))))
        )
        return _clamp_price(price), size

    if isinstance(level, (list, tuple)) and len(level) >= 1:
        price = _clamp_price(_safe_float(level[0]))
        size = _safe_float(level[1]) if len(level) >= 2 else None
        return price, size

    return None, None


def _all_valid_levels(levels: Any) -> List[Tuple[float, float | None]]:
    if not isinstance(levels, list):
        return []
    pairs: List[Tuple[float, float | None]] = []
    for level in levels:
        price, size = _extract_price_size(level)
        if price is not None:
            pairs.append((float(_clamp_price(price)), size))
    return pairs


def _candidate_payloads(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = [payload]
    for key in ("data", "book", "result"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    return candidates


def _parse_top_of_book(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {"bid": None, "ask": None, "bid_size": None, "ask_size": None}

    for candidate in _candidate_payloads(payload):
        bid = _clamp_price(
            _safe_float(
                candidate.get(
                    "best_bid",
                    candidate.get("bestBid", candidate.get("bid")),
                )
            )
        )
        ask = _clamp_price(
            _safe_float(
                candidate.get(
                    "best_ask",
                    candidate.get("bestAsk", candidate.get("ask")),
                )
            )
        )
        if bid is not None or ask is not None:
            return {
                "bid": bid,
                "ask": ask,
                "bid_size": None,
                "ask_size": None,
            }

        bids = candidate.get("bids")
        asks = candidate.get("asks")
        valid_bids = _all_valid_levels(bids)
        valid_asks = _all_valid_levels(asks)
        best_bid_level = max(valid_bids, key=lambda item: item[0]) if valid_bids else None
        best_ask_level = min(valid_asks, key=lambda item: item[0]) if valid_asks else None
        bid = best_bid_level[0] if best_bid_level is not None else None
        ask = best_ask_level[0] if best_ask_level is not None else None
        if bid is not None or ask is not None:
            return {
                "bid": bid,
                "ask": ask,
                "bid_size": best_bid_level[1] if best_bid_level is not None else None,
                "ask_size": best_ask_level[1] if best_ask_level is not None else None,
            }

    return {"bid": None, "ask": None, "bid_size": None, "ask_size": None}


def get_book(asset_id: str) -> dict:
    asset_id = str(asset_id or "").strip()
    if not asset_id:
        return {
            "asset_id": asset_id,
            "bid": None,
            "ask": None,
            "bid_size": None,
            "ask_size": None,
            "raw": {},
        }

    url = f"{POLYMARKET_CLOB_BASE_URL}/book"
    try:
        response = requests.get(url, params={"token_id": asset_id}, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError("failed to fetch Polymarket book for asset_id=%s: %s" % (asset_id, exc)) from exc

    if not isinstance(payload, dict):
        payload = {}

    top = _parse_top_of_book(payload)
    return {
        "asset_id": asset_id,
        "bid": top.get("bid"),
        "ask": top.get("ask"),
        "bid_size": top.get("bid_size"),
        "ask_size": top.get("ask_size"),
        "raw": payload,
    }


def get_binary_quotes(yes_asset_id: str, no_asset_id: str) -> dict:
    yes_book = get_book(yes_asset_id)
    no_book = get_book(no_asset_id)
    return {
        "yes_asset_id": str(yes_asset_id or ""),
        "no_asset_id": str(no_asset_id or ""),
        "yes_bid": yes_book.get("bid"),
        "yes_ask": yes_book.get("ask"),
        "yes_bid_size": yes_book.get("bid_size"),
        "yes_ask_size": yes_book.get("ask_size"),
        "no_bid": no_book.get("bid"),
        "no_ask": no_book.get("ask"),
        "no_bid_size": no_book.get("bid_size"),
        "no_ask_size": no_book.get("ask_size"),
    }


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python3 -m shared.polymarket_quotes <YES_ASSET_ID> <NO_ASSET_ID>")
        return

    payload = get_binary_quotes(sys.argv[1], sys.argv[2])
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
