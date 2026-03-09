import json
import os
import time
from typing import Any, Dict


class Journal:
    def __init__(self, base_dir: str = "binary_bot/logs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.files = {
            "events": os.path.join(base_dir, "events.jsonl"),
            "orders": os.path.join(base_dir, "orders.jsonl"),
            "fills": os.path.join(base_dir, "fills.jsonl"),
            "snapshots": os.path.join(base_dir, "snapshots.jsonl"),
        }

    def _write(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def event(self, event_type: str, payload: Dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "event_type": event_type,
            "payload": payload,
        }
        self._write(self.files["events"], row)

    def order(self, payload: Dict[str, Any]) -> None:
        self._write(self.files["orders"], payload)

    def fill(self, payload: Dict[str, Any]) -> None:
        self._write(self.files["fills"], payload)

    def snapshot(self, payload: Dict[str, Any]) -> None:
        self._write(self.files["snapshots"], payload)
