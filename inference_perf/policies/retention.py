# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KV-cache retention hints sent with trace-replay requests.

A retention table, computed offline from a recorded run, maps each event to
token-range runs `[start, end|null, priority]`. The runs for an event are sent
verbatim as the directives of a `kv.retain` action in a KvHints envelope, as
the top-level `kv_hints` request field.
"""

import json
from typing import Any

KV_HINTS_PROTOCOL_VERSION = "0.1"
RETAIN_ACTION_TYPE = "kv.retain"
RETAIN_ACTION_VERSION = "1.0"


def build_kv_hints(scope: str | None, directives: list[dict[str, Any]], event_id: str) -> dict[str, Any]:
    """Wrap retention directives in the vLLM KvHints envelope (a single
    ``kv.retain`` action). Sent as the top-level ``kv_hints`` request field."""
    return {
        "protocol_version": KV_HINTS_PROTOCOL_VERSION,
        "message_id": event_id,
        "actions": [
            {
                "action_id": f"{event_id}:retain",
                "action_type": RETAIN_ACTION_TYPE,
                "action_version": RETAIN_ACTION_VERSION,
                "payload": {"scope": scope, "directives": directives},
            }
        ],
    }


class TableRetentionPolicy:
    """Retention directives looked up from a precomputed per-event table."""

    def __init__(self, table_path: str, duration_s: float) -> None:
        self.table_path = table_path
        self.duration_s = duration_s
        self._table: dict[str, list[list[Any]]] | None = None

    def directives(self, event_key: str) -> list[dict[str, Any]] | None:
        """The event's precomputed token-range runs, emitted as directives
        verbatim. Each run is [start, end|None, priority]; runs are
        non-increasing in priority along the prompt and the final run is the
        release of the tail nothing downstream reuses. None when the table has
        no entry (the turn then falls to LRU)."""
        if self._table is None:
            with open(self.table_path) as f:
                self._table = json.load(f)
        runs = self._table.get(event_key)
        if not runs:
            return None
        return [
            {
                "start": int(a),
                "end": (None if b is None else int(b)),
                "priority": int(v),
                "duration": self.duration_s,
            }
            for a, b, v in runs
        ]
