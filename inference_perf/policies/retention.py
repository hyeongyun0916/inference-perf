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
from typing import Any, Callable, Optional

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


def finalize_output(
    out: list[dict[str, Any]], protect_output: bool, prompt_end: Optional[int], duration_s: float
) -> list[dict[str, Any]]:
    """Shape the generated-token side of the directives.

    With live outputs the next turn's prompt contains this turn's output, so an
    open last range (end null) keeps protecting it. With recorded outputs the
    generated tokens are never reused: every protected range is closed at the
    prompt end and the rest is released explicitly with ``[prompt_end, null, 0]``."""
    if protect_output or not out:
        return out
    last = out[-1]
    if last["end"] is None and last["priority"] == 0:
        return out
    if last["end"] is None:
        if prompt_end is None:
            return out
        last["end"] = max(int(prompt_end), int(last["start"]) + 1)
    out.append({"start": int(last["end"]), "end": None, "priority": 0, "duration": duration_s})
    return out


def _shift_for(entry: Any, msg_deltas: Optional[list[int]]) -> Optional[Callable[[int], int]]:
    bounds = entry.get("bounds") if isinstance(entry, dict) else None
    if not bounds or not msg_deltas or not any(msg_deltas):
        return None
    return prefix_shift(bounds, msg_deltas)


def _shifted(out: list[dict[str, Any]], shift: Callable[[int], int]) -> list[dict[str, Any]]:
    res: list[dict[str, Any]] = []
    for d in out:
        a = shift(d["start"])
        b = None if d["end"] is None else shift(d["end"])
        if res and a <= res[-1]["start"]:
            a = res[-1]["start"] + 1  # keep ranges strictly advancing
        if b is not None and b <= a:
            b = a + 1
        if res:
            res[-1]["end"] = a  # ranges stay contiguous after the shift
        res.append({**d, "start": int(a), "end": (None if b is None else int(b))})
    return res


class TableRetentionPolicy:
    """Retention directives looked up from a precomputed per-event table."""

    def __init__(self, table_path: str, duration_s: float, protect_output: bool = True) -> None:
        self.table_path = table_path
        self.duration_s = duration_s
        self.protect_output = protect_output
        self._table: dict[str, Any] | None = None

    def directives(self, event_key: str, msg_deltas: Optional[list[int]] = None) -> list[dict[str, Any]] | None:
        """The event's precomputed token-range runs, emitted as directives
        verbatim. Each run is [start, end|None, priority]; runs are
        non-increasing in priority along the prompt and the final run is the
        release of the tail nothing downstream reuses. None when the table has
        no entry (the turn then falls to LRU)."""
        if self._table is None:
            with open(self.table_path) as f:
                self._table = json.load(f)
        entry = self._table.get(event_key)
        runs = entry.get("runs") if isinstance(entry, dict) else entry
        if not runs:
            return None
        out = [
            {
                "start": int(a),
                "end": (None if b is None else int(b)),
                "priority": int(v),
                "duration": self.duration_s,
            }
            for a, b, v in runs
        ]
        shift = _shift_for(entry, msg_deltas)
        if shift is not None:
            out = _shifted(out, shift)
        ntok = entry.get("ntok") if isinstance(entry, dict) else None
        prompt_end = None if ntok is None else (shift(int(ntok)) if shift is not None else int(ntok))
        return finalize_output(out, self.protect_output, prompt_end, self.duration_s)


def prefix_shift(bounds: list[list[Any]], msg_deltas: list[int]) -> Callable[[int], int]:
    """Map a recorded-prompt token offset onto the live prompt.

    ``bounds`` are the recorded prompt's message-prefix boundaries, [w, p,
    blocks]: rendering the first ``w`` messages (plus ``p`` chars of the next)
    shares ``blocks`` 16-token blocks with the full recorded prompt. A run
    boundary at token ``t`` lies after the messages of the largest bound at or
    before it, so it moves by the summed token delta of those messages.
    ``msg_deltas[i]`` is the live-minus-recorded token count of message ``i``."""
    cum = [0]
    for d in msg_deltas:
        cum.append(cum[-1] + int(d))
    pts = sorted((int(blocks) * 16, int(w)) for w, _p, blocks in bounds)

    def shift(t: int) -> int:
        w = 0
        for tok, ww in pts:
            if tok <= t:
                w = max(w, ww)
            else:
                break
        w = min(w, len(msg_deltas))
        return max(0, int(t) + cum[w])

    return shift
