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

"""The kv_hints envelope and the retention-table lookup."""

import json
from pathlib import Path
from typing import Any

from inference_perf.policies import TableRetentionPolicy, build_kv_hints


def _policy(tmp_path: Path, table: dict[str, Any]) -> TableRetentionPolicy:
    p = tmp_path / "belady.json"
    p.write_text(json.dumps(table))
    return TableRetentionPolicy(str(p), duration_s=86400.0)


def test_build_kv_hints_wraps_directives_in_a_retain_action() -> None:
    directives = [{"start": 0, "end": 16, "priority": 80, "duration": 30.0}]
    env = build_kv_hints("sess-1", directives, "ev-9")
    assert env["protocol_version"] == "0.1"
    assert env["message_id"] == "ev-9"
    (action,) = env["actions"]
    assert action["action_id"] == "ev-9:retain"
    assert action["action_type"] == "kv.retain"
    assert action["action_version"] == "1.0"
    assert action["payload"] == {"scope": "sess-1", "directives": directives}


def test_build_kv_hints_without_scope() -> None:
    env = build_kv_hints(None, [], "ev-1")
    assert env["actions"][0]["payload"]["scope"] is None


def test_table_runs_are_emitted_verbatim(tmp_path: Path) -> None:
    pol = _policy(tmp_path, {"s:e1": [[0, 960, 91], [960, 3056, 79], [3056, None, 0]]})
    ds = pol.directives("s:e1")
    assert ds is not None
    assert [(d["start"], d["end"], d["priority"]) for d in ds] == [(0, 960, 91), (960, 3056, 79), (3056, None, 0)]
    assert all(d["duration"] == 86400.0 for d in ds)


def test_event_missing_from_table_gets_no_directives(tmp_path: Path) -> None:
    pol = _policy(tmp_path, {"s:e1": [[0, 16, 50]]})
    assert pol.directives("s:e2") is None


def test_prefix_shift_moves_offsets_by_the_deltas_of_preceding_messages() -> None:
    from inference_perf.policies.retention import prefix_shift

    # recorded prompt: messages[:2] end at block 10 (token 160), messages[:4] at block 25 (token 400)
    bounds = [[2, 0, 10], [4, 0, 25]]
    # message 1 (an assistant turn) came back 30 tokens longer live; message 3, 50 shorter
    deltas = [0, 30, 0, -50, 0]
    shift = prefix_shift(bounds, deltas)
    assert shift(0) == 0
    assert shift(160) == 190  # after messages 0..1: +30
    assert shift(400) == 380  # after messages 0..3: +30 - 50
    assert shift(96) == 96  # inside the first two messages: no substituted message before it


def test_table_directives_are_shifted_when_message_deltas_are_given(tmp_path: Path) -> None:
    # dict entry: runs over the recorded prompt, bounds = message-prefix depths in blocks
    table = {
        "s:e1": {
            "ntok": 480,
            "runs": [[0, 160, 90], [160, 400, 60], [400, None, 0]],
            "bounds": [[2, 0, 10], [4, 0, 25]],
        }
    }
    policy = _policy(tmp_path, table)
    plain = policy.directives("s:e1")
    shifted = policy.directives("s:e1", msg_deltas=[0, 30, 0, -50, 0])
    assert plain is not None and shifted is not None
    assert [(d["start"], d["end"]) for d in plain[:2]] == [(0, 160), (160, 400)]
    assert [(d["start"], d["end"]) for d in shifted[:2]] == [(0, 190), (190, 380)]
    assert [d["priority"] for d in shifted] == [d["priority"] for d in plain]


def test_recorded_outputs_close_the_open_range_and_release_the_generated_tail(tmp_path: Path) -> None:
    """With recorded outputs the generated tokens never reach a later prompt, so they must
    not inherit the last range's protection."""
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"e": {"runs": [[0, 32, 90], [32, None, 70]], "ntok": 40}}))
    out = TableRetentionPolicy(str(p), duration_s=60.0, protect_output=False).directives("e")
    assert [(d["start"], d["end"], d["priority"]) for d in out] == [(0, 32, 90), (32, 40, 70), (40, None, 0)]


def test_live_outputs_keep_the_open_range_protecting_the_generated_tail(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"e": {"runs": [[0, 32, 90], [32, None, 70]], "ntok": 40}}))
    out = TableRetentionPolicy(str(p), duration_s=60.0, protect_output=True).directives("e")
    assert [(d["start"], d["end"], d["priority"]) for d in out] == [(0, 32, 90), (32, None, 70)]
