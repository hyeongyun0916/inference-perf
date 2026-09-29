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
