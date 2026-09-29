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

"""A turn in the retention table carries its runs as a kv_hints envelope.

Checked on an as_recorded tool-call turn, whose tool-choice step returns early:
the envelope must be attached before it.
"""

import asyncio
import json
from pathlib import Path

from inference_perf.apis.chat import ChatMessage
from inference_perf.config.datagen.replay import ToolChoiceMode
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)
from inference_perf.policies import TableRetentionPolicy


def test_table_entry_becomes_kv_hints_on_an_as_recorded_turn(tmp_path: Path) -> None:
    p = tmp_path / "belady.json"
    p.write_text(json.dumps({"sid:e1": [[0, 32, 60], [32, None, 0]]}))
    data = SessionChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="hello")],
        event_id="sid:e1",
        registry=EventOutputRegistry(),
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
        expected_output_is_tool_call=True,
        tool_definitions=[{"type": "function", "name": "f"}],
        tool_choice_mode=ToolChoiceMode.AS_RECORDED,
        retention_policy=TableRetentionPolicy(str(p), duration_s=86400.0),
    )
    payload = asyncio.run(data.to_request_body("m", 100, ignore_eos=False, streaming=False))
    (action,) = payload["extra_body"]["kv_hints"]["actions"]
    assert action["payload"]["scope"] == "sid"
    assert action["payload"]["directives"] == [
        {"start": 0, "end": 32, "priority": 60, "duration": 86400.0},
        {"start": 32, "end": None, "priority": 0, "duration": 86400.0},
    ]
    assert "tool_choice" not in payload
