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

"""Retention directive injection on the OTel trace replay path.

These tests exercise SessionChatCompletionAPIData.to_request_body to
confirm that, when a retention policy is attached, the resulting payload
carries `extra_body.retention_directives` shaped as the policy expects.
"""

import asyncio

from inference_perf.apis.chat import ChatMessage
from inference_perf.datagen.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)
from inference_perf.policies.retention import WorkflowAwarePolicy


def _make_data(
    *,
    retention_policy,
    system_prompt_tokens=200,
    wait_ms=10000,
    expected_output_is_tool_call=True,
) -> SessionChatCompletionAPIData:
    return SessionChatCompletionAPIData(
        messages=[
            ChatMessage(role="system", content="x" * (system_prompt_tokens * 4)),
            ChatMessage(role="user", content="hello"),
        ],
        event_id="session-A:event-1",
        registry=EventOutputRegistry(),
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=2,
        wait_ms=wait_ms,
        expected_output_is_tool_call=expected_output_is_tool_call,
        retention_policy=retention_policy,
    )


def _to_payload(data: SessionChatCompletionAPIData) -> dict:
    return asyncio.run(data.to_request_body("test-model", 100, ignore_eos=False, streaming=False))


def test_no_retention_policy_omits_extra_body():
    payload = _to_payload(_make_data(retention_policy=None))
    assert "extra_body" not in payload or "retention_directives" not in payload.get("extra_body", {})


def test_workflow_aware_unused_block_emits_no_directive():
    """Terminal block (output never reused): no directive → immediate evict."""
    payload = _to_payload(_make_data(
        retention_policy=WorkflowAwarePolicy(),
    ))
    extra = payload.get("extra_body") or {}
    assert "retention_directives" not in extra


# --- Forward-increment reuse protection (render-LCP, proactive per turn) ------
from inference_perf.datagen import replay_graph_session_datagen as _dg  # noqa: E402
from inference_perf.datagen.replay_graph_types import InputSegment  # noqa: E402


def test_forward_directive_protects_new_increment(monkeypatch):
    """Protect THIS turn's own new content [depth, prompt_len + max_tokens).

    Full render -> [1,2,3,4,5] (prompt_len 5); the 1-message inherited prefix
    -> [1,2,3], so LCP depth = 3 (where inherited history ends). With
    max_tokens=10 the new-increment + output span is [3, 5+10) = [3, 15)."""
    async def fake_render(render_url, body, cacheable=False):
        # 1 message == inherited-prefix render; 2 == the full request.
        return [1, 2, 3] if len(body["messages"]) == 1 else [1, 2, 3, 4, 5]

    monkeypatch.setattr(_dg, "_render_token_ids", fake_render)
    data = _make_data(retention_policy=WorkflowAwarePolicy(render_url="http://render"))
    data.input_segments = [
        InputSegment(type="shared", message_count=1, token_count=48, source_event_id="prev"),
        InputSegment(type="unique", message_count=1, token_count=8),
    ]
    data.remaining_reuse = 5  # breadth >= 4 -> high-breadth priority tier

    payload = {"model": "m", "max_tokens": 10,
               "messages": [{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"}]}
    dirs = asyncio.run(data._forward_reuse_directives(payload))
    assert dirs is not None and len(dirs) == 1
    assert dirs[0]["start"] == 3 and dirs[0]["end"] == 15
    assert dirs[0]["priority"] == WorkflowAwarePolicy().high_breadth_priority
    assert dirs[0]["duration"] > 0


def test_no_downstream_reuse_emits_no_directive(monkeypatch):
    """A turn nothing reuses downstream (remaining_reuse == 0) emits nothing,
    without even rendering."""
    async def fake_render(render_url, body, cacheable=False):  # must not be needed
        raise AssertionError("render should not run when nothing reuses this turn")

    monkeypatch.setattr(_dg, "_render_token_ids", fake_render)
    data = _make_data(retention_policy=WorkflowAwarePolicy(render_url="http://render"))
    data.input_segments = [InputSegment(type="unique", message_count=2, token_count=16)]
    data.remaining_reuse = 0
    dirs = asyncio.run(data._forward_reuse_directives(
        {"model": "m", "messages": [{"role": "user", "content": "u"}]}))
    assert dirs == []


def test_forward_render_failure_falls_back_to_none(monkeypatch):
    """Render failure returns None so the caller uses the legacy profile path."""
    async def boom(render_url, body, cacheable=False):
        raise RuntimeError("render down")

    monkeypatch.setattr(_dg, "_render_token_ids", boom)
    data = _make_data(retention_policy=WorkflowAwarePolicy(render_url="http://render"))
    data.input_segments = [
        InputSegment(type="shared", message_count=1, token_count=48, source_event_id="prev"),
        InputSegment(type="unique", message_count=1, token_count=8),
    ]
    data.remaining_reuse = 5  # must reach the render to exercise its failure
    dirs = asyncio.run(data._forward_reuse_directives(
        {"model": "m", "messages": [{"role": "system", "content": "s"},
                                    {"role": "user", "content": "u"}]}))
    assert dirs is None
