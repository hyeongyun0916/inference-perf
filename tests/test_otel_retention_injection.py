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
