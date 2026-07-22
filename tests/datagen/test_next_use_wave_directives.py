# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""next_use_wave: _forward_reuse_directives maps wave gaps -> priority/duration.

Renders are monkeypatched to deterministic token ids so the boundary depths are
predictable; the assertion is on which policy mapping the directive uses.
"""

import asyncio

import inference_perf.datagen.replay_graph_session_datagen as dg
from inference_perf.policies import WorkflowAwarePolicy


def _render_stub(n_tokens):
    # message-count -> token ids [0..k); more messages -> longer prefix.
    async def _r(url, body, cacheable=False):
        return list(range(len(body.get("messages", [])) + 1))
    return _r


def _api(segs, covers, policy):
    # model_construct (not __new__) so pydantic's private/fields-set state is
    # initialized; __new__ alone leaves __setattr__ unable to run (see report).
    return dg.SessionChatCompletionAPIData.model_construct(
        forward_segments=segs,
        forward_covers_output=covers,
        retention_policy=policy,
        event_id="t0",
    )


def test_directive_priority_from_wave_gap(monkeypatch):
    monkeypatch.setattr(dg, "_render_token_ids", _render_stub(0))
    pol = WorkflowAwarePolicy(
        priority_mode="next_use_wave", k_wave=25, per_wave_s=10.0,
        queue_margin_s=60.0, ttl_buffer_s=10.0, render_url="http://x",
    )
    # one boundary at 1 message, min_gap 1.0 -> priority 75, max_gap 2 -> ttl 90.
    segs = ((1, 0, 3, 1.0, 2),)
    obj = _api(segs, False, pol)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    directives = asyncio.run(obj._forward_reuse_directives(payload))
    assert len(directives) == 1
    assert directives[0]["priority"] == 75
    assert directives[0]["duration"] == 2 * 10.0 + 70.0


def test_directive_floor_when_gap_none(monkeypatch):
    monkeypatch.setattr(dg, "_render_token_ids", _render_stub(0))
    pol = WorkflowAwarePolicy(
        priority_mode="next_use_wave", k_wave=25, per_wave_s=10.0,
        queue_margin_s=60.0, ttl_buffer_s=10.0, render_url="http://x",
    )
    segs = ((1, 0, 1, None, None),)  # covered-only, no forward-future reuser
    obj = _api(segs, False, pol)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    directives = asyncio.run(obj._forward_reuse_directives(payload))
    assert directives[0]["priority"] == 1
    assert directives[0]["duration"] == 70.0


def test_tiered_mode_unchanged(monkeypatch):
    monkeypatch.setattr(dg, "_render_token_ids", _render_stub(0))
    pol = WorkflowAwarePolicy(
        priority_mode="tiered", high_breadth_priority=90, mid_breadth_priority=70,
        low_breadth_priority=50, per_span_s=9.0, queue_margin_s=10.0,
        ttl_buffer_s=5.0, render_url="http://x",
    )
    segs = ((1, 0, 4, 1.0, 2),)  # breadth 4 -> tiered high 90; wave fields ignored
    obj = _api(segs, False, pol)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    directives = asyncio.run(obj._forward_reuse_directives(payload))
    assert directives[0]["priority"] == 90
    assert directives[0]["duration"] == 4 * 9.0 + 10.0 + 5.0


def test_covers_output_uses_last_seg_wave(monkeypatch):
    monkeypatch.setattr(dg, "_render_token_ids", _render_stub(0))
    pol = WorkflowAwarePolicy(
        priority_mode="next_use_wave", k_wave=25, per_wave_s=10.0,
        queue_margin_s=60.0, ttl_buffer_s=10.0, render_url="http://x",
    )
    # one boundary -> a real directive, so last_min_gap/last_max_gap get set;
    # covers=True -> the covers_output directive must inherit those wave gaps.
    segs = ((1, 0, 3, 1.0, 2),)
    obj = _api(segs, True, pol)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    directives = asyncio.run(obj._forward_reuse_directives(payload))
    assert len(directives) == 2
    assert directives[-1]["covers_output"] is True
    assert directives[-1]["priority"] == 75  # _priority_for_wave(1.0)
    assert directives[-1]["duration"] == 2 * 10.0 + 70.0  # _ttl_for_wave(2)
