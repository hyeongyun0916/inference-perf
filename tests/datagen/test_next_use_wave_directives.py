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
    ranges = [d for d in directives if not d.get("covers_output")]
    assert len(ranges) == 1
    assert ranges[0]["priority"] == 75
    assert ranges[0]["duration"] == 2 * 10.0 + 70.0


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


def test_floor_tail_covers_non_reused_remainder(monkeypatch):
    # Forward reuse stops short of the full prompt -> the tail [prev, len) must
    # be FLOOR-covered so no prompt block is left uncovered (owner-clear guard).
    monkeypatch.setattr(dg, "_render_token_ids", _render_stub(0))
    pol = WorkflowAwarePolicy(
        priority_mode="next_use_wave", k_wave=25, per_wave_s=10.0,
        queue_margin_s=60.0, ttl_buffer_s=10.0, render_url="http://x",
    )
    # 2 messages -> full_ids [0,1,2]; seg reuses only the first message -> depth
    # 2, leaving [2,3) as the non-reused tail.
    segs = ((1, 0, 3, 1.0, 2),)
    obj = _api(segs, False, pol)
    payload = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
    }
    directives = asyncio.run(obj._forward_reuse_directives(payload))
    ranges = [d for d in directives if not d.get("covers_output")]
    assert len(ranges) == 2
    assert ranges[0]["priority"] == 75  # forward reuse keeps its priority
    tail = ranges[-1]
    assert tail["start"] == 2 and tail["end"] == 3
    assert tail["priority"] == 1  # FLOOR
    assert tail["duration"] == 60.0 + 10.0  # queue_margin + ttl_buffer


def test_output_is_floor_covered_when_not_forward_reused(monkeypatch):
    # The turn's output blocks are cached regardless of downstream reuse. Left
    # uncovered, the server owner-clears the protection a later turn put on the
    # block it reuses, so the output must carry a FLOOR directive too.
    monkeypatch.setattr(dg, "_render_token_ids", _render_stub(0))
    pol = WorkflowAwarePolicy(
        priority_mode="next_use_wave", k_wave=25, per_wave_s=10.0,
        queue_margin_s=60.0, ttl_buffer_s=10.0, render_url="http://x",
    )
    segs = ((1, 0, 3, 1.0, 2),)
    obj = _api(segs, False, pol)  # forward_covers_output = False
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    directives = asyncio.run(obj._forward_reuse_directives(payload))
    out = [d for d in directives if d.get("covers_output")]
    assert len(out) == 1
    assert out[0]["priority"] == 1  # FLOOR
    assert out[0]["duration"] == 60.0 + 10.0


def test_no_forward_reuse_emits_floor_full_prefix(monkeypatch):
    # The owner-clear fix: a turn that reuses nothing downstream used to return
    # [] (no directive -> server owner-clears the shared prefix it re-caches).
    # It now emits a single FLOOR directive over the whole prompt [0, len).
    monkeypatch.setattr(dg, "_render_token_ids", _render_stub(0))
    pol = WorkflowAwarePolicy(
        priority_mode="next_use_wave", k_wave=25, per_wave_s=10.0,
        queue_margin_s=60.0, ttl_buffer_s=10.0, render_url="http://x",
    )
    obj = _api((), False, pol)  # no forward segments, no covered output
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    directives = asyncio.run(obj._forward_reuse_directives(payload))
    ranges = [d for d in directives if not d.get("covers_output")]
    assert len(ranges) == 1
    d = ranges[0]
    assert d["start"] == 0 and d["end"] == 2  # full prompt (full_ids [0,1])
    assert d["priority"] == 1  # FLOOR
    assert d["duration"] == 60.0 + 10.0
