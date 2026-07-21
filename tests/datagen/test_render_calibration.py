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

"""Render-based exact coordinate calibration: anchor→token mapping via
mocked /render responses (no server needed)."""

import asyncio
from types import SimpleNamespace

import inference_perf.datagen.replay_graph_session_datagen as m
from inference_perf.datagen.replay_graph_session_datagen import (
    SessionChatCompletionAPIData,
    _lcp_len,
)
from inference_perf.models import ReuseSegment


def _obj(profile, messages, max_tokens=100):
    o = SessionChatCompletionAPIData.model_construct(
        event_id="s1:e1",
        reuse_depth_profile=profile,
        retention_policy=SimpleNamespace(render_url="http://mock"),
    )
    payload = {
        "model": "m",
        "messages": messages,
        "max_tokens": max_tokens,
    }
    return o, payload


def _fake_render(token_map):
    """token_map: #messages -> token id list to return."""
    async def fake(url, body, cacheable=False):
        return token_map[len(body["messages"])]
    return fake


MSGS = [{"role": "user", "content": f"m{i}"} for i in range(4)]
# full render = 100 tokens; prefixes diverge from full at 30 / 70
FULL = list(range(100))
TOKMAP = {
    4: FULL,
    1: FULL[:30] + [999],          # msgs[:1] → boundary 30 (suffix diverges)
    2: FULL[:70] + [999, 998],     # msgs[:2] → boundary 70
}


def test_lcp():
    assert _lcp_len([1, 2, 3], [1, 2, 9]) == 2
    assert _lcp_len([], [1]) == 0
    assert _lcp_len([1], [1]) == 1


def test_anchor_mapping_and_monotonicity(monkeypatch):
    monkeypatch.setattr(m, "_render_token_ids", _fake_render(TOKMAP))
    profile = [
        ReuseSegment(start=0, end=999, breadth=3, cold_gap=1, end_msg=1),
        ReuseSegment(start=999, end=5000, breadth=2, cold_gap=2, end_msg=2),
    ]
    o, payload = _obj(profile, MSGS)
    out = asyncio.run(o._calibrate_profile_via_render(payload))
    assert [(s.start, s.end) for s in out] == [(0, 30), (30, 70)]
    assert [s.breadth for s in out] == [3, 2]  # metadata preserved


def test_covers_output_and_duplicate_anchor_collapse(monkeypatch):
    monkeypatch.setattr(m, "_render_token_ids", _fake_render(TOKMAP))
    # Both boundaries share the same structural anchor (full msgs + output) —
    # recorded token values scatter, but calibration collapses them.
    profile = [
        ReuseSegment(start=0, end=2678, breadth=6, cold_gap=1,
                     end_msg=4, covers_output=True),
        ReuseSegment(start=2678, end=48221, breadth=1, cold_gap=9,
                     end_msg=4, covers_output=True),
    ]
    o, payload = _obj(profile, MSGS, max_tokens=100)
    out = asyncio.run(o._calibrate_profile_via_render(payload))
    # prompt(100) + max_tokens cap(100) = 200; duplicate collapses to zero width
    assert (out[0].start, out[0].end) == (0, 200)
    assert (out[1].start, out[1].end) == (200, 200)


def test_fallback_on_render_failure(monkeypatch):
    async def boom(url, body, cacheable=False):
        raise RuntimeError("down")
    monkeypatch.setattr(m, "_render_token_ids", boom)
    profile = [ReuseSegment(start=0, end=10, breadth=1, end_msg=1)]
    o, payload = _obj(profile, MSGS)
    out = asyncio.run(o._calibrate_profile_via_render(payload))
    assert out is None  # caller falls back to char-ratio rescale


def test_unanchored_profile_returns_none():
    profile = [ReuseSegment(start=0, end=10, breadth=1)]  # legacy, no anchors
    o, payload = _obj(profile, MSGS)
    out = asyncio.run(o._calibrate_profile_via_render(payload))
    assert out is None
