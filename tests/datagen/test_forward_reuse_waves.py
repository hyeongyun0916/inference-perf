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

"""Wave (causal-level) gaps per depth boundary in _forward_reuse_depths.

wave(node) = 0 for a root, else 1 + max(wave(pred)). Per boundary, min_gap /
max_gap are the min / max wave-gap (wave[reuser] - wave[target]) over
forward-future reusers (wave-gap >= 0), min_gap carrying a +(1 - 1/ts_gap)
within-wave t_start tiebreak. registry=None -> messages verbatim.
"""

from types import SimpleNamespace

from inference_perf.datagen.replay_graph_session_datagen import (
    _forward_reuse_depths,
)


def _m(role, content):
    return {"role": role, "content": content}


def _ev(event_id, t, preds, messages):
    return SimpleNamespace(
        event_id=event_id, t_start_ms=t,
        predecessor_event_ids=list(preds),
        call=SimpleNamespace(messages=messages),
    )


def test_same_wave_cousin_gap_zero():
    # Two roots (both wave 0) sharing the system prefix -> gap 0 for target t0.
    sys = _m("system", "S")
    t0 = _ev("t0", 0, [], [sys, _m("user", "A")])
    t1 = _ev("t1", 10, [], [sys, _m("user", "B")])  # root, shares sys, wave 0
    out = _forward_reuse_depths([t0, t1], target="t0")
    segs, _covers = out["t0"]
    # one boundary (the shared system message); min_gap ~ 0 + tiebreak(ts_gap=1)=0
    (bw, bp, breadth, min_gap, max_gap) = segs[0]
    assert breadth == 1
    assert min_gap == 0.0 and max_gap == 0


def test_descendant_gap_one_per_wave():
    # Chain t0(root, w0) -> t1(w1): t1 reuses t0's prefix; gap = 1.
    sys = _m("system", "S")
    t0 = _ev("t0", 0, [], [sys, _m("user", "Q")])
    t1 = _ev("t1", 10, ["t0"], [sys, _m("user", "Q"), _m("assistant", "R")])
    out = _forward_reuse_depths([t0, t1], target="t0")
    segs, _covers = out["t0"]
    (bw, bp, breadth, min_gap, max_gap) = segs[0]
    assert breadth == 1
    assert min_gap == 1.0 and max_gap == 1


def test_min_and_max_gap_span_two_reusers():
    # t0 root (w0). Reused by t1 (w1, child) and t2 (w2, grandchild). Both reuse
    # the shared system prefix -> min_gap 1, max_gap 2.
    sys = _m("system", "S")
    t0 = _ev("t0", 0, [], [sys, _m("user", "Q")])
    t1 = _ev("t1", 10, ["t0"], [sys, _m("user", "Q"), _m("assistant", "R1")])
    t2 = _ev("t2", 20, ["t1"], [sys, _m("user", "Q"), _m("assistant", "R1"),
                                _m("user", "Q2")])
    out = _forward_reuse_depths([t0, t1, t2], target="t0")
    segs, _covers = out["t0"]
    # shallowest boundary is reused by both t1 (gap1) and t2 (gap2).
    shallow = min(segs, key=lambda s: (s[0], s[1]))
    assert shallow[3] == 1.0   # min_gap
    assert shallow[4] == 2     # max_gap


def test_tiebreak_fraction_within_wave():
    # Two same-wave (root) cousins at different t_start distances share sys.
    # nearest ts_gap=1 -> min_gap 0.0 (0 + (1 - 1/1)).
    sys = _m("system", "S")
    t0 = _ev("t0", 0, [], [sys, _m("user", "A")])
    near = _ev("t1", 10, [], [sys, _m("user", "B")])   # index 1 -> ts_gap 1
    far = _ev("t2", 20, [], [sys, _m("user", "C")])    # index 2 -> ts_gap 2
    out = _forward_reuse_depths([t0, near, far], target="t0")
    segs, _covers = out["t0"]
    (bw, bp, breadth, min_gap, max_gap) = segs[0]
    # nearest reuser ts_gap 1 -> tiebreak fraction 0 -> min_gap exactly 0.0
    assert min_gap == 0.0
    assert breadth == 2


def test_min_gap_non_decreasing_with_depth():
    # Priority validator: deeper boundaries must have >= min_gap (=> <= priority).
    sys = _m("system", "S")
    t0 = _ev("t0", 0, [], [sys, _m("user", "DEEP")])
    shallow_reuser = _ev("t1", 10, ["t0"], [sys, _m("user", "OTHER")])  # w1, sys only
    deep_reuser = _ev("t2", 20, ["t1"], [sys, _m("user", "DEEP"),
                                         _m("assistant", "X")])  # w2, sys+DEEP
    out = _forward_reuse_depths([t0, shallow_reuser, deep_reuser], target="t0")
    segs, _covers = out["t0"]
    gaps = [s[3] for s in sorted(segs, key=lambda s: (s[0], s[1]))]
    assert gaps == sorted(gaps), f"min_gap must be non-decreasing with depth: {gaps}"


def test_inflight_only_boundary_has_none_gap():
    # A depth shared only by an in-flight turn: covered (breadth floor 1) but no
    # forward-future reuser -> min_gap/max_gap None.
    sys = _m("system", "S")
    t0 = _ev("t0", 0, [], [sys, _m("user", "DEEP")])
    inflight = _ev("t1", 10, [], [sys, _m("user", "DEEP")])
    out = _forward_reuse_depths(
        [t0, inflight], target="t0",
        completed_ids=set(), dispatched_ids={"t0", "t1"},  # t1 in-flight
    )
    segs, _covers = out["t0"]
    assert segs
    for (_bw, _bp, breadth, min_gap, max_gap) in segs:
        assert breadth == 1
        assert min_gap is None and max_gap is None
