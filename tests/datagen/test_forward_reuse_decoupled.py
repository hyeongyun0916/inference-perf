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

"""Unit tests for lifecycle-decoupled coverage in _forward_reuse_depths.

breadth (priority tier + TTL) counts FUTURE (not-yet-dispatched) turns only;
a depth reused only by an IN-FLIGHT (dispatched, not-completed) turn gets a
floor breadth of 1 (keeps coverage, low tier); COMPLETED turns count for
neither. registry=None makes _live_msgs return ev.call.messages verbatim.
"""

from types import SimpleNamespace

from inference_perf.datagen.replay_graph_session_datagen import (
    _forward_reuse_depths,
)


def _m(role, content):
    return {"role": role, "content": content}


def _ev(event_id, t, preds, messages):
    return SimpleNamespace(
        event_id=event_id,
        t_start_ms=t,
        predecessor_event_ids=list(preds),
        call=SimpleNamespace(messages=messages),
    )


def _breadths(segs):
    return [b for (_w, _p, b, *_rest) in segs]


# Shared prompt prefix: all turns share message[0]; deeper sharing varies.
def _session():
    sys = _m("system", "SYSPROMPT")
    # target t0; three others share the system prefix at least.
    t0 = _ev("t0", 0, [], [sys, _m("user", "A0")])
    t1 = _ev("t1", 10, [], [sys, _m("user", "A1")])       # shares depth 1 (sys)
    t2 = _ev("t2", 20, [], [sys, _m("user", "A2")])       # shares depth 1 (sys)
    t3 = _ev("t3", 30, [], [sys, _m("user", "A0X")])  # shares sys+partial msg1
    return t0, t1, t2, t3


def test_future_counts_toward_breadth():
    # No lifecycle sets given -> everyone is 'future' -> legacy behavior.
    t0, t1, t2, t3 = _session()
    out = _forward_reuse_depths([t0, t1, t2, t3], target="t0")
    segs, _covers = out["t0"]
    # At least the shallow (whole system message) boundary counted for 3 others.
    assert max(_breadths(segs)) == 3


def test_completed_excluded_from_breadth():
    # t1,t2 completed -> excluded; only t3 (future) shares -> breadth drops.
    t0, t1, t2, t3 = _session()
    out = _forward_reuse_depths(
        [t0, t1, t2, t3], target="t0",
        completed_ids={"t1", "t2"}, dispatched_ids={"t0", "t1", "t2"},
    )
    segs, _covers = out["t0"]
    # t3 is future (not in dispatched) and shares the system prefix -> breadth 1.
    assert max(_breadths(segs)) == 1


def test_inflight_gets_floor_not_tier_bump():
    # t1 future (breadth), t2 in-flight (dispatched, not completed) -> t2 must
    # NOT raise the tier; it only holds coverage at floor 1.
    t0, t1, t2, t3 = _session()
    out = _forward_reuse_depths(
        [t0, t1, t2, t3], target="t0",
        completed_ids=set(), dispatched_ids={"t0", "t2"},
    )
    segs, _covers = out["t0"]
    # future sharers of the system prefix = t1, t3 -> breadth 2 (t2 excluded
    # from the count). No boundary exceeds 2 despite t2 also sharing.
    assert max(_breadths(segs)) == 2


def test_inflight_only_depth_is_covered_at_floor():
    # A deep block shared ONLY by an in-flight turn must still be covered
    # (floor breadth 1) so owner-clear does not fire.
    sys = _m("system", "SYSPROMPT")
    t0 = _ev("t0", 0, [], [sys, _m("user", "DEEPSHARE")])
    inflight = _ev("t1", 10, [], [sys, _m("user", "DEEPSHARE")])  # shares depth 2
    out = _forward_reuse_depths(
        [t0, inflight], target="t0",
        completed_ids=set(), dispatched_ids={"t0", "t1"},  # t1 in-flight
    )
    segs, _covers = out["t0"]
    # t1 is in-flight -> future breadth 0 everywhere, but the shared depth is
    # still emitted at floor breadth 1 (coverage held).
    assert segs, "in-flight-only shared depth must still be covered"
    assert set(_breadths(segs)) == {1}


def test_backward_future_cousin_is_counted():
    # A turn earlier in t_start but NOT yet dispatched is 'future' -> counted
    # (closes the original cross-branch coverage gap; uses dispatch, not t_start).
    sys = _m("system", "SYSPROMPT")
    target = _ev("t5", 50, [], [sys, _m("user", "Q")])
    cousin = _ev("t2", 20, [], [sys, _m("user", "Q")])  # earlier t_start, shares deep
    out = _forward_reuse_depths(
        [cousin, target], target="t5",
        completed_ids=set(), dispatched_ids={"t5"},  # cousin NOT dispatched -> future
    )
    segs, _covers = out["t5"]
    assert max(_breadths(segs)) == 1


def test_ancestor_excluded():
    sys = _m("system", "SYSPROMPT")
    anc = _ev("t0", 0, [], [sys, _m("user", "Q")])
    child = _ev("t1", 10, ["t0"], [sys, _m("user", "Q"), _m("assistant", "R")])
    out = _forward_reuse_depths([anc, child], target="t1")
    segs, _covers = out["t1"]
    # Only ancestor exists as a comparison turn; it is excluded -> no segments.
    assert segs == ()


def test_missing_predecessors_no_crash():
    sys = _m("system", "SYSPROMPT")
    a = SimpleNamespace(event_id="a", t_start_ms=0, call=SimpleNamespace(
        messages=[sys, _m("user", "X")]))  # no predecessor_event_ids attr
    b = SimpleNamespace(event_id="b", t_start_ms=10, call=SimpleNamespace(
        messages=[sys, _m("user", "Y")]))
    out = _forward_reuse_depths([a, b], target="a")
    assert "a" in out  # forward-only, no crash


def test_non_increasing_breadth_with_mixed_lifecycle():
    sys = _m("system", "SYSPROMPT")
    # target shares sys with two futures; a deeper block only with an in-flight.
    t0 = _ev("t0", 0, [], [sys, _m("user", "DEEP")])
    f1 = _ev("f1", 10, [], [sys, _m("user", "B1")])   # shares sys (shallow)
    f2 = _ev("f2", 20, [], [sys, _m("user", "B2")])   # shares sys (shallow)
    inf = _ev("g1", 30, [], [sys, _m("user", "DEEP")])  # shares sys + deep msg[1]
    out = _forward_reuse_depths(
        [t0, f1, f2, inf], target="t0",
        completed_ids=set(), dispatched_ids={"t0", "g1"},  # g1 in-flight
    )
    segs, _covers = out["t0"]
    bs = _breadths(segs)
    assert bs == sorted(bs, reverse=True), f"breadth must be non-increasing: {bs}"
    assert bs[0] == 2 and bs[-1] == 1  # shallow future=2, deep in-flight floor=1
