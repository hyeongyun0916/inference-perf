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

"""Unit tests for non-ancestor coverage in _forward_reuse_depths.

The comparison set for a turn is every NON-ANCESTOR turn (all later turns plus
backward cross-branch turns), excluding only its own lineage. A cross-branch
cousin that caches a shared prefix must be protected even when the turn has no
forward reuse, otherwise a same-scope turn owner-clears the owner's protection.
registry=None makes _live_msgs return ev.call.messages verbatim, so these tests
are pure content.
"""

from types import SimpleNamespace

from inference_perf.datagen.replay_graph_session_datagen import _forward_reuse_depths


def _m(role, content):
    return {"role": role, "content": content}


def _ev(event_id, t, preds, messages):
    return SimpleNamespace(
        event_id=event_id,
        t_start_ms=t,
        predecessor_event_ids=list(preds),
        call=SimpleNamespace(messages=messages),
    )


def test_forward_reuse_basic():
    """Linear chain: forward turns reuse the shared prefix; breadth counts them."""
    sys = _m("system", "S" * 400)
    q0, a0, q1 = _m("user", "q0"), _m("assistant", "a0"), _m("user", "q1")
    a1, q2 = _m("assistant", "a1"), _m("user", "q2")
    e0 = _ev("e0", 0, [], [sys, q0])
    e1 = _ev("e1", 10, ["e0"], [sys, q0, a0, q1])
    e2 = _ev("e2", 20, ["e1"], [sys, q0, a0, q1, a1, q2])

    fd = _forward_reuse_depths([e0, e1, e2], registry=None)

    # e0 reused by e1 and e2 at depth 2 -> breadth 2; e1 reused by e2 at depth 4.
    assert fd["e0"] == (((2, 0, 2),), True), fd["e0"]
    assert fd["e1"] == (((4, 0, 1),), True), fd["e1"]


def test_backward_cousin_covered():
    """Leaf whose forward turn diverges at message[0] still gets coverage from an
    earlier cross-branch cousin (a non-ancestor that shares the prefix)."""
    sys, u1, a1 = _m("system", "S" * 400), _m("user", "u1"), _m("assistant", "a1")
    e1 = _ev("e1", 0, ["e0"], [sys, u1, a1])
    e2 = _ev("e2", 10, ["e1"], [sys, u1, a1, _m("user", "AAA"), _m("assistant", "ra")])
    e3 = _ev("e3", 20, ["e1"], [sys, u1, a1, _m("user", "BBB"), _m("assistant", "rb")])
    # e3's only later turn starts a different sub-conversation (diverges at msg 0)
    e4 = _ev("e4", 30, ["e3"], [_m("user", "different first msg"), _m("assistant", "x")])

    fd = _forward_reuse_depths([e1, e2, e3, e4], registry=None)

    segs, covers = fd["e3"]
    assert segs, f"e3 (leaf) should be covered by backward cousin e2, got {segs}"
    assert segs[0][0] == 3, f"expected 3 whole-message shared prefix, got {segs}"
    assert covers is False, "an earlier cousin never marks covers_output"


def test_partial_overlap_cousin_is_counted():
    """A partial-overlap turn ({p,r}) is NOT an ancestor of {p,q}, so it IS in the
    comparison set — breadth reflects BOTH cross-branch cousins, not just an
    identical-predecessor sibling."""
    sys, u, a = _m("system", "S" * 400), _m("user", "u"), _m("assistant", "a")
    base = [sys, u, a]
    e_a = _ev("e_a", 0, ["p", "q"], base + [_m("user", "AA")])
    e_c = _ev("e_c", 5, ["p", "r"], base + [_m("user", "CC")])  # partial overlap
    e_b = _ev("e_b", 10, ["q", "p"], base + [_m("user", "BB")])  # {p,q}, order-free

    fd = _forward_reuse_depths([e_a, e_c, e_b], registry=None)

    segs, _ = fd["e_b"]
    assert segs and segs[0][0] == 3, segs
    assert segs[0][2] == 2, f"breadth 2 (both cousins e_a and e_c), got {segs}"


def test_ancestor_excluded():
    """A turn does NOT compare against its own ancestor: e1 descends from e0 and
    shares e0's full prompt, but e0 is e1's ancestor so it is excluded; with no
    other (non-ancestor) turn, e1 has no comparison target and stays empty. (An
    all-backward comparison would instead count e0 here.)"""
    sys, u, a1, u2, a2 = (
        _m("system", "S" * 400), _m("user", "u"), _m("assistant", "a1"),
        _m("user", "u2"), _m("assistant", "a2"),
    )
    e0 = _ev("e0", 0, [], [sys, u, a1, u2, a2])
    e1 = _ev("e1", 10, ["e0"], [sys, u, a1, u2, a2, _m("user", "u3")])

    fd = _forward_reuse_depths([e0, e1], registry=None)

    # e0 is reused forward by e1 (depth 5, full prompt) and covers its output.
    assert fd["e0"] == (((5, 0, 1),), True), fd["e0"]
    # e1's only companion is its ancestor e0 -> excluded -> no segments.
    assert fd["e1"] == ((), False), fd["e1"]


def test_missing_predecessors_no_crash():
    """Events without predecessor_event_ids don't crash; forward reuse still works."""
    sys = _m("system", "S" * 400)
    e0 = SimpleNamespace(event_id="e0", t_start_ms=0,
                         call=SimpleNamespace(messages=[sys, _m("user", "q0")]))
    e1 = SimpleNamespace(event_id="e1", t_start_ms=10,
                         call=SimpleNamespace(messages=[sys, _m("user", "q0"),
                                                        _m("assistant", "a0"),
                                                        _m("user", "q1")]))
    fd = _forward_reuse_depths([e0, e1], registry=None)
    assert fd["e0"] == (((2, 0, 1),), True), fd["e0"]
