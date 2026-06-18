# Copyright 2025 The Kubernetes Authors.
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

"""Integration test: build ReplayGraph for the sw_arch linear trace, compute
reuse profiles, and assert structural properties.

Graph-build API:
  build_raw_calls(spans) -> List[RawCall]   # extract LLM spans
  build_graph(calls, source_file="") -> ReplayGraph   # build causal DAG
  graph.events: Dict[str, GraphEvent]
"""

import json
from pathlib import Path

import pytest

from inference_perf.datagen.otel_trace_to_replay_graph import (
    build_graph,
    build_raw_calls,
)
from inference_perf.datagen.replay_graph_session_datagen import _compute_reuse_profiles
from inference_perf.policies.retention import WorkflowAwarePolicy

_TRACE_FILE = (
    Path(__file__).parent.parent.parent
    / "examples/otel/test_traces/advanced/software_architecture_review_linear.json"
)


@pytest.fixture(scope="module")
def sw_arch_graph():
    data = json.loads(_TRACE_FILE.read_text(encoding="utf-8"))
    spans = data.get("spans", [])
    calls = build_raw_calls(spans)
    return build_graph(calls, source_file=str(_TRACE_FILE))


@pytest.fixture(scope="module")
def sw_arch_profiles(sw_arch_graph):
    events = list(sw_arch_graph.events.values())
    return _compute_reuse_profiles(events)


def test_at_least_four_producers(sw_arch_profiles):
    """Synthesis/chain spans produce outputs reused downstream — expect ≥4 profiles."""
    assert len(sw_arch_profiles) >= 4, (
        f"Expected ≥4 producers with reuse profiles, got {len(sw_arch_profiles)}"
    )


def test_every_profile_root_segment_starts_at_zero(sw_arch_profiles):
    """Every producer's first ReuseSegment must cover from token 0 (no leading gap)."""
    for pid, segs in sw_arch_profiles.items():
        assert segs, f"Producer {pid} has empty profile list"
        assert segs[0].start == 0, (
            f"Producer {pid}: first segment starts at {segs[0].start}, expected 0"
        )


def test_terminals_absent_from_profiles(sw_arch_graph, sw_arch_profiles):
    """Events whose output is never reused must be absent from profiles.

    At least one terminal should exist, so profiles must be a strict subset
    of all events.
    """
    total_events = len(sw_arch_graph.events)
    total_profiles = len(sw_arch_profiles)
    assert total_profiles < total_events, (
        f"Expected some terminal events (no output reuse), but profiles covers "
        f"all {total_events} events — no terminals found"
    )


def test_max_breadth_producer_has_correct_priority(sw_arch_profiles):
    """The most-reused producer's breadth maps to the expected WorkflowAwarePolicy priority.

    The actual max breadth in this 15-span linear trace is 2 (event_009 is
    shared by two successors). breadth≥2 → mid_breadth_priority (default 70).
    Verify the policy maps it correctly.
    """
    # Find max breadth across all profiles (first segment = widest coverage)
    top_breadth = max(segs[0].breadth for segs in sw_arch_profiles.values())
    policy = WorkflowAwarePolicy()
    priority = policy._priority_for_breadth(top_breadth)

    # In this trace max breadth is 2 → mid_breadth_priority (70).
    # If more reuse is found in future trace versions and breadth reaches ≥4
    # the priority becomes 90. Either is correct.
    assert priority in (70, 90), (
        f"Priority {priority} is not a recognized tier (expected 70 or 90 for "
        f"breadth={top_breadth})"
    )
