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

"""Unit tests for _lifecycle_sets — maps the qualified-keyed output registry and
the per-session dispatched set into raw-id (completed, dispatched) frozensets."""

from inference_perf.datagen.replay_graph_session_datagen import _lifecycle_sets


class _FakeRegistry:
    def __init__(self, recorded):
        self._recorded = list(recorded)  # qualified ids

    def get_event_ids(self):
        return list(self._recorded)


def test_completed_filtered_by_session_prefix():
    graph_events = {"a": object(), "b": object(), "c": object()}
    # 'S:a','S:b' recorded for THIS session; 'OTHER:c' is another session's
    # completion whose raw id 'c' exists in graph_events -> must NOT leak in.
    reg = _FakeRegistry(["S:a", "S:b", "OTHER:c"])
    completed, dispatched = _lifecycle_sets(graph_events, "S", reg, {"a"})
    assert completed == frozenset({"a", "b"})  # 'c' excluded (only OTHER recorded it)
    assert dispatched == frozenset({"a"})


def test_none_registry_gives_empty_completed():
    graph_events = {"a": object()}
    completed, dispatched = _lifecycle_sets(graph_events, "S", None, set())
    assert completed == frozenset()
    assert dispatched == frozenset()


def test_dispatched_is_a_snapshot_copy():
    graph_events = {"a": object(), "b": object()}
    live = {"a"}
    _completed, dispatched = _lifecycle_sets(graph_events, "S", None, live)
    live.add("b")  # mutate after the call
    assert dispatched == frozenset({"a"})  # snapshot unaffected
