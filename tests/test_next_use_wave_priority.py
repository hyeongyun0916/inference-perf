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

"""Unit tests for next_use_wave priority/TTL mapping in WorkflowAwarePolicy."""

from inference_perf.policies import WorkflowAwarePolicy


def _pol(**kw):
    return WorkflowAwarePolicy(
        priority_mode="next_use_wave", k_wave=25, per_wave_s=10.0,
        queue_margin_s=60.0, ttl_buffer_s=10.0, **kw,
    )


def test_priority_gap_zero_is_max():
    assert _pol()._priority_for_wave(0.0) == 100


def test_priority_decreases_with_gap():
    p = _pol()
    assert p._priority_for_wave(1.0) == 75   # 100 - 25*1
    assert p._priority_for_wave(2.0) == 50
    assert p._priority_for_wave(1.0) > p._priority_for_wave(2.0)


def test_priority_clamped_to_floor():
    # gap 5 -> 100 - 125 = -25 -> clamps to 1
    assert _pol()._priority_for_wave(5.0) == 1


def test_priority_none_is_floor():
    # No forward-future reuser (covered only) -> evict-first floor.
    assert _pol()._priority_for_wave(None) == 1


def test_priority_tiebreak_fraction_lowers_within_bucket():
    p = _pol()
    # min_gap carries the +(1 - 1/ts_gap) tiebreak already; a larger fraction
    # (farther nearest reuser in t_start) yields a lower priority, but stays
    # above the next integer-wave bucket.
    near = p._priority_for_wave(1.0)        # ts_gap == 1 -> fraction 0
    far = p._priority_for_wave(1.5)         # fraction 0.5
    assert near == 75 and far == round(100 - 25 * 1.5) == 62
    assert far < near


def test_ttl_uses_max_gap():
    p = _pol()  # margin 60 + buffer 10 = 70 base; per_wave_s 10
    assert p._ttl_for_wave(3) == 3 * 10.0 + 70.0
    assert p._ttl_for_wave(0) == 70.0


def test_ttl_none_is_base_floor():
    assert _pol()._ttl_for_wave(None) == 70.0
