# Copyright 2025 The Kubernetes Authors.
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

"""Retention policy implementations for KV-cache retention directives.

Policies compute per-turn retention_directives that instruct the inference
server to protect specific token ranges in its KV-cache with explicit
priorities and TTLs. The directives are injected into the request via
the extra_body mechanism in ChatCompletionAPIData.
"""

from typing import Any, Optional

from inference_perf.models import ReuseSegment


class WorkflowAwarePolicy:
    """DAG-aware retention (KVFlow, arXiv 2507.07400).

    The trace-replay datagen supplies a per-producer reuse-depth profile: a
    list of segments over the producer's prompt token range [start, end), each
    with:

      - breadth: how many later calls reuse up to `end`. Higher breadth ->
        higher priority tier (high/mid/low).
      - intervening_spans: calls running during the idle gap before the
        farthest reuse. TTL = intervening_spans * per_span_s + queue_margin_s
        + ttl_buffer_s.

    No profile (output never reused) -> no directive (immediate LRU evict).
    """

    def __init__(
        self,
        ttl_buffer_s: float = 5.0,
        high_breadth_priority: int = 90,
        mid_breadth_priority: int = 70,
        low_breadth_priority: int = 50,
        per_span_s: float = 9.0,
        queue_margin_s: float = 10.0,
    ) -> None:
        self.ttl_buffer_s = ttl_buffer_s
        self.high_breadth_priority = high_breadth_priority
        self.mid_breadth_priority = mid_breadth_priority
        self.low_breadth_priority = low_breadth_priority
        self.per_span_s = per_span_s
        self.queue_margin_s = queue_margin_s

    def compute_directives(
        self,
        reuse_depth_profile: Optional[list[ReuseSegment]],
        scope: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        profile = reuse_depth_profile
        # No profile → no directive (immediate LRU evict). Covers None and [].
        if not profile:
            return None
        directives: list[dict[str, Any]] = [
            {
                "start": seg.start,
                "end": seg.end,
                "priority": self._priority_for_breadth(seg.breadth),
                "duration": seg.intervening_spans * self.per_span_s
                + self.queue_margin_s + self.ttl_buffer_s,
            }
            for seg in profile
        ]
        result: dict[str, Any] = {"retention_directives": directives}
        if scope is not None:
            result["retention_scope"] = scope
        return result

    def _priority_for_breadth(self, breadth: int) -> int:
        # More future reuses (breadth) → higher priority tier.
        if breadth >= 4:
            return self.high_breadth_priority
        if breadth >= 2:
            return self.mid_breadth_priority
        return self.low_breadth_priority
