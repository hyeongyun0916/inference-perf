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
      - cold_gap: longest run of intervening calls the region goes untouched
        between consecutive reuses. TTL = cold_gap * per_span_s + queue_margin_s
        + ttl_buffer_s, so a recency-hot region (cold_gap ~= 0) gets only the
        margins and is not over-retained.

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
        min_breadth: int = 0,
        min_remaining_reuse: int = 0,
        render_url: str | None = None,
    ) -> None:
        self.ttl_buffer_s = ttl_buffer_s
        self.high_breadth_priority = high_breadth_priority
        self.mid_breadth_priority = mid_breadth_priority
        self.low_breadth_priority = low_breadth_priority
        self.per_span_s = per_span_s
        self.queue_margin_s = queue_margin_s
        # Emission gate: skip directives when the producer's best reuse breadth
        # is below this (0 = off) — few reusers do not justify a budget slot.
        self.min_breadth = min_breadth
        # Remaining-reuse gate: skip when this prompt is reused fewer than
        # min_remaining_reuse more times downstream (0 = off). Near a session's
        # end remaining reuse drops and protection stops paying; the client
        # knows this from the DAG, the server does not.
        self.min_remaining_reuse = min_remaining_reuse
        # Exact-coordinate calibration: base URL of a vLLM server exposing
        # /v1/chat/completions/render. When set, the datagen resolves each
        # directive boundary to exact materialized token positions before
        # compute_directives is called (None = legacy char-ratio rescale).
        self.render_url = render_url

    def compute_directives(
        self,
        reuse_depth_profile: Optional[list[ReuseSegment]],
        scope: Optional[str] = None,
        remaining_reuse: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        profile = reuse_depth_profile
        # No profile → no directive (immediate LRU evict). Covers None and [].
        if not profile:
            return None
        # Emission gates: skip protection that would not pay for its budget slot.
        if self.min_breadth and max(seg.breadth for seg in profile) < self.min_breadth:
            return None
        if (self.min_remaining_reuse and remaining_reuse is not None
                and remaining_reuse < self.min_remaining_reuse):
            return None
        directives: list[dict[str, Any]] = [
            {
                "start": seg.start,
                "end": seg.end,
                "priority": self._priority_for_breadth(seg.breadth),
                "duration": seg.cold_gap * self.per_span_s
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
