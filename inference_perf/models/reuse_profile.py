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

"""Reuse-depth profile model for workflow-aware KV-cache retention."""

from typing import Optional

from pydantic import BaseModel, Field


class ReuseSegment(BaseModel):
    """One step of a producing call's reuse-depth profile.

    Over the producer's own token positions [start, end): how many later
    calls reuse a prefix reaching at least `end` tokens (`breadth`), and the
    longest run of intervening calls this region goes untouched between
    consecutive reuses (`cold_gap`, the TTL basis — the region must survive at
    least this long to still be cached at its next reuse). breadth is
    non-increasing with depth, so directives derived from it satisfy the
    prefix-cache non-increasing-priority constraint automatically.
    """
    start: int = Field(..., ge=0, description="Token start position (inclusive)")
    end: int = Field(..., ge=0, description="Token end position (exclusive)")
    breadth: int = Field(..., ge=1, description="Number of future calls reusing up to `end`")
    cold_gap: int = Field(default=0, ge=0, description="Max consecutive intervening calls this region goes untouched between reuses; TTL basis (a recency-hot region has cold_gap ~= 0 and needs no retention)")
    # start/end are RECORDED-token estimates; end_msg structurally anchors
    # `end` (producer message count) so render calibration can resolve it to
    # exact materialized-token coordinates.
    end_msg: Optional[int] = Field(default=None, ge=0, description="Producer message count that `end` corresponds to (None = unanchored)")
    covers_output: bool = Field(default=False, description="Covers the producer's generated output; start/end are ignored and the server protects the post-prompt region it generates.")
