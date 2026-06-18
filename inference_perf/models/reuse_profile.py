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

from pydantic import BaseModel, Field


class ReuseSegment(BaseModel):
    """One step of a producing call's reuse-depth profile.

    Over the producer's own token positions [start, end): how many later
    calls reuse a prefix reaching at least `end` tokens (`breadth`), and how
    many calls run during the idle gap before the farthest such reuse
    (`intervening_spans`, the TTL basis). breadth is non-increasing with depth,
    so directives derived from it satisfy the prefix-cache
    non-increasing-priority constraint automatically.
    """
    start: int = Field(..., ge=0, description="Token start position (inclusive)")
    end: int = Field(..., ge=0, description="Token end position (exclusive)")
    breadth: int = Field(..., ge=1, description="Number of future calls reusing up to `end`")
    intervening_spans: int = Field(default=0, ge=0, description="Calls running during the idle gap before the FARTHEST reuse of this region; TTL basis (real idle ~= intervening_spans * per-span latency)")
