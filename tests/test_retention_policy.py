"""Tests for KV-cache retention policy implementations."""

from inference_perf.models import ReuseSegment
from inference_perf.policies.retention import WorkflowAwarePolicy


class TestWorkflowAwareProfile:
    def test_layered_directive_from_profile(self):
        # TTL = intervening_spans*per_span_s + queue_margin_s + ttl_buffer_s
        p = WorkflowAwarePolicy(ttl_buffer_s=3.0, per_span_s=9.0, queue_margin_s=10.0)
        out = p.compute_directives([
            ReuseSegment(start=0, end=175, breadth=5, intervening_spans=4),
            ReuseSegment(start=175, end=400, breadth=2, intervening_spans=3),
        ], scope="s1")
        dirs = out["retention_directives"]
        assert out["retention_scope"] == "s1"
        assert [(d["start"], d["end"]) for d in dirs] == [(0, 175), (175, 400)]
        assert [d["priority"] for d in dirs] == [90, 70]
        assert dirs[0]["priority"] >= dirs[1]["priority"]
        assert dirs[0]["duration"] == 4 * 9.0 + 10.0 + 3.0   # 49s
        assert dirs[1]["duration"] == 3 * 9.0 + 10.0 + 3.0   # 40s

    def test_empty_profile_is_evict(self):
        p = WorkflowAwarePolicy()
        assert p.compute_directives([], scope="s1") is None

    def test_none_profile_is_evict(self):
        p = WorkflowAwarePolicy()
        assert p.compute_directives(None, scope="s1") is None

    def test_breadth_one_is_leaf_priority(self):
        p = WorkflowAwarePolicy()
        dirs = p.compute_directives(
            [ReuseSegment(start=0, end=50, breadth=1)], scope="s1"
        )["retention_directives"]
        assert dirs[0]["priority"] == 50
