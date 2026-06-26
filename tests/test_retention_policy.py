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


class TestBuildFromConfig:
    def test_per_span_and_queue_margin_propagate(self):
        # per_span_s / queue_margin_s set in config must reach the policy so the
        # oracle TTL can be matched to the workload's actual per-span wall-clock.
        from inference_perf.config.datagen.replay import RetentionPolicyConfig
        from inference_perf.main import _build_retention_policy

        cfg = RetentionPolicyConfig(
            type="workflow_aware",
            per_span_s=18.0,
            queue_margin_s=12.0,
            ttl_buffer_s=7.0,
        )
        p = _build_retention_policy(cfg)
        assert p.per_span_s == 18.0
        assert p.queue_margin_s == 12.0
        assert p.ttl_buffer_s == 7.0

    def test_defaults_unchanged_when_omitted(self):
        from inference_perf.config.datagen.replay import RetentionPolicyConfig
        from inference_perf.main import _build_retention_policy

        p = _build_retention_policy(RetentionPolicyConfig())
        assert p.per_span_s == 9.0
        assert p.queue_margin_s == 10.0
        assert p.ttl_buffer_s == 5.0
