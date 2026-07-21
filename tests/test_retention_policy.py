"""Tests for KV-cache retention policy implementations."""

from inference_perf.models import ReuseSegment
from inference_perf.policies.retention import WorkflowAwarePolicy


class TestWorkflowAwareProfile:
    def test_layered_directive_from_profile(self):
        # TTL = cold_gap*per_span_s + queue_margin_s + ttl_buffer_s
        p = WorkflowAwarePolicy(ttl_buffer_s=3.0, per_span_s=9.0, queue_margin_s=10.0)
        out = p.compute_directives([
            ReuseSegment(start=0, end=175, breadth=5, cold_gap=4),
            ReuseSegment(start=175, end=400, breadth=2, cold_gap=3),
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

    def test_single_reuse_is_low_priority(self):
        # breadth 1 (one future reuser) → low tier (50), regardless of cold_gap.
        p = WorkflowAwarePolicy()
        dirs = p.compute_directives(
            [ReuseSegment(start=0, end=50, breadth=1, cold_gap=4)], scope="s1"
        )["retention_directives"]
        assert dirs[0]["priority"] == 50

    def test_priority_ignores_cold_gap(self):
        # Priority is breadth-only; cold_gap changes the TTL, not the tier.
        p = WorkflowAwarePolicy()
        assert p._priority_for_breadth(2) == p._priority_for_breadth(2)
        assert p._priority_for_breadth(4) > p._priority_for_breadth(2)
        assert p._priority_for_breadth(2) > p._priority_for_breadth(1)


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


class TestDirectiveRescale:
    def _mk(self, actual_chars, recorded_chars, tools=None):
        from inference_perf.datagen.replay_graph_session_datagen import (
            EventOutputRegistry,
            SessionChatCompletionAPIData,
            WorkerSessionTracker,
        )
        from inference_perf.apis.chat import ChatMessage
        return SessionChatCompletionAPIData(
            event_id="s:e",
            registry=EventOutputRegistry(),
            worker_tracker=WorkerSessionTracker(),
            completion_queue=None,
            total_events_in_session=1,
            messages=[ChatMessage(role="user", content="x" * actual_chars)],
            original_messages=[{"role": "user", "content": "x" * recorded_chars}],
            tool_definitions=tools,
        )

    def test_boundaries_scale_to_materialized_size(self):
        # recorded est 1000 tok, materialized est 2000 tok -> boundaries double.
        d = self._mk(actual_chars=8000, recorded_chars=4000)
        out = d._rescale_profile_to_materialized(
            [ReuseSegment(start=0, end=500, breadth=2, cold_gap=1)]
        )
        assert (out[0].start, out[0].end) == (0, 1000)

    def test_tools_preamble_shifts_ranges(self):
        tools = [{"name": "t", "description": "d" * 400}]
        d = self._mk(actual_chars=4000, recorded_chars=4000, tools=tools)
        out = d._rescale_profile_to_materialized(
            [ReuseSegment(start=0, end=500, breadth=1, cold_gap=0)]
        )
        assert out[0].start > 0 and out[0].end > 500  # shifted by tools estimate

    def test_noop_when_sizes_match_and_no_tools(self):
        d = self._mk(actual_chars=4000, recorded_chars=4000)
        prof = [ReuseSegment(start=0, end=500, breadth=1, cold_gap=0)]
        assert d._rescale_profile_to_materialized(prof) is prof


class TestMinBreadthFilter:
    def test_below_threshold_emits_none(self):
        p = WorkflowAwarePolicy(min_breadth=8)
        out = p.compute_directives(
            [ReuseSegment(start=0, end=100, breadth=3, cold_gap=1)], scope="s1"
        )
        assert out is None

    def test_at_threshold_emits(self):
        p = WorkflowAwarePolicy(min_breadth=8)
        out = p.compute_directives(
            [ReuseSegment(start=0, end=100, breadth=8, cold_gap=1)], scope="s1"
        )
        assert out is not None and len(out["retention_directives"]) == 1

    def test_default_off(self):
        p = WorkflowAwarePolicy()
        out = p.compute_directives(
            [ReuseSegment(start=0, end=100, breadth=1, cold_gap=1)], scope="s1"
        )
        assert out is not None


class TestMinRemainingReuse:
    def test_suppressed_below_threshold(self):
        p = WorkflowAwarePolicy(min_remaining_reuse=8)
        out = p.compute_directives(
            [ReuseSegment(start=0, end=100, breadth=2, cold_gap=1)],
            scope="s1", remaining_reuse=5,
        )
        assert out is None

    def test_emitted_at_threshold(self):
        p = WorkflowAwarePolicy(min_remaining_reuse=8)
        out = p.compute_directives(
            [ReuseSegment(start=0, end=100, breadth=2, cold_gap=1)],
            scope="s1", remaining_reuse=8,
        )
        assert out is not None and out["retention_directives"][0]["start"] == 0

    def test_off_when_zero(self):
        p = WorkflowAwarePolicy(min_remaining_reuse=0)
        out = p.compute_directives(
            [ReuseSegment(start=0, end=100, breadth=2, cold_gap=1)],
            scope="s1", remaining_reuse=0,
        )
        assert out is not None
