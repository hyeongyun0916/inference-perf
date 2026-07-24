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

"""Shared graph-backed SessionGenerator runtime.

This module contains the session replay runtime that is agnostic to how a
ReplayGraph was produced. Concrete generators are responsible for producing
ReplaySession objects; this base class handles session lifecycle, worker
affinity, lazy request materialization, and session completion tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, replace as dc_replace
from multiprocessing.managers import SyncManager
from typing import Any, Dict, List, Optional, Set, Tuple

from aiohttp import ClientResponse

from inference_perf.apis import (
    ChatCompletionAPIData,
    ErrorResponseInfo,
    InferenceInfo,
    LazyLoadInferenceAPIData,
    SessionLifecycleMetric,
    StreamedResponseMetrics,
    UnaryResponseMetrics,
)
from inference_perf.apis.chat import ChatMessage
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.apis.streaming_parser import parse_sse_stream
from inference_perf.config import APIConfig, APIType, DataConfig, SessionReplayConfig
from inference_perf.config.datagen.replay import BadToolCallHandling
from inference_perf.datagen.base import LazyLoadDataMixin, SessionGenerator
from inference_perf.datagen.replay_graph_types import InputSegment, ReplayGraph
from inference_perf.models import ReuseSegment
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)


class SessionReplayLazyLoadData(LazyLoadInferenceAPIData):
    """LazyLoadInferenceAPIData extended with per-session addressing for OTel trace replay.

    Instead of a global data_index into all_events, the worker identifies the
    request by (session_index, local_event_index) so it can build only the
    requested session's graph on demand without materializing all sessions upfront.
    """

    data_index: int = -1
    session_index: int
    local_event_index: int


# --- bad_tool_call_handling ------------------------------------------------
# Server-side tool-call parsers can emit malformed JSON in
# tool_calls[i].function.arguments — for example vLLM's `qwen3_xml` parser
# leaks closing XML markers (`</parameter></function>`) into the JSON string
# value at decode time. vLLM still returns 200 on the response, but on the
# *next* turn the chat template's `json.loads(arguments)` raises and vLLM
# returns HTTP 400. Replaying the bad bytes verbatim therefore halts the
# session.
#
# This mitigation lives ENTIRELY in the substitution path
# (`_build_messages_with_substitution`). The response path is byte-identical
# to upstream main: it stores the raw tool_calls in the registry exactly as
# the model emitted them. At substitution time, when a downstream event
# pulls a predecessor's stored message, we run `_detect_bad_tool_calls` on
# its `tool_calls` and, if `bad_tool_call_handling=use_recorded`, substitute
# the recorded assistant message at this slot.
#
# Gated per-event by the `bad_tool_call_handling` field on the OTel replay
# config. When the value is `none` (the default), the inline detection is
# short-circuited and behavior is identical to upstream main.


def _detect_bad_tool_calls(
    tool_calls: Optional[List[Dict[str, Any]]],
) -> List[Tuple[int, str, str]]:
    """Return [(index, function_name, json_error)] for tool_calls whose
    `arguments` field is a string that fails json.loads(). Empty list = ok."""
    bad: List[Tuple[int, str, str]] = []
    if not tool_calls:
        return bad
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError as e:
            bad.append((i, fn.get("name", "?"), str(e)))
    return bad


# --- end bad_tool_call_handling --------------------------------------------
def _prefix_reuse_counts(events) -> Dict[str, int]:
    """Per event: number of consecutive later events (time-ordered) that reuse
    this event's full prompt as a leading shared/output prefix, stopping at the
    first that does not. Matches the simulator's ``fwd_reuse`` (prefix-containment
    forward count) so sim and ip gate on an identical signal."""
    order = sorted(events, key=lambda e: float(getattr(e, "t_start_ms", 0) or 0))

    def _lens(ev) -> Tuple[int, int]:
        segs = getattr(ev.call, "input_segments", []) or []
        total = sum(seg.token_count for seg in segs)
        leading = 0
        for seg in segs:
            if seg.type == "unique":
                break  # prefix-cache contiguity ends at the consumer's own content
            leading += seg.token_count
        return total, leading

    lens = [_lens(e) for e in order]
    out: Dict[str, int] = {}
    for i, ev in enumerate(order):
        prompt_len_i = lens[i][0]
        cnt = 0
        if prompt_len_i > 0:
            for j in range(i + 1, len(order)):
                if lens[j][1] >= prompt_len_i:  # event j reuses >= i's full prompt
                    cnt += 1
                else:
                    break
        out[ev.event_id] = cnt
    return out


def _live_msgs(ev, registry):
    """ev's messages with RECORDED assistant outputs replaced by LIVE outputs
    from the registry, so forward-reuse depth reflects what is actually shared
    at serve time (the server caches the live conversation, not the trace).
    Falls back to the recorded message when a live output isn't available yet
    (future/own turns)."""
    msgs = ev.call.messages
    if registry is None:
        return msgs
    segs = getattr(ev.call, "input_segments", None)
    if not segs:
        return msgs
    out_msgs = []
    cursor = 0
    for seg in segs:
        seg_msgs = msgs[cursor:cursor + seg.message_count]
        sid = getattr(seg, "source_event_id", None)
        if getattr(seg, "type", None) == "output" and seg.message_count == 1 and sid:
            live = registry.get_message_by_event_id(sid)
            if live is not None:
                out_msgs.append(live)
            else:
                txt = registry.get_output_by_event_id(sid)
                if txt is not None and seg_msgs:
                    m = dict(seg_msgs[0])
                    m["content"] = txt
                    out_msgs.append(m)
                else:
                    out_msgs.extend(seg_msgs)
        else:
            out_msgs.extend(seg_msgs)
        cursor += seg.message_count
    if cursor < len(msgs):
        out_msgs.extend(msgs[cursor:])
    return out_msgs


def _forward_reuse_depths(
    events, registry=None, target=None, completed_ids=None, dispatched_ids=None
) -> Dict[str, Tuple[tuple, bool]]:
    """Per event, the reuse-depth PROFILE + covers_output, TOKEN-granular.

    Compares recorded message content directly (bypassing the segment
    source_event_id attribution, which under-detects).
    The comparison set is every non-ancestor turn, split by lifecycle:
    COMPLETED turns (event_id in completed_ids) are skipped — they will not
    reuse anything again. FUTURE turns (event_id not in dispatched_ids, or
    dispatched_ids is None) count toward BREADTH (priority tier + TTL) and
    coverage. IN-FLIGHT turns (dispatched but not completed) hold COVERAGE only:
    a depth reused solely by in-flight turns is emitted at a FLOOR breadth of 1
    (low priority tier, minimal TTL) so a same-scope turn cannot owner-clear a
    block an in-flight (e.g. preempted, re-prefilling) turn will hit again,
    without inflating the tier. This keeps breadth = genuine future reuse
    (no over-protection) while coverage tracks anything still in use.
    Returns (segments, covers_output) where segments is a tuple
    of (shared_msgs, partial_chars, breadth, min_gap, max_gap) ascending by
    depth:
      - shared_msgs / partial_chars: a depth boundary — whole leading messages
        plus a char-prefix of the first diverging message.
      - breadth: how many comparison turns reuse this event's prompt AT LEAST
        that deep. Shallower prefixes have higher breadth (more reusers), so
        they get a higher priority tier.
      - min_gap / max_gap: min/max causal wave-gap (wave = topological level,
        root=0) over FORWARD-future reusers reaching this boundary; min_gap
        adds a (1 - 1/ts_gap) within-wave t_start tiebreak. Both None when no
        forward-future reuser reaches this boundary (coverage-floor only).
      - covers_output: a LATER event reuses this event's FULL prompt and extends
        past it, so this event's generated output is reused too. Backward
        siblings never set this (an earlier turn cannot consume this output).
    Reuse is prefix-contiguous, so [messages[:shared_msgs] + first partial_chars
    of the next message] is exactly a reused prefix; the client renders each
    boundary and LCPs it against the full render to get the exact TOKEN depth."""
    order = sorted(events, key=lambda e: float(getattr(e, "t_start_ms", 0) or 0))

    def _txt(m: Dict[str, Any]) -> str:
        c = m.get("content")
        if isinstance(c, str):
            return c
        return json.dumps(c, sort_keys=True, default=str) if c is not None else ""

    def _key(m: Dict[str, Any]) -> tuple:
        tc = m.get("tool_calls")
        return (m.get("role"), _txt(m), json.dumps(tc, sort_keys=True, default=str) if tc else "")

    msgs = [_live_msgs(ev, registry) for ev in order]
    keys = [[_key(m) for m in ms] for ms in msgs]

    def _boundary(i: int, j: int) -> Tuple[int, int]:
        a, ka, na = msgs[i], keys[i], len(msgs[i])
        b, kb = msgs[j], keys[j]
        w = 0
        for x, y in zip(ka, kb, strict=False):
            if x == y:
                w += 1
            else:
                break
        p = 0
        if w < na and w < len(b) and a[w].get("role") == b[w].get("role"):
            ca, cb = _txt(a[w]), _txt(b[w])
            lim = min(len(ca), len(cb))
            while p < lim and ca[p] == cb[p]:
                p += 1
        return w, p

    # Comparison set = every NON-ANCESTOR turn (forward turns + backward
    # cross-branch turns), excluding only this turn's own lineage. Ancestors are
    # upstream and share only a shallow prefix; cross-branch cousins can share a
    # deep prefix the causal graph does not link, and leaving that uncovered lets
    # a same-scope turn owner-clear the owner's protection.
    _eid2idx = {getattr(ev, "event_id", None): j for j, ev in enumerate(order)}
    _idx_preds = [
        [_eid2idx[p] for p in (getattr(ev, "predecessor_event_ids", ()) or ()) if p in _eid2idx]
        for ev in order
    ]

    def _ancestors(i: int) -> set:
        seen: set = set()
        stack = list(_idx_preds[i])
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(_idx_preds[x])
        return seen

    _wave_memo: Dict[int, int] = {}

    def _wave(idx: int) -> int:
        # Causal level: root=0, else 1 + max over predecessors (AND readiness).
        if idx in _wave_memo:
            return _wave_memo[idx]
        ps = _idx_preds[idx]
        _wave_memo[idx] = 0 if not ps else 1 + max(_wave(p) for p in ps)
        return _wave_memo[idx]

    out: Dict[str, Tuple[tuple, bool]] = {}
    completed = completed_ids or frozenset()
    for i, ev in enumerate(order):
        if target is not None and ev.event_id != target:
            continue
        na = len(msgs[i])
        depths_future: List[Tuple[int, int]] = []
        depths_inflight: List[Tuple[int, int]] = []
        # Forward-future reusers only: (w, p, wave_gap, ts_gap) for wave stats.
        fut_wave: List[Tuple[int, int, int, int]] = []
        covers = False
        anc = _ancestors(i)
        wave_i = _wave(i)
        for j in range(len(order)):  # all non-ancestor turns
            if j == i or j in anc:
                continue
            eid = getattr(order[j], "event_id", None)
            if eid in completed:  # completed: neither breadth nor coverage
                continue
            is_future = dispatched_ids is None or eid not in dispatched_ids
            w, p = _boundary(i, j)
            if w > 0 or p > 0:
                (depths_future if is_future else depths_inflight).append((w, p))
                if is_future:
                    wg = _wave(j) - wave_i
                    if wg >= 0:  # forward: reuser is at or after target's wave
                        fut_wave.append((w, p, wg, abs(j - i) or 1))
            # covers_output: a FUTURE, later turn reuses this full prompt+output.
            if is_future and j > i and w >= na and len(msgs[j]) > na:
                covers = True

        def _gaps(boundary: Tuple[int, int], fut_wave=fut_wave):
            # min/max wave-gap over forward-future reusers reaching `boundary`.
            reaching = [(wg, ts) for (w, p, wg, ts) in fut_wave
                        if (w, p) >= boundary]
            if not reaching:
                return None, None
            min_wg = min(wg for wg, _ in reaching)
            ts_at_min = min(ts for wg, ts in reaching if wg == min_wg)
            min_gap = float(min_wg) + (1.0 - 1.0 / ts_at_min)
            max_gap = max(wg for wg, _ in reaching)
            return min_gap, max_gap

        # future_breadth per boundary; depths reused only by in-flight turns get
        # a floor breadth of 1. Boundaries ascending -> breadth non-increasing
        # (future_breadth is monotone; floors sit only where future_breadth==0,
        # necessarily at or beyond the deepest future boundary).
        segs_list: List[Tuple[int, int, int, Optional[float], Optional[int]]] = []
        for (bw, bp) in sorted(set(depths_future) | set(depths_inflight)):
            fb = sum(1 for d in depths_future if d >= (bw, bp))
            min_gap, max_gap = _gaps((bw, bp))
            if fb >= 1:
                segs_list.append((bw, bp, fb, min_gap, max_gap))
            elif any(d >= (bw, bp) for d in depths_inflight):
                segs_list.append((bw, bp, 1, min_gap, max_gap))  # coverage floor
        out[ev.event_id] = (tuple(segs_list), covers)
    return out


def _lifecycle_sets(graph_events, session_id, registry, dispatched_events) -> Tuple[frozenset, frozenset]:
    """Split a session's turns into (completed_ids, dispatched_ids) raw-id sets.

    The output registry is keyed by qualified id ("session:raw"); completed =
    an event whose output is recorded. dispatched_events is the per-session set
    of turns already sent; it is copied so the caller can add the current turn
    afterwards without mutating the returned snapshot.

    Args:
        graph_events: mapping of raw event_id -> graph node (keys used).
        session_id: this session's id (registry prefix).
        registry: EventOutputRegistry or None.
        dispatched_events: set of raw event_ids already dispatched.

    Returns:
        (completed_ids, dispatched_ids) as raw-id frozensets.
    """
    if registry is None:
        completed = frozenset()
    else:
        recorded = set(registry.get_event_ids())  # qualified ids
        completed = frozenset(
            rid for rid in graph_events if f"{session_id}:{rid}" in recorded
        )
    return completed, frozenset(dispatched_events)


def _compute_reuse_profiles(events) -> Dict[str, List[ReuseSegment]]:
    """Build each producer's reuse-depth profile from consumers' input_segments.

    A consumer reuses only the LEADING contiguous run of shared/output segments
    (the first 'unique' segment breaks prefix-cache contiguity). breadth(t) =
    #consumers reusing >= t tokens. Producers with no reuse are absent (terminal)."""
    by_id = {e.event_id: e for e in events}
    all_starts = sorted(e.t_start_ms for e in events)
    # pid -> [(reused_tokens, consumer_t_start_ms, msgs, covers_output)]
    # msgs = producer messages covered by the consumer's leading shared run;
    # covers_output = the run also includes the producer's generated output.
    # These anchor each boundary structurally for exact render calibration.
    consumers: Dict[str, List[Tuple[int, int, int, bool]]] = defaultdict(list)
    for c in events:
        reused_per_pred: Dict[str, int] = defaultdict(int)
        msgs_per_pred: Dict[str, int] = defaultdict(int)
        out_per_pred: Dict[str, bool] = defaultdict(bool)
        for seg in c.call.input_segments:
            if seg.type == "unique":
                break  # prefix-cache contiguity ends at the consumer's first own/new content
            if seg.source_event_id is not None:  # leading shared/output run
                reused_per_pred[seg.source_event_id] += seg.token_count
                if seg.type == "output":
                    out_per_pred[seg.source_event_id] = True
                else:
                    msgs_per_pred[seg.source_event_id] += seg.message_count
        for sid, toks in reused_per_pred.items():
            if toks <= 0:
                continue
            if by_id.get(sid) is None:
                continue
            consumers[sid].append(
                (toks, c.t_start_ms, msgs_per_pred[sid], out_per_pred[sid])
            )
    profiles: Dict[str, List[ReuseSegment]] = {}
    for sid, lst in consumers.items():
        prod_end = by_id[sid].t_end_ms
        prev = 0
        segs = []
        anchor = {t: (m, o) for t, _, m, o in lst}  # boundary -> structural anchor
        for b in sorted({t for t, _, _, _ in lst}):
            covering = [(t, cs) for t, cs, _, _ in lst if t >= b]
            # Longest untouched gap in this region between consecutive uses
            # (producer end, then each covering reuse start). Frequent reuse ->
            # small cold_gap; a genuine set-aside -> large; the policy turns
            # this into a TTL. (gap-to-FARTHEST-reuse would conflate the two.)
            marks = sorted([prod_end] + [cs for _, cs in covering])
            cold_gap = max(
                (sum(1 for st in all_starts if marks[i] < st < marks[i + 1])
                 for i in range(len(marks) - 1)),
                default=0,
            )
            end_msg, covers_output = anchor[b]
            segs.append(ReuseSegment(
                start=prev, end=b,
                breadth=len(covering),
                cold_gap=cold_gap,
                end_msg=end_msg,
                covers_output=covers_output,
            ))
            prev = b
        profiles[sid] = segs
    return profiles


class EventFailedError(Exception):
    """Raised by EventOutputRegistry.require_async when the awaited event failed."""

    def __init__(self, event_id: str) -> None:
        super().__init__(f"Predecessor event {event_id!r} failed")
        self.event_id = event_id


class SessionInferenceInfo(InferenceInfo):
    """InferenceInfo subclass that also carries the raw output text."""

    output_text: Optional[str] = None
    output_message: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        *,
        request_metrics: Optional[RequestMetrics] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if request_metrics is None:
            request_metrics = RequestMetrics(text=Text(input_tokens=input_tokens or 0))
            if "response_metrics" not in kwargs:
                kwargs["response_metrics"] = UnaryResponseMetrics(output_tokens=output_tokens or 0)
        super().__init__(request_metrics=request_metrics, **kwargs)


class WorkerSessionTracker:
    """Per-worker tracking of event completions and session failures."""

    def __init__(self) -> None:
        self._event_completions: Dict[str, Dict[str, float]] = {}
        self._failed_sessions: Set[str] = set()
        # Per-session set of predecessor event_ids whose live tool_call
        # response was detected malformed at substitution time and replaced
        # with the recorded assistant message. Empty when
        # bad_tool_call_handling is `none`. Stored as a set so a
        # predecessor with multiple downstream consumers (DAG fan-out) is
        # counted once, not once per consumer.
        self._recorded_substitution_event_ids: Dict[str, Set[str]] = {}
        # Per-session set of event_ids that have reached a terminal state on this worker —
        # whether they completed, were skipped, or failed. Used to detect when a session is
        # fully drained so the worker can evict its (otherwise never-freed) built graph.
        # A set (not a counter) makes drain accounting idempotent: an event that passes
        # through more than one terminal path is counted at most once.
        self._drained_events: Dict[str, Set[str]] = {}

    def record_event_drained(self, session_id: str, event_id: str) -> int:
        """Mark one event as drained (terminal) and return the session's drained count.

        Called from every terminal path (completion, skip, request failure). Idempotent
        per event_id. The caller compares the returned count against the session's total
        event count to decide whether the session is fully drained and can be evicted.
        """
        drained = self._drained_events.setdefault(session_id, set())
        drained.add(event_id)
        return len(drained)

    def forget_session(self, session_id: str) -> None:
        """Drop all per-worker tracking for a session (called after eviction)."""
        self._event_completions.pop(session_id, None)
        self._failed_sessions.discard(session_id)
        self._drained_events.pop(session_id, None)
        self._recorded_substitution_event_ids.pop(session_id, None)

    def record_event_completed(self, session_id: str, event_id: str, completion_time: float) -> None:
        if session_id not in self._event_completions:
            self._event_completions[session_id] = {}
        self._event_completions[session_id][event_id] = completion_time

    def is_event_completed(self, session_id: str, event_id: str) -> bool:
        return session_id in self._event_completions and event_id in self._event_completions[session_id]

    def get_event_completion_time(self, session_id: str, event_id: str) -> Optional[float]:
        return self._event_completions.get(session_id, {}).get(event_id)

    def mark_session_failed(self, session_id: str) -> None:
        self._failed_sessions.add(session_id)

    def is_session_failed(self, session_id: str) -> bool:
        return session_id in self._failed_sessions

    def get_session_event_count(self, session_id: str) -> int:
        return len(self._event_completions.get(session_id, {}))

    def get_session_completion_times(self, session_id: str) -> Dict[str, float]:
        return self._event_completions.get(session_id, {}).copy()

    def record_recorded_substitution(self, session_id: str, event_id: str) -> None:
        """Tag a predecessor event_id whose live tool_call response was
        replaced with the recorded message. Idempotent."""
        self._recorded_substitution_event_ids.setdefault(session_id, set()).add(event_id)

    def get_session_recorded_substitution_event_ids(self, session_id: str) -> List[str]:
        # Sorted for deterministic test/log output.
        return sorted(self._recorded_substitution_event_ids.get(session_id, set()))


class EventOutputRegistry:
    """Per-worker registry mapping event_id → actual output text and input messages."""

    def __init__(self) -> None:
        self._event_output_text: Dict[str, str] = {}
        self._event_output_message: Dict[str, Dict[str, Any]] = {}
        self._event_input_messages: Dict[str, Any] = {}
        self._event_signals: Dict[str, asyncio.Event] = {}
        self._failed_event_ids: Set[str] = set()

    def record(
        self,
        event_id: str,
        output_text: str,
        messages: List[Any],
        output_message: Optional[Dict[str, Any]] = None,
    ) -> None:
        if event_id in self._event_output_text:
            raise ValueError(
                f"Event {event_id} has already been recorded. "
                f"Each event should only complete once. This indicates a bug in the replay logic."
            )

        self._event_output_text[event_id] = output_text
        self._event_input_messages[event_id] = list(messages) if messages else []
        if output_message is not None:
            self._event_output_message[event_id] = output_message

        if event_id in self._event_signals:
            self._event_signals[event_id].set()
            logger.debug(f"Set asyncio.Event signal for event {event_id}")

    def get_output_by_event_id(self, event_id: str) -> Optional[str]:
        return self._event_output_text.get(event_id)

    def get_message_by_event_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        return self._event_output_message.get(event_id)

    def get_messages_by_event_id(self, event_id: str) -> Optional[List[Any]]:
        return self._event_input_messages.get(event_id)

    def get_event_ids(self) -> List[str]:
        return list(self._event_output_text.keys())

    def record_failure(self, event_id: str) -> None:
        self._failed_event_ids.add(event_id)
        if event_id not in self._event_signals:
            self._event_signals[event_id] = asyncio.Event()
        self._event_signals[event_id].set()
        logger.debug(f"Recorded failure for event {event_id}")

    def is_event_failed(self, event_id: str) -> bool:
        return event_id in self._failed_event_ids

    async def require_async(self, event_id: str, timeout_sec: float = 3600.0) -> str:
        if event_id in self._failed_event_ids:
            raise EventFailedError(event_id)

        output = self._event_output_text.get(event_id)
        if output is not None:
            return output

        if event_id not in self._event_signals:
            self._event_signals[event_id] = asyncio.Event()
        signal = self._event_signals[event_id]

        if event_id in self._failed_event_ids:
            raise EventFailedError(event_id)
        output = self._event_output_text.get(event_id)
        if output is not None:
            return output

        logger.debug(f"Event {event_id} waiting on asyncio signal (zero threads)")

        try:
            await asyncio.wait_for(signal.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"EventOutputRegistry: output for '{event_id}' not available after "
                f"{timeout_sec:.1f}s. Check that the predecessor is not blocked or failed."
            ) from e

        if event_id in self._failed_event_ids:
            raise EventFailedError(event_id)

        output = self._event_output_text.get(event_id)
        assert output is not None, (
            f"asyncio signal fired for {event_id} but output missing from local cache — this is a bug in record()"
        )
        logger.debug(f"Event {event_id} woke from asyncio signal")
        return output


# ---------------------------------------------------------------------------
# Render-based exact coordinate calibration (client-side helper): the serving
# vLLM's /v1/chat/completions/render endpoint returns the exact token_ids for
# a payload. Prefix renders are cached by content hash (see _render_token_ids).
_render_semaphore = asyncio.Semaphore(8)
_render_cache: Dict[str, List[int]] = {}
_render_session = None  # lazy aiohttp.ClientSession


def _lcp_len(a: List[int], b: List[int]) -> int:
    """Length of the longest common prefix of two token-id lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


async def _render_token_ids(
    render_url: str, body: Dict[str, Any], cacheable: bool = False
) -> Optional[List[int]]:
    """POST body to <render_url>/v1/chat/completions/render → token_ids."""
    global _render_session
    import hashlib

    key = None
    if cacheable:
        key = hashlib.md5(
            json.dumps(body, sort_keys=True, default=str).encode()
        ).hexdigest()
        cached = _render_cache.get(key)
        if cached is not None:
            return cached
    if _render_session is None:
        import aiohttp
        _render_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            connector=aiohttp.TCPConnector(force_close=True),
        )
    async with _render_semaphore:
        async with _render_session.post(
            render_url.rstrip("/") + "/v1/chat/completions/render", json=body
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"render HTTP {resp.status}: {(await resp.text())[:200]}")
            data = await resp.json()
    ids = data.get("token_ids") or (data.get("prompt_token_ids"))
    if not isinstance(ids, list):
        raise RuntimeError(f"render response missing token_ids: {list(data)[:8]}")
    if key is not None:
        _render_cache[key] = ids
    return ids


class SessionChatCompletionAPIData(ChatCompletionAPIData):
    """ChatCompletionAPIData subclass for graph-backed session replay."""

    model_config = {"arbitrary_types_allowed": True}

    event_id: str
    # 이 이벤트의 전체 프롬프트를 leading prefix로 재사용하는 이후 연속 이벤트 수
    # (첫 미포함에서 중단). = 시뮬레이터 fwd_reuse. 정책의 잔여재사용 게이트용.
    remaining_reuse: int = 0
    # Forward per-DEPTH reuse profile of THIS event's prompt (token-granular),
    # from _forward_reuse_depths: a tuple of (shared_msgs, partial_chars,
    # breadth, min_gap, max_gap) segments ascending by depth (breadth =
    # #later turns reusing >= that depth; min_gap/max_gap = causal wave-gap
    # bounds over forward-future reusers), plus whether the output is reused.
    # Each depth-range is protected at a priority tier set by its own breadth.
    forward_segments: tuple = ()
    forward_covers_output: bool = False
    registry: EventOutputRegistry
    worker_tracker: WorkerSessionTracker
    completion_queue: Any
    total_events_in_session: int
    # Back-reference to the worker's datagen so the last event of a session can evict the
    # session's lazily-built graph from this worker (the worker never calls cleanup_session
    # otherwise, so built graphs would accumulate for the whole stage). Set in load_lazy_data.
    generator: Optional["ReplayGraphSessionGeneratorBase"] = None
    predecessor_event_ids: List[str] = field(default_factory=list)
    wait_ms: int = 0
    input_segments: List[InputSegment] = field(default_factory=list)
    original_messages: List[Dict[str, Any]] = field(default_factory=list)
    expected_output_content: Optional[str] = None
    skip_request: bool = False
    expected_output_is_tool_call: bool = False
    expected_output_tool_names: Optional[List[str]] = None
    # KV-cache invalidation configuration
    inject_random_session_id: bool = False
    session_random_string: Optional[str] = None
    override_tool_call_max_tokens: bool = False
    # Mitigation for tool-call responses with malformed JSON in `arguments`.
    # `none` (default) is byte-identical to upstream main; `use_recorded`
    # substitutes the recorded assistant message at the affected slot.
    bad_tool_call_handling: BadToolCallHandling = BadToolCallHandling.NONE
    # When True, output/shared segments are NOT substituted with live predecessor
    # output; the recorded assistant messages are sent as-is. Predecessor wait
    # timing is still enforced.
    disable_output_substitution: bool = False
    # Set by _build_messages_with_substitution when it calls record_failure
    # early (e.g. recorded fallback also malformed). Lets the caller pass the
    # right reason string to _fail_and_notify instead of a generic fallback.
    _substitution_failure_reason: Optional[str] = None
    # KV-cache retention policy (WorkflowAwarePolicy) for directive injection
    retention_policy: Any = None
    # Per-producer reuse-depth profile. None = not reused → no directive (EVICT).
    reuse_depth_profile: Optional[List[ReuseSegment]] = None

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> Dict[str, Any]:
        payload = await super().to_request_body(effective_model_name, max_tokens, ignore_eos, streaming)

        if self.expected_output_is_tool_call and self.tool_definitions:
            if self.override_tool_call_max_tokens:
                payload["ignore_eos"] = False
                # The recorded output_tokens might come from a different model/tokenizer.
                # The replay model may need significantly more tokens to express the
                # same tool call (different tokenizer, different tool-call preamble).
                # Use a generous cap and let ignore_eos=False stop generation naturally.
                payload["max_tokens"] = max(payload.get("max_tokens", 0) * 4, 4096)

            if "tool_choice" in payload:
                logger.warning(
                    f"Event {self.event_id}: payload already has tool_choice={payload['tool_choice']!r}; "
                    f"overwriting with replay-enforced value."
                )
            names = self.expected_output_tool_names or []
            # name is set explicitly in to_request_body and NOT passed through
            # _clean_parameters, so it survives schema cleaning unchanged.
            available = {t["name"] for t in self.tool_definitions if "name" in t}
            if len(names) == 1 and names[0] in available:
                # Force the exact function the trace recorded, so the successor's
                # role:tool messages (referencing it by name/index) stay coherent.
                payload["tool_choice"] = {"type": "function", "function": {"name": names[0]}}
            else:
                # Fall back to "required" when:
                # - the recorded tool name is not in this call's tool_definitions (the
                #   trace's tool lists don't always match its outputs — vLLM rejects
                #   a tool_choice that names a function not present in tools), or
                # - there were multiple tool calls (vLLM only accepts one name at a time).
                payload["tool_choice"] = "required"

        if self.retention_policy is not None:
            extra = await self._compute_retention_extra_body(payload)
            if extra:
                existing_extra = payload.get("extra_body") or {}
                existing_extra.update(extra)
                payload["extra_body"] = existing_extra

        return payload

    async def _compute_retention_extra_body(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build the extra_body retention_directives for this request.

        Primary path (render_url set): forward-increment protection via render
        + LCP (see _forward_reuse_directives) — bypasses the message-DAG
        producer profile, which under-credits reuse (empty profile -> no
        directive -> falls to LRU).

        Fallback (no render_url, or render failure): legacy profile path —
        char-ratio rescale of the producer reuse-depth profile.
        """
        if getattr(self.retention_policy, "render_url", None):
            directives = await self._forward_reuse_directives(payload)
            if directives is not None:
                scope = self._extract_session_id()
                _turn = getattr(self, "event_id", None)
                if not directives:
                    # reuses nothing → no directive (LRU). Still emit scope+turn
                    # so the server logs rid↔session/turn (metadata only; no
                    # directives = no protection applied).
                    r0: dict[str, Any] = {"retention_turn": _turn}
                    if scope is not None:
                        r0["retention_scope"] = scope
                    return r0
                result: dict[str, Any] = {
                    "retention_directives": directives,
                    "retention_turn": _turn,
                }
                if scope is not None:
                    result["retention_scope"] = scope
                return result
            # directives is None → render failed; fall through to legacy path.
        profile = self._rescale_profile_to_materialized(self.reuse_depth_profile)
        return self.retention_policy.compute_directives(
            reuse_depth_profile=profile,
            scope=self._extract_session_id(),
            remaining_reuse=self.remaining_reuse,
        )

    async def _forward_reuse_directives(
        self, payload: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Protect the forward-reused prefix of THIS turn's prompt at the exact
        TOKEN boundary (start=0), plus its output when the output is reused.

        _forward_reuse_depths gives the deepest prefix of this turn's own
        prompt some later turn reuses (whole leading messages + a char-prefix
        of the first diverging message). Render that prefix and the full
        prompt and take their LCP -> the exact materialized token boundary D;
        protect [0, D), and add a covers_output directive if
        forward_covers_output (server resolves it to the generated-output
        range).

        Every turn re-protects the prefix it still carries downstream, so the
        reused conversation prefix stays retained (TTL refreshed) while the
        non-reused tail falls to LRU — prompt-centric and block-granular, even
        where an earlier turn produced the tokens.

        Returns the directives ([] = nothing reused downstream), or None on
        render failure (caller falls back to the legacy profile path).
        """
        segs = self.forward_segments
        covers_out = self.forward_covers_output
        if not segs and not covers_out:
            return []
        base_body: Dict[str, Any] = {
            "model": payload.get("model"),
            "messages": payload.get("messages") or [],
            "max_tokens": 1,
        }
        if payload.get("tools"):
            base_body["tools"] = payload["tools"]
        messages = base_body["messages"]
        pol = self.retention_policy
        try:
            full_ids = await _render_token_ids(pol.render_url, base_body)
        except Exception as e:
            logger.warning(
                "Event %s: forward-reuse render failed (%s); falling back to "
                "profile path", self.event_id, e,
            )
            return None
        if not full_ids:
            return None
        # One directive per depth-range; priority tiered by THAT range's breadth.
        # Segments ascending in depth with non-increasing breadth -> priorities
        # non-increasing across token positions (prefix-cache validator-safe).
        directives: List[Dict[str, Any]] = []
        prev = 0
        last_breadth = 1
        last_min_gap: float | None = None
        last_max_gap: int | None = None
        wave_mode = getattr(pol, "priority_mode", "tiered") == "next_use_wave"
        for seg in segs:
            w, p, breadth = seg[0], seg[1], seg[2]
            min_gap = seg[3] if len(seg) > 3 else None
            max_gap = seg[4] if len(seg) > 4 else None
            pref_msgs = list(messages[:w])
            if p > 0 and w < len(messages):
                dm = messages[w]
                c = dm.get("content") if isinstance(dm, dict) else None
                if isinstance(c, str) and c:
                    pref_msgs.append({**dm, "content": c[:p]})
            if not pref_msgs:
                continue
            try:
                prefix_body = dict(base_body)
                prefix_body["messages"] = pref_msgs
                prefix_ids = await _render_token_ids(
                    pol.render_url, prefix_body, cacheable=True
                )
            except Exception as e:
                logger.warning(
                    "Event %s: forward-reuse render failed (%s); falling back to "
                    "profile path", self.event_id, e,
                )
                return None
            if prefix_ids is None:
                return None
            depth = _lcp_len(prefix_ids, full_ids)
            if depth > prev:
                if wave_mode:
                    priority = pol._priority_for_wave(min_gap)
                    duration = pol._ttl_for_wave(max_gap)
                else:
                    priority = pol._priority_for_breadth(breadth)
                    duration = (breadth * pol.per_span_s
                                + pol.queue_margin_s + pol.ttl_buffer_s)
                directives.append({
                    "start": prev,
                    "end": depth,
                    "priority": priority,
                    "duration": duration,
                })
                prev = depth
                last_breadth = breadth
                last_min_gap = min_gap
                last_max_gap = max_gap
        if covers_out:
            if wave_mode:
                priority = pol._priority_for_wave(last_min_gap)
                duration = pol._ttl_for_wave(last_max_gap)
            else:
                priority = pol._priority_for_breadth(last_breadth)
                duration = (last_breadth * pol.per_span_s
                            + pol.queue_margin_s + pol.ttl_buffer_s)
            directives.append({
                "covers_output": True,
                "priority": priority,
                "duration": duration,
            })
        return directives

    async def _calibrate_profile_via_render(
        self, payload: Dict[str, Any]
    ) -> Optional[List[ReuseSegment]]:
        """Resolve profile boundaries to EXACT materialized-token coordinates
        via the serving vLLM's /v1/chat/completions/render endpoint (exact
        token_ids for a payload, harmony/chat-template rendering included).

        For each distinct end_msg anchor, render the message prefix and take
        the LCP with the full render — the LCP length is the boundary in
        server token space regardless of generation-suffix differences.
        covers_output becomes len(full) + max_tokens (safe overcover; blocks
        past actual generation never materialize).

        Returns None on any failure (caller falls back to char-ratio rescale).
        Prefix renders are cached globally since past messages are immutable
        once substituted, so anchors repeat across a session's later turns.
        """
        profile = self.reuse_depth_profile
        if not profile:
            return None
        if not any(seg.end_msg is not None or seg.covers_output for seg in profile):
            return None  # unanchored legacy profile
        try:
            base_body: Dict[str, Any] = {
                "model": payload.get("model"),
                "messages": payload.get("messages") or [],
                "max_tokens": 1,
            }
            if payload.get("tools"):
                base_body["tools"] = payload["tools"]
            full_ids = await _render_token_ids(
                self.retention_policy.render_url, base_body
            )
            if not full_ids:
                return None
            prompt_len = len(full_ids)
            messages = base_body["messages"]
            out_cap = int(payload.get("max_tokens") or 0)

            # Resolve each distinct message-anchor once.
            anchors: Dict[int, int] = {}
            for seg in profile:
                m = seg.end_msg
                if seg.covers_output or m is None or m in anchors:
                    continue
                if m <= 0:
                    anchors[m] = 0
                    continue
                if m >= len(messages):
                    anchors[m] = prompt_len
                    continue
                prefix_body = dict(base_body)
                prefix_body["messages"] = messages[:m]
                prefix_ids = await _render_token_ids(
                    self.retention_policy.render_url, prefix_body, cacheable=True
                )
                if prefix_ids is None:
                    return None
                anchors[m] = _lcp_len(prefix_ids, full_ids)

            calibrated: List[ReuseSegment] = []
            prev = 0
            for seg in profile:
                if seg.covers_output:
                    # Reuse extends through the producer's generated output.
                    # Exact output length is unknown at directive time; cover
                    # up to the max_tokens cap — blocks past actual generation
                    # never materialize, so overcover is free.
                    new_end = prompt_len + out_cap
                else:
                    new_end = anchors.get(seg.end_msg, seg.end)  # type: ignore[arg-type]
                new_end = max(new_end, prev)  # keep monotone
                calibrated.append(ReuseSegment(
                    start=prev, end=new_end,
                    breadth=seg.breadth, cold_gap=seg.cold_gap,
                    end_msg=seg.end_msg, covers_output=seg.covers_output,
                ))
                prev = new_end
            logger.debug(
                "Event %s: render-calibrated %d segment(s), prompt_len=%d",
                self.event_id, len(calibrated), prompt_len,
            )
            return calibrated
        except Exception as e:
            logger.warning(
                "Event %s: render calibration failed (%s: %s); falling back to rescale",
                self.event_id, type(e).__name__, e,
            )
            return None

    def _rescale_profile_to_materialized(
        self, profile: Optional[List[ReuseSegment]]
    ) -> Optional[List[ReuseSegment]]:
        """Map directive coordinates from recorded-trace token space into the
        materialized request's token space.

        The profile is computed at graph time from RECORDED messages'
        estimated token depths, but the actual request differs (substituted
        live outputs, tool/template preamble, tokenizer differences — observed
        ~2x on tool-heavy traces). Leaving directives in recorded coordinates
        protects only a leading fraction of the real reused prefix, squeezing
        the rest into thrash (measured -25pp hit vs plain LRU).

        Best-effort correction: scale boundaries by the materialized/recorded
        content-size ratio (same estimator both sides, so its bias cancels)
        and shift by the tool-definitions preamble the server renders ahead of
        the messages."""
        if not profile:
            return profile

        def _content(m: Any) -> str:
            c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            return c if isinstance(c, str) else ""

        from inference_perf.datagen.otel_trace_to_replay_graph import estimate_tokens

        recorded = sum(estimate_tokens(_content(m)) for m in (self.original_messages or []))
        actual = sum(estimate_tokens(_content(m)) for m in (self.messages or []))
        if recorded <= 0 or actual <= 0:
            return profile
        scale = actual / recorded
        offset = 0
        if self.tool_definitions:
            try:
                offset = estimate_tokens(json.dumps(self.tool_definitions))
            except (TypeError, ValueError):
                offset = 0
        if abs(scale - 1.0) < 0.02 and offset == 0:
            return profile
        return [
            ReuseSegment(
                start=offset + int(seg.start * scale),
                end=offset + int(seg.end * scale),
                breadth=seg.breadth,
                cold_gap=seg.cold_gap,
            )
            for seg in profile
        ]

    def _extract_session_id(self) -> str:
        return self.event_id.split(":")[0] if ":" in self.event_id else self.event_id

    def _fail_and_notify(self, session_id: str, reason: str) -> None:
        """Mark this event and session as failed, notify the completion queue.

        Called from wait_for_predecessors_and_substitute when we decide to skip
        the request before it reaches the model server (predecessor failed,
        session already failed, or substitution produced an invalid message).
        Mirrors the queue notification logic in process_failure so the main-
        process completion loop is not left waiting indefinitely.
        """
        self.skip_request = True
        was_already_failed = self.worker_tracker.is_session_failed(session_id)
        self.worker_tracker.mark_session_failed(session_id)
        self.registry.record_failure(self.event_id)
        logger.info(f"Event {self.event_id} skipping — {reason}")

        if not was_already_failed and self.completion_queue is not None:
            completion_time = time.perf_counter()
            completed_so_far = self.worker_tracker.get_session_event_count(session_id)
            cancelled = self.total_events_in_session - completed_so_far - 1
            completion_data = {
                "session_id": session_id,
                "completion_time": completion_time,
                "failed": True,
                "failure_reason": reason,
                "cancelled_events": cancelled,
                "event_completion_times": self.worker_tracker.get_session_completion_times(session_id),
            }
            try:
                self.completion_queue.put_nowait(completion_data)
                logger.debug(f"Pushed skip-failure notification for session {session_id} (cancelled_events={cancelled})")
            except Exception as e:
                logger.error(f"Failed to push skip-failure notification for session {session_id}: {e}")

        # This event is now terminal (skipped). Count it toward the worker drain so the
        # session is evicted once its last event drains.
        self._mark_drained_and_maybe_evict(session_id)

    def _mark_drained_and_maybe_evict(self, session_id: str) -> None:
        """Record this event as drained on the worker; evict the session once all drain.

        Every event ends in exactly one terminal state on the worker — completed, skipped,
        or request-failed — and each terminal path calls this exactly once. When the number
        of drained events reaches total_events_in_session, the session is fully done on this
        worker and its lazily-built graph can be freed.

        This is what keeps a worker's resident memory bounded to roughly the concurrent
        working set. Without it, each worker retains every session's graph it ever built for
        the lifetime of the stage (the parent calls cleanup_session, but workers never do),
        so memory grows with the number of sessions processed — catastrophically for large
        corpora or high duplicate_sessions_target, especially when sessions fail fast and the
        session pool churns quickly.

        Eviction is safe here precisely because the session is fully drained: no further
        events for it remain in flight or in the worker's queue, so nothing will try to read
        the freed graph. (A session that fails mid-way still drains every event — the
        successors flow through the skip path and are counted here too.)
        """
        if self.generator is None:
            return
        drained = self.worker_tracker.record_event_drained(session_id, self.event_id)
        if drained >= self.total_events_in_session:
            self.generator.evict_worker_session(session_id)

    async def wait_for_predecessors_and_substitute(self) -> None:
        session_id = self._extract_session_id()

        if self.worker_tracker.is_session_failed(session_id):
            self._fail_and_notify(session_id, "session already failed (pre-wait check)")
            return

        if self.predecessor_event_ids:
            logger.debug(f"Event {self.event_id} waiting for {len(self.predecessor_event_ids)} predecessor(s)")
            try:
                await asyncio.gather(
                    *[self.registry.require_async(event_id, timeout_sec=3600.0) for event_id in self.predecessor_event_ids]
                )
            except EventFailedError:
                self._fail_and_notify(session_id, "predecessor failed")
                return
            except (TimeoutError, asyncio.TimeoutError) as e:
                self._fail_and_notify(session_id, f"predecessor wait failed: {type(e).__name__}")
                return
            logger.debug(f"Event {self.event_id} all predecessors done")

        if self.wait_ms > 0:
            wait_sec = self.wait_ms / 1000.0
            logger.debug(f"Event {self.event_id} waiting {wait_sec:.3f}s (wait_ms={self.wait_ms})")
            await asyncio.sleep(wait_sec)

        # Substitute output segments with actual predecessor outputs, or inject random session ID into unique segments
        needs_substitution = (not self.disable_output_substitution) and any(
            seg.type == "output" or seg.type == "shared" for seg in self.input_segments
        )
        # Inject random string if flag is enabled OR session is a duplicate
        is_duplicate = ReplayGraphSessionGeneratorBase.is_duplicate_session(session_id)
        needs_random_injection = (self.inject_random_session_id or is_duplicate) and any(
            seg.type == "unique" for seg in self.input_segments
        )
        if needs_substitution or needs_random_injection:
            if needs_substitution:
                logger.debug(f"Event {self.event_id} substituting output/shared segments")
            if needs_random_injection:
                reason = "flag enabled" if self.inject_random_session_id else "duplicate session"
                logger.debug(f"Event {self.event_id} injecting random session ID ({reason})")

            substituted = self._build_messages_with_substitution()
            # _build_messages_with_substitution calls record_failure and returns
            # early when substitution is not possible (e.g. tool call expected but
            # live model returned plain text). Detect that and skip the request.
            if self.registry.is_event_failed(self.event_id):
                reason = self._substitution_failure_reason or "Unknown"
                self._fail_and_notify(session_id, reason)
                return
            self.messages = [
                ChatMessage(
                    role=m["role"],
                    content=m.get("content"),
                    reasoning_content=m.get("reasoning") or m.get("reasoning_content"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                )
                for m in substituted
            ]
            logger.debug(f"Event {self.event_id} substitution/injection complete, {len(self.messages)} messages")

    def _build_messages_with_substitution(self) -> List[Dict[str, Any]]:
        # NOTE: when input_segments is empty, the original_messages list is returned
        # by reference (not copied). Callers must not mutate the returned list.
        if not self.input_segments:
            return self.original_messages

        result: List[Dict[str, Any]] = []
        cursor = 0

        # Track live tool-call assistant messages as (index_in_result, tool_calls)
        # for a post-pass rewrite of tool_call_id in the role:tool messages that
        # follow. Post-pass rather than inline because those messages live in a
        # later ("unique") segment not yet in `result` when the "output" segment
        # is processed; the recorded index equals len(result) at append time.
        pending_id_rewrites: List[Tuple[int, List[Dict[str, Any]]]] = []

        for seg in self.input_segments:
            seg_msgs = self.original_messages[cursor : cursor + seg.message_count]

            if seg.type == "output":
                # An output segment must cover exactly one message — the assistant
                # turn that will be replaced by the predecessor's live output.
                if seg.message_count != 1:
                    logger.error(
                        f"Event {self.event_id}: output segment has message_count={seg.message_count} "
                        f"(expected 1). Using recorded messages to avoid index corruption."
                    )
                    result.extend(seg_msgs)
                    cursor += seg.message_count
                    continue

                if seg.source_event_id:
                    actual_message = self.registry.get_message_by_event_id(seg.source_event_id)
                    if actual_message:
                        live_tool_calls = actual_message.get("tool_calls")
                        # Inline detection: if the live model produced tool_calls
                        # AND bad_tool_call_handling is enabled, check each
                        # `arguments` for valid JSON. Any failure means the next
                        # request would 400 on the chat template's json.loads.
                        bad_live = (
                            _detect_bad_tool_calls(live_tool_calls)
                            if (self.bad_tool_call_handling == BadToolCallHandling.USE_RECORDED and live_tool_calls)
                            else []
                        )
                        if bad_live:
                            # Substitute the recorded assistant message at this slot.
                            # The recorded message is `seg_msgs[0]` (output segments
                            # cover exactly one message — the assistant turn). Its
                            # `tool_call_id` flows naturally into the role:tool
                            # successors that follow in `original_messages`, so no
                            # entry is appended to pending_id_rewrites for this
                            # position. The wire body for the demoted slot is
                            # structurally identical to a healthy replay.
                            recorded_message = seg_msgs[0]
                            recorded_tool_calls = recorded_message.get("tool_calls") or []
                            # Defensive: if the recorded trace ALSO has malformed
                            # tool_calls at this slot, we have no clean fallback.
                            # Hard-fail the event; EventFailedError will cascade
                            # to downstream events that await this one. Parallel
                            # DAG branches continue.
                            bad_recorded = _detect_bad_tool_calls(recorded_tool_calls)
                            if bad_recorded:
                                logger.error(
                                    f"Event {self.event_id}: bad_tool_call_handling=use_recorded "
                                    f"detected malformed live tool_calls from {seg.source_event_id}, "
                                    f"but the recorded fallback is also malformed "
                                    f"(errors={[e for (_, _, e) in bad_recorded]}). Failing event."
                                )
                                self._substitution_failure_reason = (
                                    f"recorded fallback for {seg.source_event_id} is also malformed"
                                )
                                self.registry.record_failure(self.event_id)
                                return result  # partial; caller checks is_event_failed
                            # Tag the predecessor event_id for telemetry. Set
                            # semantics dedupe across DAG fan-out.
                            session_id = self._extract_session_id()
                            pred_event_id = (
                                seg.source_event_id.split(":", 1)[1] if ":" in seg.source_event_id else seg.source_event_id
                            )
                            self.worker_tracker.record_recorded_substitution(session_id, pred_event_id)
                            result.append(recorded_message)
                            logger.warning(
                                f"Event {self.event_id}: substituted RECORDED message for "
                                f"{seg.source_event_id} (live had {len(bad_live)} malformed "
                                f"tool_call(s); recorded structurally clean)"
                            )
                        else:
                            if live_tool_calls:
                                # Record the position of this assistant message so the post-pass
                                # can rewrite tool_call_id in the role:tool messages that follow.
                                pending_id_rewrites.append((len(result), live_tool_calls))
                            result.append(actual_message)
                            logger.debug(
                                f"Event {self.event_id}: substituted output segment with structured message from {seg.source_event_id}"
                            )
                    else:
                        actual_output = self.registry.get_output_by_event_id(seg.source_event_id)
                        logger.debug(
                            f"Registry get for event {self.event_id} output segment from {seg.source_event_id} generated: {actual_output}"
                        )
                        logger.warning(
                            f"Event {self.event_id}: unable to get the actual output message from {seg.source_event_id}. "
                            f"Using output text instead. "
                        )
                        if actual_output:
                            if self.expected_output_is_tool_call:
                                # Live model returned plain text where a tool call was expected;
                                # the successor's role:tool messages would have dangling
                                # tool_call_id refs and likely get rejected. Fail this event so
                                # downstream events skip rather than send broken requests.
                                logger.warning(
                                    f"Event {self.event_id}: original output was a tool call but live model "
                                    f"returned plain text. Marking event as failed to prevent downstream "
                                    f"requests with dangling tool_call_id references."
                                )
                                self._substitution_failure_reason = (
                                    "substitution failed (tool call expected but plain text returned)"
                                )
                                self.registry.record_failure(self.event_id)
                                return result  # partial result; caller should check skip_request
                            for msg in seg_msgs:
                                substituted = dict(msg)
                                substituted["content"] = actual_output
                                result.append(substituted)
                            logger.debug(
                                f"Event {self.event_id}: substituted output segment with text output from {seg.source_event_id}"
                            )
                        else:
                            logger.debug(
                                f"Event {self.event_id}: output segment from {seg.source_event_id} "
                                f"not available, using recorded content"
                            )
                            result.extend(seg_msgs)
                else:
                    logger.debug(f"Event {self.event_id}: output segment has no source_event_id, using recorded content")
                    result.extend(seg_msgs)
            elif seg.type == "shared":
                if seg.source_event_id is None:
                    logger.error(f"CRITICAL: Event {self.event_id} shared segment has no source_event_id")
                    result.extend(seg_msgs)
                    continue
                seg_msgs_from_parent = self.registry.get_messages_by_event_id(seg.source_event_id)
                if seg_msgs_from_parent is None:
                    logger.error(
                        f"CRITICAL: Event {self.event_id} shared segment from {seg.source_event_id} "
                        f"has no messages in registry (should not happen after require_async)"
                    )
                    result.extend(seg_msgs)
                else:
                    # The shared segment is a prefix of the parent's messages, which may
                    # have more messages than the shared prefix length.
                    seg_msgs_from_parent = seg_msgs_from_parent[: seg.message_count]

                    logger.debug(
                        f"Registry get for event {self.event_id} from {seg.source_event_id} "
                        f"shared segment: using {len(seg_msgs_from_parent)} messages (prefix of parent's messages)"
                    )

                    if len(seg_msgs_from_parent) != seg.message_count:
                        logger.warning(
                            f"Event {self.event_id} shared segment from {seg.source_event_id} "
                            f"expected {seg.message_count} messages but parent only has {len(seg_msgs_from_parent)}. "
                            f"Using recorded messages as fallback."
                        )
                        result.extend(seg_msgs)
                    else:
                        for msg in seg_msgs_from_parent:
                            if isinstance(msg, ChatMessage):
                                result.append({k: v for k, v in msg.model_dump().items() if v is not None})
                            else:
                                result.append(dict(msg))
            elif seg.type == "unique":
                # Inject the random session string (flag enabled, or a duplicate session)
                # to invalidate KV-cache reuse across sessions.
                for msg in seg_msgs:
                    session_id = self._extract_session_id()
                    is_duplicate = ReplayGraphSessionGeneratorBase.is_duplicate_session(session_id)
                    should_inject = (self.inject_random_session_id or is_duplicate) and self.session_random_string

                    if should_inject:
                        msg_copy = dict(msg)
                        original_content = msg_copy.get("content", "")
                        msg_copy["content"] = f"[SESS:{self.session_random_string}] {original_content}"
                        result.append(msg_copy)
                        reason = "flag enabled" if self.inject_random_session_id else "duplicate session"
                        logger.debug(f"Event {self.event_id}: injected random session string ({reason})")
                    else:
                        result.append(msg)
            else:
                result.extend(seg_msgs)

            cursor += seg.message_count

        # Post-pass: rewrite tool_call_id in role:tool messages to match the live
        # tool call IDs instead of the recorded (now stale) ones. By index rather
        # than name, since the live model may call the same function twice — the
        # i-th role:tool message corresponds to the i-th tool call in the
        # preceding assistant message (per the OpenAI spec). Scans forward from
        # the assistant message, skipping intervening non-tool messages.
        for assistant_idx, live_tool_calls in pending_id_rewrites:
            tool_result_idx = 0
            for result_idx in range(assistant_idx + 1, len(result)):
                if tool_result_idx >= len(live_tool_calls):
                    break
                msg = result[result_idx]
                if msg.get("role") == "tool":
                    live_id = live_tool_calls[tool_result_idx].get("id")
                    if live_id:
                        msg = dict(msg)  # copy before mutating — the dict may be shared
                        msg["tool_call_id"] = live_id
                        result[result_idx] = msg
                        logger.debug(
                            f"Event {self.event_id}: rewrote tool_call_id at position {result_idx} "
                            f"to live ID {live_id!r} (index {tool_result_idx})"
                        )
                    tool_result_idx += 1

        return result

    def on_completion(self, info: InferenceInfo) -> None:
        output_text = info.output_text if isinstance(info, SessionInferenceInfo) else ""
        output_text = output_text or ""
        output_message = info.output_message if isinstance(info, SessionInferenceInfo) else None
        self.registry.record(self.event_id, output_text, self.messages, output_message=output_message)
        logger.debug(
            f"calling registry record for event {self.event_id} num input messages {len(self.messages)} and output: {output_text}"
        )
        completion_time = time.perf_counter()
        session_id = self._extract_session_id()
        event_id = self.event_id.split(":", 1)[1] if ":" in self.event_id else self.event_id
        self.worker_tracker.record_event_completed(session_id, event_id, completion_time)
        logger.debug(f"Recorded event completion in worker tracker for {self.event_id}")

        completed_count = self.worker_tracker.get_session_event_count(session_id)

        if completed_count == self.total_events_in_session:
            logger.debug(f"Session {session_id} completed all {self.total_events_in_session} events in worker")

            completion_data = {
                "session_id": session_id,
                "completion_time": completion_time,
                "failed": self.worker_tracker.is_session_failed(session_id),
                "event_completion_times": self.worker_tracker.get_session_completion_times(session_id),
            }
            # Telemetry: only emit recorded-substitution keys when at least
            # one substitution fired in this session, so upstream-default runs
            # (handling=none, or handling set but no malformed tool_calls
            # observed) produce an identical wire format.
            recorded_subst_ids = self.worker_tracker.get_session_recorded_substitution_event_ids(session_id)
            if recorded_subst_ids:
                completion_data["recorded_substitution_event_ids"] = recorded_subst_ids
                completion_data["n_recorded_substitutions"] = len(recorded_subst_ids)

            if self.completion_queue is not None:
                try:
                    self.completion_queue.put_nowait(completion_data)
                    logger.debug(f"Pushed session {session_id} completion to queue")
                except Exception as e:
                    logger.error(f"Failed to push session {session_id} completion to queue: {e}")

        # This event is now terminal (completed). Count it toward the worker drain and evict
        # the session once its last event drains. Done last: eviction clears worker_tracker
        # state for the session, so the completion-queue notification above must be built first.
        self._mark_drained_and_maybe_evict(session_id)

    async def process_response(
        self,
        response: ClientResponse,
        config: APIConfig,
        tokenizer: CustomTokenizer,
        lora_adapter: Optional[str] = None,
    ) -> SessionInferenceInfo:
        """Process the LLM response, capture output text, and register it."""
        logger.debug(f"process_response called for event {self.event_id}")
        output_text: str = ""

        def _get_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") in ("text", "input_text")
                    ]
                )
            return ""

        if config.streaming:
            # Accumulate tool_call chunks and reasoning_content alongside text content.
            # delta.tool_calls is a list of partial objects; each chunk carries an
            # index that identifies which tool call it belongs to.
            tool_call_chunks: Dict[int, Dict[str, Any]] = {}
            reasoning_content_chunks: list[str] = []

            def _extract_streaming_content(data: Dict[str, Any]) -> Optional[str]:
                delta = data.get("choices", [{}])[0].get("delta", {})
                for chunk in delta.get("tool_calls") or []:
                    idx = chunk.get("index", 0)
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": chunk.get("id", ""),
                            "type": chunk.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        }
                    fn = chunk.get("function") or {}
                    if fn.get("name"):
                        tool_call_chunks[idx]["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tool_call_chunks[idx]["function"]["arguments"] += fn["arguments"]
                    if chunk.get("id"):
                        tool_call_chunks[idx]["id"] = chunk["id"]

                # Accumulate reasoning chunks (prefer "reasoning", fall back to "reasoning_content")
                reasoning_chunk = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning_chunk is not None:
                    reasoning_content_chunks.append(str(reasoning_chunk))

                content = delta.get("content")
                return str(content) if content is not None else None

            text_content, chunk_times, raw_content, response_chunks, server_usage = await parse_sse_stream(
                response, extract_content=_extract_streaming_content
            )

            # Combine reasoning_content with text_content in output_text (used for token count)
            reasoning_text = "".join(reasoning_content_chunks) if reasoning_content_chunks else ""
            if reasoning_text:
                output_text = reasoning_text + text_content
            else:
                output_text = text_content

            streaming_output_message: Optional[Dict[str, Any]] = None
            if tool_call_chunks:
                live_tool_calls = [tool_call_chunks[i] for i in sorted(tool_call_chunks)]
                streaming_output_message = {"role": "assistant", "tool_calls": live_tool_calls}
                if text_content:
                    streaming_output_message["content"] = text_content
            else:
                streaming_output_message = {"role": "assistant", "content": text_content}

            if reasoning_text:
                streaming_output_message["reasoning_content"] = reasoning_text

            prompt_text = "".join([_get_text(msg.content) for msg in self.messages if msg.content])
            prompt_len = tokenizer.count_tokens(prompt_text)
            server_completion_tokens = server_usage.get("completion_tokens") if server_usage else None
            if server_completion_tokens is not None:
                output_len = int(server_completion_tokens)
            else:
                tc_text = ""
                if tool_call_chunks:
                    tc_text = json.dumps([tool_call_chunks[i] for i in sorted(tool_call_chunks)], ensure_ascii=False)
                output_len = tokenizer.count_tokens(output_text + tc_text)
            info = SessionInferenceInfo(
                request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                response_metrics=StreamedResponseMetrics(
                    response_chunks=response_chunks,
                    chunk_times=chunk_times,
                    output_tokens=output_len,
                    output_token_times=chunk_times,
                    server_usage=server_usage,
                ),
                lora_adapter=lora_adapter,
                output_text=output_text or None,
                output_message=streaming_output_message,
                extra_info={"raw_response": raw_content},
            )
        else:
            data = await response.json()
            prompt_len = tokenizer.count_tokens("".join([_get_text(m.content) for m in self.messages]))
            choices = data.get("choices", [])
            output_message: Optional[Dict[str, Any]] = None
            tool_calls = None
            if choices:
                msg_dict = choices[0].get("message", {})
                text_content = msg_dict.get("content", "") or ""
                tool_calls = msg_dict.get("tool_calls")
                reasoning_content = msg_dict.get("reasoning") or msg_dict.get("reasoning_content")

                # Combine reasoning_content with output text for the content field
                if reasoning_content:
                    output_text = reasoning_content + text_content
                else:
                    output_text = text_content

                if tool_calls:
                    output_message = {"role": "assistant", "tool_calls": tool_calls}
                    if text_content:
                        output_message["content"] = text_content
                else:
                    output_message = {"role": "assistant", "content": text_content}

                if reasoning_content:
                    output_message["reasoning_content"] = reasoning_content
            usage = data.get("usage") or {}
            server_completion_tokens = usage.get("completion_tokens")
            if server_completion_tokens is not None:
                output_len = int(server_completion_tokens)
            else:
                tc_text = ""
                if tool_calls:
                    tc_text = json.dumps(tool_calls, ensure_ascii=False)
                output_len = tokenizer.count_tokens(output_text + tc_text)
            info = SessionInferenceInfo(
                request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                response_metrics=UnaryResponseMetrics(output_tokens=output_len),
                lora_adapter=lora_adapter,
                output_text=output_text or None,
                output_message=output_message,
            )

        # Register output and notify successors.
        self.on_completion(info)

        if output_text:
            logger.debug(f"Registered output for event {self.event_id}: {len(output_text)} chars : {output_text}")
        else:
            logger.debug(f"Registered empty output for event {self.event_id}")

        return info

    async def process_failure(
        self,
        response: Optional[ClientResponse],
        config: APIConfig,
        tokenizer: CustomTokenizer,
        exception: Exception,
        lora_adapter: Optional[str] = None,
    ) -> InferenceInfo:
        logger.error(f"Request failed for event {self.event_id}: {type(exception).__name__}: {str(exception)}")

        session_id = self._extract_session_id()
        was_already_failed = self.worker_tracker.is_session_failed(session_id)
        self.worker_tracker.mark_session_failed(session_id)
        self.registry.record_failure(self.event_id)

        if not was_already_failed and self.completion_queue is not None:
            completion_time = time.perf_counter()
            completed_so_far = self.worker_tracker.get_session_event_count(session_id)
            cancelled = self.total_events_in_session - completed_so_far - 1
            completion_data = {
                "session_id": session_id,
                "completion_time": completion_time,
                "failed": True,
                "cancelled_events": cancelled,
                "event_completion_times": self.worker_tracker.get_session_completion_times(session_id),
            }

            try:
                logger.debug(f"Pushing immediate failure notification for session {session_id}")
                self.completion_queue.put_nowait(completion_data)
                logger.info(f"Session {session_id} failure notification sent to main process (cancelled_events={cancelled})")
            except Exception as e:
                logger.error(f"Failed to push session {session_id} failure notification to queue: {e}")

        # This event is now terminal (request failed). Count it toward the worker drain. Note
        # that when an event fails, its successors are still queued; they will be dequeued and
        # flow through the skip path, each counted there. Eviction therefore only fires when
        # the last of them drains — after which nothing re-reads the session — so the graph is
        # never rebuilt by a late successor.
        self._mark_drained_and_maybe_evict(session_id)

        return SessionInferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=0)),
            response_metrics=UnaryResponseMetrics(output_tokens=0),
            lora_adapter=lora_adapter,
            output_text="",
        )


@dataclass
class ReplaySessionState:
    """Tracks graph traversal state for one session."""

    session_id: str
    graph: ReplayGraph
    ready_events: Set[str]
    dispatched_events: Set[str]
    completed_events: Set[str]
    event_completion_times: Dict[str, float]
    is_active: bool = False
    is_complete: bool = False
    failed: bool = False
    failure_reason: Optional[str] = None
    cancelled_events: int = 0
    # Populated from completion_data when the worker finishes the
    # session. None means the worker did not push the keys (i.e.
    # bad_tool_call_handling=none); empty list means handling was
    # enabled but no substitutions fired.
    n_recorded_substitutions: Optional[int] = None
    recorded_substitution_event_ids: Optional[List[str]] = None
    random_string: Optional[str] = None  # Random string for KV-cache invalidation (shared by all events in session)


@dataclass
class ReplaySessionEvent:
    """Represents a single replayable event derived from a graph event."""

    call_id: str
    event_id: str
    session_index: int
    t_start_ms: int
    t_end_ms: int
    model: str
    messages: List[Dict[str, Any]]
    expected_output: str
    input_segments: List[InputSegment]
    expected_output_tokens: int
    temperature: Optional[float]
    max_tokens_recorded: Optional[int]
    predecessor_event_ids: List[str] = field(default_factory=list)
    wait_ms: int = 0
    tool_definitions: Optional[List[Dict[str, Any]]] = None
    reuse_depth_profile: Optional[List[ReuseSegment]] = None


@dataclass
class ReplaySession:
    """Represents one replayable session backed by a ReplayGraph."""

    session_id: str
    source_id: str
    session_index: int
    graph: ReplayGraph
    start_offset_ms: int = 0


class ReplayGraphSessionGeneratorBase(SessionGenerator, LazyLoadDataMixin):
    """Shared runtime for ReplayGraph-backed session replay generators."""

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        mp_manager: Optional[SyncManager] = None,
        base_seed: Optional[int] = None,
        num_workers: int = 1,
        replay_config: Optional[SessionReplayConfig] = None,
        retention_policy: Any = None,
    ) -> None:
        super().__init__(api_config, config, tokenizer)
        self.config = config
        self.replay_config = replay_config
        self.mp_manager = mp_manager
        self.num_workers = max(1, num_workers)
        self.base_seed = base_seed if base_seed is not None else 42
        self.retention_policy = retention_policy

        self.output_registry = EventOutputRegistry()
        self.worker_tracker = WorkerSessionTracker()
        if mp_manager is not None:
            self.session_completion_queue: Any = mp_manager.Queue()
        else:
            self.session_completion_queue = None

        self.sessions: List[Optional[ReplaySession]] = []
        self._session_ids: List[str] = []
        self._session_id_to_index: Dict[str, int] = {}
        self.session_graph_state: Dict[str, ReplaySessionState] = {}
        # Per-session event lists keyed by session_index. Populated on demand by
        # _ensure_session_built (lazy path) or all at once by initialize_sessions (eager path).
        self._session_events: Dict[int, List[ReplaySessionEvent]] = {}
        # Flat list kept only for the eager initialize_sessions() path / back-compat.
        self.all_events: List[ReplaySessionEvent] = []
        self._skipped_session_count: int = 0

    def initialize_sessions(self, sessions: List[ReplaySession]) -> None:
        """Finalize generator state from fully-built sessions (eager path)."""
        # Duplicate sessions if needed to meet total session requirements
        if self.replay_config and self.replay_config.duplicate_sessions_target is not None:
            sessions = self._duplicate_sessions_if_needed(sessions, self.replay_config.duplicate_sessions_target)

        if not sessions:
            raise ValueError("No valid replay sessions found")

        # Assign session_index to match position in list after shuffling/duplication
        for i, session in enumerate(sessions):
            session.session_index = i

        self.sessions = list(sessions)
        self._build_replay_schedule()
        logger.info(
            "Built replay schedule: %d events across %d sessions (eager)",
            len(self.all_events),
            len(self.sessions),
        )

    def initialize_sessions_lazy(self, session_ids: List[str]) -> None:
        """Set up placeholder slots for on-demand graph building (lazy path).

        Allocates None session slots and records their IDs so get_session_count()
        works immediately. Each session's graph is built later by _ensure_session_built,
        triggered the first time the session is dispatched (parent) or replayed (worker).
        """
        if not session_ids:
            raise ValueError("No valid trace records found after filtering")
        self.sessions = [None] * len(session_ids)
        self._session_ids = list(session_ids)
        self._session_id_to_index = {sid: i for i, sid in enumerate(session_ids)}
        self._session_events = {}
        self.all_events = []
        logger.info("Lazy init: %d session slots allocated", len(session_ids))

    def _build_session(self, session_index: int) -> Optional[ReplaySession]:
        """Build the ReplaySession for one slot. Implemented by lazy subclasses."""
        raise NotImplementedError("Lazy generators must implement _build_session()")

    def _ensure_session_built(self, session_index: int) -> None:
        """Build and register session_index's graph if not already done. Idempotent."""
        if session_index < 0 or session_index >= len(self.sessions):
            raise IndexError(f"Session index {session_index} out of range (total: {len(self.sessions)})")
        if self.sessions[session_index] is not None or session_index in self._session_events:
            return
        session = self._build_session(session_index)
        if session is None:
            # No graph (e.g. malformed spans, or all calls errored with include_errors=False).
            # Register an empty event list as the "already attempted" sentinel so this slot is
            # not retried on subsequent calls. The dispatcher skips it (see is_session_buildable).
            self._session_events[session_index] = []
            self._skipped_session_count += 1
            return
        session.session_index = session_index
        self.sessions[session_index] = session
        events = self._build_session_schedule(session)
        self._session_events[session_index] = events
        logger.debug("Built session %s: %d events", session.session_id, len(events))

    def is_session_buildable(self, session_index: int) -> bool:
        """Return True if session_index has (or can build) a graph, False if it produced none.

        Builds the session on demand (idempotent). Lets the dispatcher skip un-buildable
        slots without raising. Skipped sessions are not reported anywhere (see dispatch_session).
        """
        self._ensure_session_built(session_index)
        session = self.sessions[session_index]
        if session is None:
            logger.warning(f"Skipping session {session_index} ({self._session_ids[session_index]!r}): no graph could be built")
            return False
        events = self._session_events.get(session_index, [])
        if not events:
            logger.warning(
                f"Skipping session {session_index} ({self._session_ids[session_index]!r}): graph built but no schedulable events"
            )
            self._skipped_session_count += 1
            return False
        logger.info("Dispatching session %s: %d events", session.session_id, len(events))
        return True

    @staticmethod
    def _duplicate_sessions_if_needed(sessions: List[ReplaySession], target_sessions: int) -> List[ReplaySession]:
        """Duplicate sessions (round-robin, unique IDs) to reach target_sessions
        when the trace corpus is smaller than needed for stress testing."""
        current_count = len(sessions)

        if current_count >= target_sessions:
            logger.info(f"Session corpus sufficient: {current_count} sessions available (target: {target_sessions})")
            return sessions

        duplicates_needed = target_sessions - current_count
        logger.warning(
            f"Session corpus small: {current_count} sessions available. "
            f"Duplicating to reach {target_sessions} sessions for stress testing."
        )

        original_sessions = list(sessions)
        duplicate_count = 0
        session_idx = 0

        while len(sessions) < target_sessions:
            source_session = original_sessions[session_idx % len(original_sessions)]
            session_idx += 1
            duplicate_count += 1

            # session_index reassigned in initialize_sessions() to match list position
            duplicate_session = ReplaySession(
                session_id=f"{source_session.session_id}_dup{duplicate_count}",
                source_id=source_session.source_id,
                session_index=-1,
                graph=source_session.graph,
                start_offset_ms=source_session.start_offset_ms,
            )

            sessions.append(duplicate_session)

        logger.info(f"Duplicated {duplicates_needed} sessions. Total sessions now: {len(sessions)}")
        return sessions

    @staticmethod
    def is_duplicate_session(session_id: str) -> bool:
        """Check if session_id matches the duplicate pattern {original_id}_dup{number}."""
        return bool(re.search(r"_dup\d+$", session_id))

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def is_preferred_worker_requested(self) -> bool:
        return True

    def _build_replay_schedule(self) -> None:
        """Build the full schedule from self.sessions (eager path / tests)."""
        self.all_events = []
        self._session_events = {}
        self._session_ids = [s.session_id for s in self.sessions if s is not None]
        self._session_id_to_index = {sid: i for i, sid in enumerate(self._session_ids)}
        for session in self.sessions:
            if session is None:
                continue
            events = self._build_session_schedule(session)
            self._session_events[session.session_index] = events
            self.all_events.extend(events)

    def _build_session_schedule(self, session: ReplaySession) -> List[ReplaySessionEvent]:
        """Register one session's graph state and return its qualified events.

        Builds the per-session ReplaySessionState (incl. the KV-cache-invalidation
        random_string, which is generated for duplicate sessions or when
        inject_random_session_id is set) and stores it in session_graph_state.
        """
        random_string = None
        is_duplicate = ReplayGraphSessionGeneratorBase.is_duplicate_session(session.session_id)
        if (self.replay_config and self.replay_config.inject_random_session_id) or is_duplicate:
            random_string = uuid.uuid4().hex[:16]
            logger.debug(f"Generated random string for session {session.session_id}: {random_string}")

        state = ReplaySessionState(
            session_id=session.session_id,
            graph=session.graph,
            ready_events=set(),
            dispatched_events=set(),
            completed_events=set(),
            event_completion_times={},
            is_active=False,
            is_complete=False,
            random_string=random_string,
        )
        self.session_graph_state[session.session_id] = state

        reuse_profiles = _compute_reuse_profiles(list(session.graph.events.values()))
        events: List[ReplaySessionEvent] = []
        for event in session.graph.events.values():
            gc = event.call

            if not gc.messages:
                logger.warning("Call %s in event %s has no messages, skipping", gc.call_id, event.event_id)
                continue

            qualified_event_id = f"{session.session_id}:{event.event_id}"
            qualified_predecessor_ids = [f"{session.session_id}:{pid}" for pid in event.predecessor_event_ids]
            qualified_segments = [
                dc_replace(seg, source_event_id=f"{session.session_id}:{seg.source_event_id}")
                if seg.source_event_id is not None
                else seg
                for seg in gc.input_segments
            ]

            events.append(
                ReplaySessionEvent(
                    call_id=gc.call_id,
                    event_id=qualified_event_id,
                    session_index=session.session_index,
                    t_start_ms=event.t_start_ms,
                    t_end_ms=event.t_end_ms,
                    model=gc.model,
                    messages=gc.messages,
                    expected_output=gc.expected_output,
                    input_segments=qualified_segments,
                    expected_output_tokens=gc.expected_output_tokens,
                    temperature=gc.temperature,
                    max_tokens_recorded=gc.max_tokens_recorded,
                    predecessor_event_ids=qualified_predecessor_ids,
                    wait_ms=min(event.wait_ms, self.replay_config.max_wait_ms) if self.replay_config else event.wait_ms,
                    tool_definitions=gc.tool_definitions,
                    reuse_depth_profile=reuse_profiles.get(event.event_id),
                )
            )
        return events

    def get_session_count(self) -> int:
        return len(self._session_ids)

    def _get_session(self, session_index: int) -> ReplaySession:
        """Return the built session at session_index, building it on demand if needed."""
        if session_index < 0 or session_index >= len(self._session_ids):
            raise IndexError(f"Session index {session_index} out of range (total: {len(self._session_ids)})")
        self._ensure_session_built(session_index)
        session = self.sessions[session_index]
        if session is None:
            raise RuntimeError(f"Session {session_index} ({self._session_ids[session_index]!r}) produced no graph")
        return session

    def get_session_event_indices(self, session_index: int) -> List[int]:
        self._ensure_session_built(session_index)
        return list(range(len(self._session_events.get(session_index, []))))

    def get_session_info(self, session_index: int) -> Dict[str, Any]:
        session = self._get_session(session_index)
        num_events = len(self._session_events.get(session_index, []))
        return {
            "session_id": session.session_id,
            "file_path": session.source_id,
            "source_id": session.source_id,
            "session_index": session.session_index,
            "num_events": num_events,
            "num_graph_events": len(session.graph.events),
            "start_offset_ms": session.start_offset_ms,
        }

    def get_session_events(self, session_index: int) -> List[LazyLoadInferenceAPIData]:
        session = self._get_session(session_index)
        events = self._session_events.get(session_index, [])
        session_worker_id = abs(hash(session.session_id)) % self.num_workers
        return [
            SessionReplayLazyLoadData(
                session_index=session_index,
                local_event_index=local,
                preferred_worker_id=session_worker_id,
            )
            for local in range(len(events))
        ]

    def build_session_metric(
        self,
        session_id: str,
        stage_id: int,
        start_time: float,
        end_time: float,
    ) -> SessionLifecycleMetric:
        state = self.session_graph_state.get(session_id)
        if state is None:
            raise ValueError(f"Unknown session: {session_id}")

        source_id = ""
        for session in self.sessions:
            if session is not None and session.session_id == session_id:
                source_id = session.source_id
                break

        session_index = self._session_id_to_index.get(session_id)
        num_events = (
            len(self._session_events[session_index])
            if session_index is not None and session_index in self._session_events
            else len(state.graph.events)
        )
        num_events_completed = len(state.completed_events)
        num_events_cancelled = state.cancelled_events if state.failed else 0

        error = None
        if state.failure_reason:
            error = ErrorResponseInfo(error_type="SessionReplayError", error_msg=state.failure_reason)

        return SessionLifecycleMetric(
            session_id=session_id,
            stage_id=stage_id,
            file_path=source_id,
            start_time=start_time,
            end_time=end_time,
            duration_sec=end_time - start_time,
            num_events=num_events,
            num_events_completed=num_events_completed,
            num_events_cancelled=num_events_cancelled,
            error=error,
            n_recorded_substitutions=state.n_recorded_substitutions,
            recorded_substitution_event_ids=state.recorded_substitution_event_ids,
        )

    def activate_session(self, session_id: str) -> None:
        state = self.session_graph_state.get(session_id)
        if state is None:
            logger.warning("Attempted to activate unknown session: %s", session_id)
            return

        state.is_active = True
        root_events = {event_id for event_id, event in state.graph.events.items() if not event.predecessor_event_ids}
        state.ready_events.update(root_events)
        logger.debug("Activated session %s with %d root events", session_id, len(root_events))

    def _process_completion_queue(self) -> None:
        if self.session_completion_queue is None:
            return

        try:
            while True:
                completion_data = self.session_completion_queue.get_nowait()
                completed_session_id = completion_data["session_id"]

                completed_state = self.session_graph_state.get(completed_session_id)
                if completed_state is not None:
                    event_times = completion_data.get("event_completion_times", {})
                    for event_id, completion_time in event_times.items():
                        if event_id not in completed_state.completed_events:
                            completed_state.completed_events.add(event_id)
                            completed_state.event_completion_times[event_id] = completion_time

                    completed_state.is_complete = True
                    completed_state.failed = completion_data.get("failed", False)
                    completed_state.failure_reason = completion_data.get("failure_reason")
                    completed_state.cancelled_events = completion_data.get("cancelled_events", 0)
                    # Bad tool-call handling telemetry. The two keys are
                    # gated worker-side behind `len(...) > 0`, so their
                    # absence here is meaningful (no substitution path
                    # exercised) and we propagate that absence to the
                    # session metric as None.
                    if "n_recorded_substitutions" in completion_data:
                        completed_state.n_recorded_substitutions = completion_data["n_recorded_substitutions"]
                        completed_state.recorded_substitution_event_ids = completion_data.get(
                            "recorded_substitution_event_ids", []
                        )
                    logger.debug(
                        "Session %s marked complete from queue notification (failed=%s)",
                        completed_session_id,
                        completed_state.failed,
                    )
        except Exception:
            pass

    def get_session_state(self, session_id: str) -> Optional[ReplaySessionState]:
        return self.session_graph_state.get(session_id)

    def check_session_completed(self, session_id: str) -> bool:
        self._process_completion_queue()

        state = self.session_graph_state.get(session_id)
        if state is None:
            logger.warning("Attempted to check unknown session: %s", session_id)
            return False

        if state.is_complete:
            return True

        shared_failed = getattr(self, "shared_failed_sessions", None)
        if shared_failed is not None:
            is_failed = session_id in shared_failed
            if is_failed:
                state.is_complete = True
                logger.info("Session %s marked as complete due to failure", session_id)
                return True

        return False

    def _resolve_event(self, data: LazyLoadInferenceAPIData) -> ReplaySessionEvent:
        """Resolve a queued lazy item to its ReplaySessionEvent.

        Per-session addressing (lazy path): build session_index on demand, index
        local_event_index. Legacy global addressing (eager path): index all_events.
        """
        if isinstance(data, SessionReplayLazyLoadData):
            self._ensure_session_built(data.session_index)
            events = self._session_events.get(data.session_index, [])
            if data.local_event_index >= len(events):
                raise IndexError(
                    f"Local event index {data.local_event_index} out of range for session {data.session_index} (total: {len(events)})"
                )
            return events[data.local_event_index]

        n = data.data_index
        if n < 0 or n >= len(self.all_events):
            raise IndexError(f"Event index {n} out of range (total: {len(self.all_events)})")
        return self.all_events[n]

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> SessionChatCompletionAPIData:
        event = self._resolve_event(data)

        chat_messages = []
        original_messages: List[Dict[str, Any]] = []
        for msg in event.messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls")
                tool_call_id = msg.get("tool_call_id")
                reasoning_content = msg.get("reasoning") or msg.get("reasoning_content")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "text", "")
                tool_calls = getattr(msg, "tool_calls", None)
                tool_call_id = getattr(msg, "tool_call_id", None)
                reasoning_content = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

            if tool_calls is not None:
                # Preserve any assistant preamble content alongside the tool calls
                # so chat_messages (which becomes the wire payload when no
                # substitution/injection runs) is not lossy.
                tc_content = content if isinstance(content, str) and content else None
                chat_messages.append(
                    ChatMessage(role=role, content=tc_content, tool_calls=tool_calls, reasoning_content=reasoning_content)
                )
                tc_msg: Dict[str, Any] = {"role": role, "tool_calls": tool_calls}
                if tc_content is not None:
                    tc_msg["content"] = tc_content
                if reasoning_content:
                    tc_msg["reasoning_content"] = reasoning_content
                original_messages.append(tc_msg)
                continue

            if isinstance(content, list):
                content_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            content_parts.append(block.get("text", ""))
                        else:
                            content_parts.append(json.dumps(block))
                    else:
                        content_parts.append(str(block))
                content = " ".join(content_parts)

            content_str = str(content)
            # Carry tool_call_id onto the ChatMessage too (not just original_messages),
            # so role:tool linkage survives on the wire when substitution is disabled.
            # tool_call_id belongs on a role:tool message only — don't leak it onto
            # user/assistant messages even if the recorded message carried one.
            wire_tool_call_id = tool_call_id if role == "tool" else None
            chat_messages.append(
                ChatMessage(
                    role=role, content=content_str, tool_call_id=wire_tool_call_id, reasoning_content=reasoning_content
                )
            )
            orig_msg: Dict[str, Any] = {"role": role, "content": content_str}
            if wire_tool_call_id is not None:
                orig_msg["tool_call_id"] = wire_tool_call_id
            if reasoning_content:
                orig_msg["reasoning_content"] = reasoning_content
            original_messages.append(orig_msg)

        max_tokens = event.expected_output_tokens
        session_id = event.event_id.split(":")[0] if ":" in event.event_id else event.event_id
        raw_event_id = event.event_id.split(":", 1)[1] if ":" in event.event_id else event.event_id
        state = self.session_graph_state.get(session_id)
        session_index = event.session_index
        total_events = len(self._session_events.get(session_index, [])) if state else 0

        gc = state.graph.events[raw_event_id].call if state and raw_event_id in state.graph.events else None

        retention_policy = getattr(self, "retention_policy", None)

        remaining_reuse = 0
        fwd_segs, fwd_covers = (), False
        if state and raw_event_id in state.graph.events:
            cache = getattr(self, "_reuse_count_cache", None)
            if cache is None:
                cache = {}
                self._reuse_count_cache = cache
            rc = cache.get(session_id)
            if rc is None:
                rc = _prefix_reuse_counts(list(state.graph.events.values()))
                cache[session_id] = rc
            remaining_reuse = rc.get(raw_event_id, 0)
            # LIVE-based forward reuse recomputed for this turn (decouples breadth
            # from in-flight coverage; see _forward_reuse_depths).
            completed_ids = dispatched_ids = None
            if retention_policy is not None:
                assert self.num_workers == 1, (
                    "Retention directive decoupling assumes single-worker "
                    "asyncio (num_workers==1); a multi-worker run needs a "
                    "thread-safe dispatched-set."
                )
                completed_ids, dispatched_ids = _lifecycle_sets(
                    state.graph.events, session_id, self.output_registry,
                    state.dispatched_events,
                )
            fd = _forward_reuse_depths(
                list(state.graph.events.values()),
                registry=getattr(self, "output_registry", None),
                target=raw_event_id,
                completed_ids=completed_ids,
                dispatched_ids=dispatched_ids,
            )
            fwd_segs, fwd_covers = fd.get(raw_event_id, ((), False))
            if retention_policy is not None:
                # Mark THIS turn dispatched AFTER computing (so it never counts
                # itself). Synchronous section, no await -> atomic under asyncio.
                state.dispatched_events.add(raw_event_id)

        return SessionChatCompletionAPIData(
            messages=chat_messages,
            max_tokens=max_tokens,
            tool_definitions=event.tool_definitions,
            event_id=event.event_id,
            remaining_reuse=remaining_reuse,
            forward_segments=fwd_segs,
            forward_covers_output=fwd_covers,
            registry=self.output_registry,
            worker_tracker=getattr(self, "worker_tracker", WorkerSessionTracker()),
            completion_queue=getattr(self, "session_completion_queue", None),
            total_events_in_session=total_events,
            predecessor_event_ids=event.predecessor_event_ids,
            wait_ms=event.wait_ms,
            input_segments=event.input_segments,
            original_messages=original_messages,
            expected_output_content=event.expected_output,
            expected_output_is_tool_call=gc.expected_output_is_tool_call if gc else False,
            expected_output_tool_names=gc.expected_output_tool_names if gc else None,
            otel_context=data.otel_context,
            session_id=data.session_id,
            preferred_worker_id=data.preferred_worker_id,
            # Pass KV-cache invalidation configuration and session random string
            inject_random_session_id=self.replay_config.inject_random_session_id if self.replay_config else False,
            session_random_string=state.random_string if state else None,
            override_tool_call_max_tokens=self.replay_config.override_tool_call_max_tokens if self.replay_config else False,
            # Mitigation knob: read once per event from replay_config. Default
            # NONE keeps the wire format byte-identical to upstream main.
            bad_tool_call_handling=getattr(self.replay_config, "bad_tool_call_handling", BadToolCallHandling.NONE)
            if self.replay_config
            else BadToolCallHandling.NONE,
            disable_output_substitution=getattr(self.replay_config, "disable_output_substitution", False)
            if self.replay_config
            else False,
            # Back-reference so the event can evict this session from the worker once drained.
            generator=self,
            # Retention policy (KV-cache retention directive injection)
            retention_policy=retention_policy,
            reuse_depth_profile=event.reuse_depth_profile,
        )

    def cleanup_session(self, session_id: str) -> None:
        state = self.session_graph_state.get(session_id)
        if state is None:
            logger.warning("Attempted to cleanup unknown session: %s", session_id)
            return

        event_count = len(state.graph.events)
        for event_id in state.graph.events.keys():
            qualified_event_id = f"{session_id}:{event_id}"
            self.output_registry._event_output_text.pop(qualified_event_id, None)
            self.output_registry._event_output_message.pop(qualified_event_id, None)
            self.output_registry._event_input_messages.pop(qualified_event_id, None)
            self.output_registry._event_signals.pop(qualified_event_id, None)
            self.output_registry._failed_event_ids.discard(qualified_event_id)

        del self.session_graph_state[session_id]

        # Free the lazily-built graph and event list so a completed session's memory is
        # released (it was only retained between dispatch and completion). Without this the
        # parent accumulates every dispatched session's graph for the whole run, which blows
        # up RAM for large corpora / high duplicate_sessions_target. Re-access (if it ever
        # happened) would rebuild on demand via _ensure_session_built.
        idx = self._session_id_to_index.get(session_id)
        if idx is not None:
            if idx < len(self.sessions):
                self.sessions[idx] = None
            self._session_events.pop(idx, None)

        logger.debug("Cleaned up session %s: removed %d events from memory", session_id, event_count)

    def evict_worker_session(self, session_id: str) -> None:
        """Free a fully-drained session's memory inside a worker process.

        The parent process frees sessions via cleanup_session in its dispatch loop, but
        workers never call cleanup_session — so in the lazy path each worker would retain
        every graph it ever built for the whole stage. This is called by the data object
        (_mark_drained_and_maybe_evict) when a session's last event drains on this worker,
        making per-worker memory track the concurrent working set instead of the full corpus.

        Reuses cleanup_session to drop the built graph, event list, traversal state, and
        output-registry entries, then clears the per-worker WorkerSessionTracker bookkeeping
        for the session (which cleanup_session does not touch).
        """
        self.cleanup_session(session_id)
        tracker = getattr(self, "worker_tracker", None)
        if tracker is not None:
            tracker.forget_session(session_id)
        logger.debug("Worker evicted fully-drained session %s", session_id)
