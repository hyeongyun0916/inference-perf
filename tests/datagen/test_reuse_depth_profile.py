from types import SimpleNamespace
from inference_perf.datagen.replay_graph_session_datagen import _compute_reuse_profiles


def _seg(src, toks, typ="shared"):
    return SimpleNamespace(type=typ, message_count=1, token_count=toks, source_event_id=src)


def _ev(eid, t_start, t_end, segs, preds):
    call = SimpleNamespace(input_segments=segs)
    return SimpleNamespace(event_id=eid, call=call, predecessor_event_ids=preds,
                           t_start_ms=t_start, t_end_ms=t_end)


def test_profile_breadth_and_gap():
    S = _ev("S", 0, 2000, [], [])
    C1 = _ev("C1", 9000, 13000, [_seg("S", 175)], ["S"])
    C2 = _ev("C2", 19000, 23000, [_seg("S", 175)], ["S"])
    C3 = _ev("C3", 24000, 26000, [_seg("S", 400)], ["S"])
    profiles = _compute_reuse_profiles([S, C1, C2, C3])
    segs = profiles["S"]
    assert [(s.start, s.end, s.breadth) for s in segs] == [(0, 175, 3), (175, 400, 1)]


def test_terminal_producer_has_no_profile():
    S = _ev("S", 0, 2000, [], [])
    C = _ev("C", 5000, 7000, [], [])
    profiles = _compute_reuse_profiles([S, C])
    assert "S" not in profiles


def test_unique_only_consumer_yields_no_profile():
    S = _ev("S", 0, 2000, [], [])
    C = _ev("C", 5000, 7000, [_seg(None, 50, typ="unique")], [])
    profiles = _compute_reuse_profiles([S, C])
    assert "S" not in profiles and "C" not in profiles


def test_only_leading_contiguous_prefix_is_credited():
    # Consumer C reuses A as a leading shared+output run (contiguous, 100t),
    # then has its own `unique` content, then injects B's output AFTER the break.
    # Only A must be credited; B must NOT (it's past the prefix-cache break).
    A = _ev("A", 0, 1000, [], [])
    B = _ev("B", 0, 1000, [], [])
    C = _ev("C", 5000, 7000, [
        _seg("A", 60, typ="shared"),
        _seg("A", 40, typ="output"),
        _seg(None, 30, typ="unique"),
        _seg("B", 50, typ="output"),
    ], ["A", "B"])
    profiles = _compute_reuse_profiles([A, B, C])
    assert "A" in profiles
    assert profiles["A"][0].end == 100  # 60 shared + 40 output (leading run)
    assert "B" not in profiles          # injected after the unique break → not prefix-hittable


def test_intervening_spans_counted():
    # S produces at [0,1000]; X,Y run during the idle gap; C reuses S at t=6000.
    # intervening = events with t_start in (S.t_end=1000, C.t_start=6000) = X,Y = 2.
    S = _ev("S", 0, 1000, [], [])
    X = _ev("X", 2000, 3000, [_seg(None, 10, typ="unique")], [])
    Y = _ev("Y", 4000, 5000, [_seg(None, 10, typ="unique")], [])
    C = _ev("C", 6000, 7000, [_seg("S", 100)], ["S"])
    profiles = _compute_reuse_profiles([S, X, Y, C])
    assert profiles["S"][0].intervening_spans == 2
    assert profiles["S"][0].end == 100
