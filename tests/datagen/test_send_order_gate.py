# Copyright 2026 The Kubernetes Authors.
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

"""Replaying a recorded send order, so an A/B differs only by policy.

The order two arms actually run in agrees on 6% of positions, and a next-use
priority is a claim about that order. The gate holds it fixed. Its failure mode
is a deadlock -- a recorded position that this run never reaches -- so most of
these tests are about the escapes, not the happy path.
"""

import asyncio
from pathlib import Path
from typing import List

from inference_perf.datagen.replay.replay_graph_session_datagen import WorkerSessionTracker


def _tracker(tmp_path: Path, keys: List[str], timeout_s: float = 20.0) -> WorkerSessionTracker:
    f = tmp_path / "order.txt"
    f.write_text("".join(f"123.4 {k}\n" for k in keys))
    t = WorkerSessionTracker()
    t.configure_send_order(str(f), timeout_s)
    return t


def test_untracked_event_is_not_held(tmp_path: Path) -> None:
    t = _tracker(tmp_path, ["a", "b"])
    asyncio.run(t.await_send_turn("zzz"))
    assert t.send_order_stats()["untracked"] == 1


def test_no_order_file_is_a_no_op() -> None:
    t = WorkerSessionTracker()
    t.configure_send_order(None, 20.0)
    asyncio.run(t.await_send_turn("anything"))
    assert t.send_order_stats() == {"violations": 0, "forced_positions": 0, "untracked": 0}


def test_later_position_waits_for_the_earlier_one(tmp_path: Path) -> None:
    t = _tracker(tmp_path, ["a", "b"])
    order = []

    async def main() -> None:
        async def send(k: str) -> None:
            await t.await_send_turn(k)
            order.append(k)

        later = asyncio.create_task(send("b"))
        await asyncio.sleep(0.05)
        assert order == [], "b went out before a"
        await send("a")
        await asyncio.wait_for(later, timeout=1.0)

    asyncio.run(main())
    assert order == ["a", "b"]
    assert t.send_order_stats()["violations"] == 0


def test_a_skipped_event_releases_the_queue_behind_it(tmp_path: Path) -> None:
    """A turn that is never sent must not block the rest of the run."""
    t = _tracker(tmp_path, ["a", "b"], timeout_s=30.0)

    async def main() -> None:
        async def send(k: str) -> None:
            await t.await_send_turn(k)

        later = asyncio.create_task(send("b"))
        await asyncio.sleep(0.05)
        t.release_send_slot("a")  # a failed and will never be sent
        await asyncio.wait_for(later, timeout=1.0)

    asyncio.run(main())
    assert t.send_order_stats()["violations"] == 0


def test_unclaimed_position_times_out_instead_of_deadlocking(tmp_path: Path) -> None:
    t = _tracker(tmp_path, ["a", "b"], timeout_s=0.1)

    async def main() -> None:
        await asyncio.wait_for(t.await_send_turn("b"), timeout=2.0)

    asyncio.run(main())
    s = t.send_order_stats()
    assert s["violations"] == 1
    assert s["forced_positions"] == 1


def test_many_waiters_all_drain_in_recorded_order(tmp_path: Path) -> None:
    keys = [f"e{i}" for i in range(40)]
    t = _tracker(tmp_path, keys, timeout_s=5.0)
    got = []

    async def main() -> None:
        async def send(k: str) -> None:
            await t.await_send_turn(k)
            got.append(k)

        # dispatched in reverse, so only the gate can produce the right order
        tasks = [asyncio.create_task(send(k)) for k in reversed(keys)]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

    asyncio.run(main())
    assert got == keys
    assert t.send_order_stats()["violations"] == 0


def test_timeout_gives_up_one_position_not_the_whole_gap(tmp_path: Path) -> None:
    """Jumping straight to our own position releases every request in between
    at once -- the first version did that and honoured 31% of the order."""
    t = _tracker(tmp_path, ["a", "b", "c", "d"], timeout_s=0.05)

    async def main() -> None:
        await asyncio.wait_for(t.await_send_turn("d"), timeout=3.0)

    asyncio.run(main())
    s = t.send_order_stats()
    # three positions ahead of "d", each given up on its own timeout
    assert s["forced_positions"] == 3
    assert s["violations"] == 3


def test_a_late_arrival_still_holds_the_queue_while_it_can(tmp_path: Path) -> None:
    """Order is honoured when the earlier turn shows up before the timeout."""
    t = _tracker(tmp_path, ["a", "b"], timeout_s=5.0)
    order = []

    async def main() -> None:
        async def send(k: str) -> None:
            await t.await_send_turn(k)
            order.append(k)

        later = asyncio.create_task(send("b"))
        await asyncio.sleep(0.2)  # well under the timeout
        await send("a")
        await asyncio.wait_for(later, timeout=1.0)

    asyncio.run(main())
    assert order == ["a", "b"]
    assert t.send_order_stats()["violations"] == 0


def test_a_drained_event_releases_its_slot_on_every_terminal_path(tmp_path: Path) -> None:
    """The gate can only wait indefinitely if dead positions free themselves.

    record_event_drained is the one path every terminal state goes through
    (completion, skip, request failure), so hooking it there is what makes an
    unbounded wait safe instead of a deadlock.
    """
    t = _tracker(tmp_path, ["a", "b"], timeout_s=3600.0)
    order = []

    async def main() -> None:
        async def send(k: str) -> None:
            await t.await_send_turn(k)
            order.append(k)

        later = asyncio.create_task(send("b"))
        await asyncio.sleep(0.05)
        assert order == []
        t.record_event_drained("s1", "a")  # "a" failed and will never be sent
        await asyncio.wait_for(later, timeout=1.0)

    asyncio.run(main())
    assert order == ["b"]
    assert t.send_order_stats()["violations"] == 0


def test_an_already_sent_event_draining_does_not_rewind_the_gate(tmp_path: Path) -> None:
    t = _tracker(tmp_path, ["a", "b", "c"], timeout_s=3600.0)

    async def main() -> None:
        await t.await_send_turn("a")
        await t.await_send_turn("b")
        t.record_event_drained("s1", "a")  # normal completion, slot long gone
        await asyncio.wait_for(t.await_send_turn("c"), timeout=1.0)

    asyncio.run(main())
    assert t.send_order_stats()["violations"] == 0
