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

"""Capping outstanding requests, so the server's waiting queue stays shallow.

Queue depth turned out to set the hit rate: on the same workload and cache,
130 deep read 61.84% and 74 deep read 68.28%, because the depth is the gap
between a prefix being written and the turn that reuses it being scheduled.
The cap is not concurrent_sessions -- every session stays live, so the sharing
between them is unchanged; only how many of their turns are outstanding is.
"""

import asyncio

from inference_perf.datagen.replay.replay_graph_session_datagen import WorkerSessionTracker


def test_no_cap_configured_never_blocks() -> None:
    t = WorkerSessionTracker()
    t.configure_inflight_cap(None)

    async def main() -> bool:
        async with t.inflight_slot():
            async with t.inflight_slot():
                return True

    assert asyncio.run(main()) is True


def test_cap_limits_how_many_run_at_once() -> None:
    t = WorkerSessionTracker()
    t.configure_inflight_cap(2)
    peak = 0

    async def main() -> None:
        nonlocal peak
        live = 0

        async def one() -> None:
            nonlocal live, peak
            async with t.inflight_slot():
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0.02)
                live -= 1

        await asyncio.gather(*(one() for _ in range(8)))

    asyncio.run(main())
    assert peak == 2


def test_the_slot_is_released_even_when_the_request_raises() -> None:
    t = WorkerSessionTracker()
    t.configure_inflight_cap(1)

    async def main() -> None:
        try:
            async with t.inflight_slot():
                raise RuntimeError("request blew up")
        except RuntimeError:
            pass

        async def reenter() -> None:
            async with t.inflight_slot():
                pass

        # a leaked slot would hang here forever
        await asyncio.wait_for(reenter(), timeout=1.0)

    asyncio.run(main())
