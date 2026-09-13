# SPDX-License-Identifier: Apache-2.0
import threading

import pytest

from sglang_omni.profiler.work_units import (
    collect_execution_observations,
    current_execution_observations,
)


def test_scopes_restore_after_failure_and_do_not_cross_threads():
    assert current_execution_observations() is None
    with collect_execution_observations() as outer:
        outer.append({"path": "outer"})
        seen = []
        worker = threading.Thread(
            target=lambda: seen.append(current_execution_observations())
        )
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert seen == [None]
        with pytest.raises(RuntimeError):
            with collect_execution_observations() as inner:
                assert inner is not outer
                inner.append({"path": "failed"})
                raise RuntimeError("execution failed")
        assert current_execution_observations() is outer
        assert outer == [{"path": "outer"}]
    assert current_execution_observations() is None
