from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni_router.update_journal import STATE_DIR_ENV


@pytest.fixture(autouse=True)
def isolate_router_state_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Keep the default journal path out of the developer's real home.

    The router resolves it under $XDG_STATE_HOME or ~/.local/state, so a test
    that triggers a weight update would otherwise write there.
    """
    state_dir = tmp_path_factory.mktemp("router-state")
    monkeypatch.setenv(STATE_DIR_ENV, str(state_dir))
    return state_dir
