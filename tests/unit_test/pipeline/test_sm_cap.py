from __future__ import annotations

import pytest

from sglang_omni.pipeline.sm_cap import (
    BOOTSTRAP_ENV,
    SM_GROUP_SIZE,
    SmCapError,
    resolve_bootstrap_path,
    sm_cap_env,
)


def test_sm_cap_env_sets_group_count_and_preload(tmp_path):
    bootstrap = tmp_path / "libgreen_ctx_bootstrap.so"
    bootstrap.write_bytes(b"")

    env = sm_cap_env(72, str(bootstrap))

    assert env["GREEN_CTX_SM"] == "72"
    assert env["GREEN_CTX_SPLIT"] == str(SM_GROUP_SIZE)
    assert env["GREEN_CTX_GROUP_COUNT"] == str(72 // SM_GROUP_SIZE)
    assert env["LD_PRELOAD"] == str(bootstrap)
    assert env[BOOTSTRAP_ENV] == str(bootstrap)


@pytest.mark.parametrize("sm_cap", [0, -8, 70, 1])
def test_sm_cap_env_rejects_non_multiples(sm_cap, tmp_path):
    with pytest.raises(SmCapError, match="multiple"):
        sm_cap_env(sm_cap, str(tmp_path / "lib.so"))


def test_resolve_bootstrap_path_requires_configuration(monkeypatch):
    monkeypatch.delenv(BOOTSTRAP_ENV, raising=False)
    with pytest.raises(SmCapError, match="tools/green_ctx"):
        resolve_bootstrap_path()


def test_resolve_bootstrap_path_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv(BOOTSTRAP_ENV, str(tmp_path / "absent.so"))
    with pytest.raises(SmCapError, match="not found"):
        resolve_bootstrap_path()


def test_resolve_bootstrap_path_prefers_explicit_argument(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.so"
    explicit.write_bytes(b"")
    monkeypatch.setenv(BOOTSTRAP_ENV, str(tmp_path / "absent.so"))

    assert resolve_bootstrap_path(str(explicit)) == str(explicit)


def _stage(**kwargs):
    from sglang_omni.config.schema import StageConfig

    return StageConfig(
        name=kwargs.pop("name", "vocoder"),
        factory_path="tests.fake:factory",
        **kwargs,
    )


def test_stage_without_cap_contributes_no_env(monkeypatch):
    from sglang_omni.pipeline.sm_cap import stage_sm_cap_env

    monkeypatch.delenv("LD_PRELOAD", raising=False)
    assert stage_sm_cap_env(_stage()) == {}


def test_stage_with_cap_contributes_bootstrap_env(tmp_path, monkeypatch):
    from sglang_omni.pipeline.sm_cap import stage_sm_cap_env

    bootstrap = tmp_path / "libgreen_ctx_bootstrap.so"
    bootstrap.write_bytes(b"")
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.setenv(BOOTSTRAP_ENV, str(bootstrap))

    env = stage_sm_cap_env(_stage(sm_cap=72))

    assert env["GREEN_CTX_SM"] == "72"
    assert env["LD_PRELOAD"] == str(bootstrap)


def test_cap_rejects_inherited_ld_preload(tmp_path, monkeypatch):
    from sglang_omni.pipeline.sm_cap import stage_sm_cap_env

    bootstrap = tmp_path / "libgreen_ctx_bootstrap.so"
    bootstrap.write_bytes(b"")
    monkeypatch.setenv(BOOTSTRAP_ENV, str(bootstrap))
    monkeypatch.setenv("LD_PRELOAD", "/some/other.so")

    with pytest.raises(SmCapError, match="LD_PRELOAD"):
        stage_sm_cap_env(_stage(sm_cap=72))


def test_stage_config_rejects_non_positive_cap():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _stage(sm_cap=0)


def test_stages_sharing_a_process_may_not_disagree_on_cap():
    """Two caps in one OS process is a placement bug, not a silent last-wins."""
    from sglang_omni.pipeline.stage_workers import (
        StageLaunchConfig,
        StageWorkerProcessSpec,
        _patched_spawn_env,
    )

    spec = StageWorkerProcessSpec(
        process_name="shared",
        stage_specs=[
            StageLaunchConfig(stage_name="a", env_defaults={"GREEN_CTX_SM": "48"}),
            StageLaunchConfig(stage_name="b", env_defaults={"GREEN_CTX_SM": "72"}),
        ],
    )
    with pytest.raises(AssertionError, match="GREEN_CTX_SM"):
        with _patched_spawn_env(spec):
            pass
