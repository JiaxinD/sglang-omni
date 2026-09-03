from __future__ import annotations

import pytest

from sglang_omni.pipeline.sm_cap import (
    BOOTSTRAP_ENV,
    RESERVED_ENV,
    SM_GROUP_SIZE,
    SmCapError,
    merged_ld_preload,
    resolve_bootstrap_path,
    sm_cap_env,
    stage_sm_cap_env,
    validate_capped_process,
)


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    library = tmp_path / "libgreen_ctx_bootstrap.so"
    library.write_bytes(b"")
    monkeypatch.setenv(BOOTSTRAP_ENV, str(library))
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", raising=False)
    for name in RESERVED_ENV:
        monkeypatch.delenv(name, raising=False)
    return str(library)


def _stage(**kwargs):
    from sglang_omni.config.schema import StageConfig

    return StageConfig(
        name=kwargs.pop("name", "vocoder"),
        factory_path="tests.fake:factory",
        **kwargs,
    )


def test_sm_cap_env_sets_group_count_and_preload(bootstrap):
    env = sm_cap_env(72, bootstrap)

    assert env["GREEN_CTX_SM"] == "72"
    assert env["GREEN_CTX_SPLIT"] == str(SM_GROUP_SIZE)
    assert env["GREEN_CTX_GROUP_COUNT"] == str(72 // SM_GROUP_SIZE)
    assert env["LD_PRELOAD"] == bootstrap
    assert env[BOOTSTRAP_ENV] == bootstrap


@pytest.mark.parametrize("sm_cap", [0, -8, 70, 1])
def test_sm_cap_env_rejects_non_multiples(sm_cap, bootstrap):
    with pytest.raises(SmCapError, match="multiple"):
        sm_cap_env(sm_cap, bootstrap)


def test_resolve_bootstrap_path_requires_configuration(monkeypatch):
    monkeypatch.delenv(BOOTSTRAP_ENV, raising=False)
    with pytest.raises(SmCapError, match="tools/green_ctx"):
        resolve_bootstrap_path()


def test_resolve_bootstrap_path_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv(BOOTSTRAP_ENV, str(tmp_path / "absent.so"))
    with pytest.raises(SmCapError, match="not found"):
        resolve_bootstrap_path()


def test_resolve_bootstrap_path_prefers_explicit_argument(tmp_path, bootstrap):
    explicit = tmp_path / "explicit.so"
    explicit.write_bytes(b"")

    assert resolve_bootstrap_path(str(explicit)) == str(explicit)


def test_inherited_ld_preload_is_kept_and_bootstrap_prepended():
    assert merged_ld_preload("/b.so", "/a.so") == "/b.so /a.so"
    assert merged_ld_preload("/b.so", None) == "/b.so"
    assert merged_ld_preload("/b.so", "/a.so /b.so") == "/a.so /b.so"


def test_stage_with_cap_preserves_parent_ld_preload(bootstrap, monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/opt/other.so")

    env = stage_sm_cap_env(_stage(sm_cap=72))

    assert env["LD_PRELOAD"] == f"{bootstrap} /opt/other.so"


def test_stage_without_cap_contributes_no_env(bootstrap):
    assert stage_sm_cap_env(_stage()) == {}


def test_stage_with_cap_contributes_bootstrap_env(bootstrap):
    env = stage_sm_cap_env(_stage(sm_cap=72))

    assert env["GREEN_CTX_SM"] == "72"
    assert env["LD_PRELOAD"] == bootstrap


@pytest.mark.parametrize("name", RESERVED_ENV)
def test_cap_rejects_derived_variables_set_by_hand(name, bootstrap):
    with pytest.raises(SmCapError, match="may not be set directly"):
        stage_sm_cap_env(_stage(sm_cap=72, env={name: "16"}))


@pytest.mark.parametrize("name", RESERVED_ENV)
def test_cap_rejects_derived_variables_inherited(name, bootstrap, monkeypatch):
    monkeypatch.setenv(name, "16")
    with pytest.raises(SmCapError, match="parent environment"):
        stage_sm_cap_env(_stage(sm_cap=72))


def test_cap_rejects_mps_active_thread_percentage(bootstrap, monkeypatch):
    """MPS may scale a client past the provisioned SMs, making the cap advisory."""
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "50")
    with pytest.raises(SmCapError, match="ACTIVE_THREAD_PERCENTAGE"):
        stage_sm_cap_env(_stage(sm_cap=72))


def test_stage_config_rejects_non_positive_cap():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _stage(sm_cap=0)


def test_colocated_stages_must_agree_on_the_cap():
    """A green context is process-wide, so an uncapped colocated stage is capped too."""
    with pytest.raises(SmCapError, match="disagree about sm_cap"):
        validate_capped_process(
            [_stage(name="engine"), _stage(name="vocoder", sm_cap=48)]
        )


def test_colocated_stages_with_the_same_cap_are_allowed():
    validate_capped_process(
        [_stage(name="engine", sm_cap=48), _stage(name="vocoder", sm_cap=48)]
    )


def test_stages_sharing_a_process_may_not_disagree_on_cap():
    """The spawn env layer is the last line of defence against two caps in one process."""
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
