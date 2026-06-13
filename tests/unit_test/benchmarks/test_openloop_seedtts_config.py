# SPDX-License-Identifier: Apache-2.0
"""SeedTTS harness plumbs the open-loop load fields through config/CLI/results."""

from __future__ import annotations

from benchmarks.eval import benchmark_tts_seedtts as B


def test_results_config_carries_open_loop_fields() -> None:
    cfg = B.TtsSeedttsBenchmarkConfig(
        model="m",
        meta="x",
        load_mode="openloop_poisson",
        arrival_seed=7,
        max_inflight_guard=64,
    )
    rc = B._build_results_config(cfg, base_url="http://x")
    assert rc["load_mode"] == "openloop_poisson"
    assert rc["arrival_seed"] == 7
    assert rc["max_inflight_guard"] == 64


def test_cli_args_thread_into_config() -> None:
    parser = B._build_arg_parser()
    args = parser.parse_args(
        [
            "--model",
            "m",
            "--meta",
            "x",
            "--load-mode",
            "openloop_fixed",
            "--arrival-seed",
            "9",
            "--max-inflight-guard",
            "48",
        ]
    )
    cfg = B._config_from_args(args)
    assert cfg.load_mode == "openloop_fixed"
    assert cfg.arrival_seed == 9
    assert cfg.max_inflight_guard == 48


def test_load_mode_defaults_to_closed_loop() -> None:
    parser = B._build_arg_parser()
    args = parser.parse_args(["--model", "m", "--meta", "x"])
    cfg = B._config_from_args(args)
    assert cfg.load_mode == "closed_loop"
    assert cfg.max_inflight_guard is None
