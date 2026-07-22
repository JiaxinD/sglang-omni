# SPDX-License-Identifier: Apache-2.0
"""Gates for the MOSS-TTS-Local frame-chain fast path and the non-streaming
vocoder CUDA graphs: bit-identity vs eager, capture safety, env kill switch,
and the capture-failure fuse. GPU tests need the real MOSS-Audio-Tokenizer-v2.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types

import pytest
import torch

CODEC_GLOB = (
    "/root/.cache/huggingface/hub/"
    "models--OpenMOSS-Team--MOSS-Audio-Tokenizer-v2/snapshots/*/"
)
N_VQ = 12
_HAS_CUDA = torch.cuda.is_available()

pytest.importorskip("sglang")

from sglang_omni.models.moss_tts_local.model_runner import (  # noqa: E402
    MossTTSLocalModelRunner,
)
from sglang_omni.models.moss_tts_local.state_pool import (  # noqa: E402
    MossTTSLocalDecodeStatePool,
)
from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (  # noqa: E402
    MOSSL_FRAME_GRAPH_ENV,
    MossNonstreamVocoderGraphRunner,
    mossl_frame_graph_enabled,
)


# ---------------------------------------------------------------------------
# Env switch
# ---------------------------------------------------------------------------


def test_env_switch_parsing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(MOSSL_FRAME_GRAPH_ENV, raising=False)
    assert mossl_frame_graph_enabled() is True
    for off in ("0", "false", "off", " 0 ", "FALSE"):
        monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, off)
        assert mossl_frame_graph_enabled() is False
    for on in ("1", "true", "on", ""):
        monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, on)
        assert mossl_frame_graph_enabled() is True


# ---------------------------------------------------------------------------
# CPU harness for the model-runner frame decode (mirrors test_pipeline stubs)
# ---------------------------------------------------------------------------

_HIDDEN = 4
_SLOT_ID = 1000
_END_ID = 1001


def _make_runner(batch_size: int):
    weight = torch.zeros(max(batch_size, 2), _HIDDEN, dtype=torch.bfloat16)
    model = types.SimpleNamespace(
        _decode_input_embedding=types.SimpleNamespace(weight=weight),
        _state_pool=None,
        config=types.SimpleNamespace(
            n_vq=N_VQ,
            audio_assistant_slot_token_id=_SLOT_ID,
            audio_end_token_id=_END_ID,
        ),
        frame_graph_max_bs=0,  # eager frame path
        device=torch.device("cpu"),
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    model.acquire_row = pool.acquire_row

    def decode_frame(hidden, *, sample_text, sample_audio):
        del sample_text, sample_audio
        bs = hidden.shape[0]
        # Deterministic, row-distinct codes; row i stops iff its hidden row is 1.
        stops = hidden[:, 0].long().clamp(0, 1)
        codes = (
            torch.arange(bs, dtype=torch.long).view(bs, 1)
            + torch.arange(N_VQ, dtype=torch.long).view(1, N_VQ)
        ) % 1024
        return stops, codes

    model.decode_frame = decode_frame
    model._prepare_multi_modal_inputs = lambda rows: (
        rows.sum(dim=-1, keepdim=True).to(torch.bfloat16).expand(-1, _HIDDEN) + 3
    ).contiguous()
    runner = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
    runner.model = model
    runner._outbox = None
    return runner


def _sched_req(rid: str, *, penalty: float = 1.0, chunked: int = 0):
    data = types.SimpleNamespace(
        req=types.SimpleNamespace(is_chunked=chunked),
        text_temperature=1.0,
        text_top_p=1.0,
        text_top_k=50,
        audio_temperature=1.0,
        audio_top_p=1.0,
        audio_top_k=50,
        sampling_seed=0,
        generation_steps=0,
        sampling_steps=None,
        audio_repetition_penalty=penalty,
        output_rows=[],
    )
    return types.SimpleNamespace(request_id=rid, data=data)


def _result(batch_size: int, stop_rows: set[int] = frozenset()):
    hidden = torch.zeros(batch_size, _HIDDEN)
    for row in stop_rows:
        hidden[row, 0] = 1.0
    return types.SimpleNamespace(
        logits_output=types.SimpleNamespace(hidden_states=hidden)
    )


def _run_scenario(batch_size, *, stop_rows=frozenset(), penalties=None, chunked=None):
    runner = _make_runner(batch_size)
    penalties = penalties or [1.0] * batch_size
    chunked = chunked or [0] * batch_size
    reqs = [
        _sched_req(f"r{i}", penalty=penalties[i], chunked=chunked[i])
        for i in range(batch_size)
    ]
    result = _result(batch_size, stop_rows)
    rows, end_id = runner._run_frame_decode(result, types.SimpleNamespace(), reqs)
    pool = runner.model._state_pool
    journal = getattr(result, "moss_journal", None)
    return {
        "rows": rows,
        "end_id": end_id,
        "journal_rids": journal.rids if journal is not None else None,
        "journal_pool_rows": journal.pool_rows if journal is not None else None,
        "journal_rows": journal.rows.clone() if journal is not None else None,
        "sampling_steps": pool.sampling_steps.clone(),
        "feedback_embeds": pool.feedback_embeds.clone(),
        "audio_token_presence": pool.audio_token_presence.clone(),
    }


_SCENARIOS = {
    "all_continue": {},
    "stop_rows": {"stop_rows": {0}},
    "rep_penalty_on": {"penalties_first": 1.3},
    "rep_penalty_stop_mix": {"stop_rows": {0}, "penalties_first": 1.3},
}


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_fast_path_bit_identical_to_legacy(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, scenario: str
):
    """The all-emit fast path (env on) must be bit-identical to the legacy
    index_select path (env off): published rows, journal, and pool state."""
    spec = _SCENARIOS[scenario]
    kwargs: dict = {"stop_rows": spec.get("stop_rows", frozenset())}
    if "penalties_first" in spec:
        kwargs["penalties"] = [spec["penalties_first"]] + [1.0] * (batch_size - 1)

    monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, "0")
    legacy = _run_scenario(batch_size, **kwargs)
    monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, "1")
    fast = _run_scenario(batch_size, **kwargs)

    assert torch.equal(legacy["rows"], fast["rows"])
    assert legacy["end_id"] == fast["end_id"]
    assert legacy["journal_rids"] == fast["journal_rids"]
    assert legacy["journal_pool_rows"] == fast["journal_pool_rows"]
    assert torch.equal(legacy["journal_rows"], fast["journal_rows"])
    assert torch.equal(legacy["sampling_steps"], fast["sampling_steps"])
    assert torch.equal(legacy["feedback_embeds"], fast["feedback_embeds"])
    assert torch.equal(
        legacy["audio_token_presence"], fast["audio_token_presence"]
    )


@pytest.mark.parametrize("batch_size", [2, 4, 8, 16])
def test_mixed_chunked_batch_matches_legacy(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
):
    """A batch with a non-final chunked row (emit/no-emit mix) must keep the
    legacy per-index behavior under the fast-path env: identical journal
    (chunked row excluded) and pool state."""
    chunked = [0] * batch_size
    chunked[-1] = 2
    monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, "0")
    legacy = _run_scenario(batch_size, chunked=chunked)
    monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, "1")
    fast = _run_scenario(batch_size, chunked=chunked)

    assert legacy["journal_rids"] == fast["journal_rids"]
    assert f"r{batch_size - 1}" not in fast["journal_rids"]
    assert torch.equal(legacy["journal_rows"], fast["journal_rows"])
    assert torch.equal(legacy["sampling_steps"], fast["sampling_steps"])
    assert torch.equal(legacy["feedback_embeds"], fast["feedback_embeds"])
    assert torch.equal(legacy["rows"], fast["rows"])


def test_all_emit_path_builds_no_host_index_tensor(monkeypatch: pytest.MonkeyPatch):
    """The steady-state decode step (every row emits) must not build an index
    tensor from a host list: torch.tensor(list, device=...) is a pageable H2D
    copy that stream-syncs, measured at 24.6% of the serving loop."""
    monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, "1")
    batch_size = 4
    runner = _make_runner(batch_size)
    reqs = [_sched_req(f"r{i}") for i in range(batch_size)]
    pool = runner.model._state_pool
    row_t, pool_rows, has_penalty = pool.prepare_active_rows(reqs)
    forward_batch = types.SimpleNamespace(
        moss_pool_row_t=row_t,
        moss_pool_rows=pool_rows,
        moss_has_audio_repetition_penalty=has_penalty,
    )
    # Warm any lazy one-time caches; only the steady-state step is asserted on.
    runner._run_frame_decode(_result(batch_size), forward_batch, reqs)

    calls: list = []
    original_tensor = torch.tensor

    def counting_tensor(*args, **kwargs):
        calls.append((args, kwargs))
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", counting_tensor)
    runner._run_frame_decode(_result(batch_size), forward_batch, reqs)
    assert not calls, (
        "all-emit fast path must not call torch.tensor (host->device sync); "
        f"saw {len(calls)} call(s)"
    )


def test_mixed_chunked_batch_still_uses_index_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    """Sanity for the counter above: the mixed emit case keeps the legacy
    index-tensor path (so the fast-path counter is not vacuous)."""
    monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, "1")
    batch_size = 3
    runner = _make_runner(batch_size)
    reqs = [
        _sched_req("r0"),
        _sched_req("r1"),
        _sched_req("r2", chunked=1),
    ]
    pool = runner.model._state_pool
    row_t, pool_rows, has_penalty = pool.prepare_active_rows(reqs)
    forward_batch = types.SimpleNamespace(
        moss_pool_row_t=row_t,
        moss_pool_rows=pool_rows,
        moss_has_audio_repetition_penalty=has_penalty,
    )

    calls: list = []
    original_tensor = torch.tensor

    def counting_tensor(*args, **kwargs):
        calls.append((args, kwargs))
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", counting_tensor)
    runner._run_frame_decode(_result(batch_size), forward_batch, reqs)
    assert calls, "mixed batch should take the index-tensor fallback"


# ---------------------------------------------------------------------------
# Non-streaming vocoder CUDA graph (GPU + real codec)
# ---------------------------------------------------------------------------


def test_nonstream_capture_uses_thread_local_error_mode():
    source = textwrap.dedent(
        inspect.getsource(MossNonstreamVocoderGraphRunner._capture)
    )
    tree = ast.parse(source)
    graph_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "graph"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
    ]
    assert graph_calls, "nonstream vocoder CUDA graph capture call not found"
    assert any(
        keyword.arg == "capture_error_mode"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "thread_local"
        for call in graph_calls
        for keyword in call.keywords
    )


BATCH_BUCKETS = [1, 2, 4]
FRAME_BUCKETS = [12, 24]


@pytest.fixture(scope="module")
def codec_bundle():
    import glob

    from transformers import AutoModel

    from sglang_omni.models.moss_tts_local.vocoder_decoder import (
        MossTTSLocalVocoderDecoder,
    )

    snaps = glob.glob(CODEC_GLOB)
    if not snaps:
        pytest.skip("MOSS-Audio-Tokenizer-v2 codec snapshot not found")
    codec = (
        AutoModel.from_pretrained(snaps[0], trust_remote_code=True).to("cuda").eval()
    )
    nonstream_decoder = MossTTSLocalVocoderDecoder(codec.decoder)
    vocab = 1024
    return codec, nonstream_decoder, vocab


@pytest.fixture(scope="module")
def graph_runner(codec_bundle):
    codec, nonstream_decoder, _ = codec_bundle
    runner = MossNonstreamVocoderGraphRunner(
        codec,
        nonstream_decoder,
        n_vq=N_VQ,
        batch_buckets=BATCH_BUCKETS,
        frame_buckets=FRAME_BUCKETS,
    )
    runner.warmup()
    if not runner.captured_keys():
        pytest.skip("no nonstream vocoder graphs captured (low VRAM?)")
    return runner


def _eager_reference(codec, nonstream_decoder, codes_list):
    """The exact production eager path of _decode_codes_rows_nonstream."""
    device = next(codec.parameters()).device
    max_len = max(int(c.shape[1]) for c in codes_list)
    audio_codes = torch.zeros(
        N_VQ, len(codes_list), max_len, device=device, dtype=torch.long
    )
    padding_mask = torch.zeros(
        len(codes_list), max_len, device=device, dtype=torch.bool
    )
    for i, c in enumerate(codes_list):
        audio_codes[:, i, : c.shape[1]] = c
        padding_mask[i, : c.shape[1]] = True
    original = codec.decoder
    codec.decoder = nonstream_decoder
    try:
        with torch.no_grad():
            decoded = codec.decode(
                audio_codes,
                padding_mask=padding_mask,
                num_quantizers=N_VQ,
                return_dict=True,
                chunk_duration=None,
            )
    finally:
        codec.decoder = original
    audio_cpu = decoded.audio.detach().to("cpu", torch.float32)
    lengths_cpu = decoded.audio_lengths.detach().to("cpu")
    return [
        audio_cpu[i, :, : int(lengths_cpu[i])].contiguous()
        for i in range(len(codes_list))
    ]


def _graphed(runner, codes_list):
    device = runner._device
    max_len = max(int(c.shape[1]) for c in codes_list)
    padded = torch.zeros(
        N_VQ, len(codes_list), max_len, device=device, dtype=torch.long
    )
    for i, c in enumerate(codes_list):
        padded[:, i, : c.shape[1]] = c
    lengths = [int(c.shape[1]) for c in codes_list]
    out = runner.decode_padded(padded, lengths)
    if out is None:
        return None
    audio, audio_lengths = out
    audio_cpu = audio.detach().to("cpu", torch.float32)
    return [
        audio_cpu[i, :, : audio_lengths[i]].contiguous()
        for i in range(len(codes_list))
    ]


def _uniform_reference(codec, nonstream_decoder, codes_list, bucket):
    """Eager decode at the exact bucket geometry the graph replays: codes
    zero-padded to (B_bucket, T_bucket), all lengths pinned to T_bucket."""
    batch_bucket, frame_bucket = bucket
    device = next(codec.parameters()).device
    padded = torch.zeros(
        N_VQ, batch_bucket, frame_bucket, device=device, dtype=torch.long
    )
    for i, c in enumerate(codes_list):
        padded[:, i, : c.shape[1]] = c
    lengths = torch.full(
        (batch_bucket,), frame_bucket, device=device, dtype=torch.long
    )
    original = codec.decoder
    codec.decoder = nonstream_decoder
    try:
        with torch.no_grad(), nonstream_decoder.assume_full_lengths():
            result = codec._decode_frame(padded, lengths)
    finally:
        codec.decoder = original
    audio_cpu = result.audio.detach().to("cpu", torch.float32)
    return [
        audio_cpu[i, :, : int(c.shape[1]) * 3840].contiguous()
        for i, c in enumerate(codes_list)
    ]


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
@pytest.mark.parametrize(
    "lengths",
    [
        [12],  # exact frame bucket
        [7],  # tail-padded within a bucket
        [24, 17],  # ragged pair, exact batch bucket
        [12, 9, 5],  # ragged triple -> padded batch bucket (B=4)
        [24, 24, 24, 24],  # full bucket, uniform
    ],
    ids=["b1_exact", "b1_tailpad", "b2_ragged", "b3_padded_bucket", "b4_full"],
)
def test_nonstream_graph_bit_identical_to_same_geometry_eager(
    codec_bundle, graph_runner, lengths
):
    """Replay must reproduce the eager decode it replaces (same bucket
    geometry) bit-for-bit. Identity vs the RAGGED eager decode is not a
    coherent gate: the eager varlen decode's bits already depend on batch
    composition (see test_geometry_delta_bounded...)."""
    codec, nonstream_decoder, vocab = codec_bundle
    torch.manual_seed(sum(lengths))
    codes_list = [
        torch.randint(0, vocab, (N_VQ, t), device="cuda", dtype=torch.long)
        for t in lengths
    ]
    bucket = graph_runner.bucket_for(len(codes_list), max(lengths))
    assert bucket is not None, "expected a graph hit for this bucket"
    graphed = _graphed(graph_runner, codes_list)
    assert graphed is not None
    reference = _uniform_reference(codec, nonstream_decoder, codes_list, bucket)
    for i in range(len(lengths)):
        assert graphed[i].shape == reference[i].shape, (
            f"utterance {i}: graphed {tuple(graphed[i].shape)} != "
            f"reference {tuple(reference[i].shape)}"
        )
        assert torch.equal(graphed[i], reference[i]), (
            f"utterance {i} not bit-identical to same-geometry eager: "
            f"max|delta|={(graphed[i] - reference[i]).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_geometry_delta_bounded_and_ragged_eager_not_batch_invariant(
    codec_bundle, graph_runner
):
    """Two facts that justify the same-geometry identity gate above:
    (1) the graphed (bucket-padded) output stays within a small bound of the
    ragged eager output; (2) the ragged eager decode itself is NOT
    batch-invariant at the bit level (same utterance, different neighbor ->
    different bits), so bucket padding introduces no new error mode. If (2)
    ever becomes exactly invariant, revisit the gate."""
    codec, nonstream_decoder, vocab = codec_bundle
    torch.manual_seed(4141)
    c24 = torch.randint(0, vocab, (N_VQ, 24), device="cuda", dtype=torch.long)
    c17 = torch.randint(0, vocab, (N_VQ, 17), device="cuda", dtype=torch.long)

    graphed = _graphed(graph_runner, [c24, c17])
    assert graphed is not None
    ragged = _eager_reference(codec, nonstream_decoder, [c24, c17])
    graph_vs_ragged = max(
        (graphed[i] - ragged[i]).abs().max().item() for i in range(2)
    )

    solo = _eager_reference(codec, nonstream_decoder, [c17])[0]
    eager_vs_eager = (solo - ragged[1]).abs().max().item()

    assert graph_vs_ragged <= 0.1, (
        f"bucket-geometry delta unexpectedly large: {graph_vs_ragged:.3e}"
    )
    assert eager_vs_eager > 0.0, (
        "ragged eager decode became batch-invariant; the same-geometry "
        "identity gate should be revisited against full ragged identity"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_uniform_full_lengths_mode_bit_identical_to_default(codec_bundle):
    """assume_full_lengths (the capture-mode pack) must match the default packed
    path bit-for-bit when all lengths equal the dense length, and must restore
    the default path on exit."""
    codec, nonstream_decoder, vocab = codec_bundle
    torch.manual_seed(7)
    for batch_size, frames in ((1, 12), (3, 24)):
        codes_list = [
            torch.randint(0, vocab, (N_VQ, frames), device="cuda", dtype=torch.long)
            for _ in range(batch_size)
        ]
        baseline = _eager_reference(codec, nonstream_decoder, codes_list)
        with nonstream_decoder.assume_full_lengths():
            uniform = _eager_reference(codec, nonstream_decoder, codes_list)
        restored = _eager_reference(codec, nonstream_decoder, codes_list)
        for i in range(batch_size):
            assert torch.equal(uniform[i], baseline[i])
            assert torch.equal(restored[i], baseline[i])


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_oversize_shapes_fall_back_to_eager(graph_runner):
    device = graph_runner._device
    too_long = torch.zeros(N_VQ, 1, max(FRAME_BUCKETS) + 1, device=device).long()
    assert graph_runner.decode_padded(too_long, [too_long.shape[2]]) is None
    too_wide = torch.zeros(
        N_VQ, max(BATCH_BUCKETS) + 1, FRAME_BUCKETS[0], device=device
    ).long()
    assert (
        graph_runner.decode_padded(too_wide, [FRAME_BUCKETS[0]] * too_wide.shape[1])
        is None
    )


class _NoHostReadbackTensor(torch.Tensor):
    """Raises on any host readback so the replay dispatch is provably sync-free."""

    def item(self):
        raise AssertionError("host readback (.item) on replay dispatch")

    def cpu(self, *args, **kwargs):
        raise AssertionError("host readback (.cpu) on replay dispatch")

    def tolist(self):
        raise AssertionError("host readback (.tolist) on replay dispatch")

    def numpy(self, *args, **kwargs):
        raise AssertionError("host readback (.numpy) on replay dispatch")

    def __bool__(self):
        raise AssertionError("host readback (__bool__) on replay dispatch")

    def __int__(self):
        raise AssertionError("host readback (__int__) on replay dispatch")


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_no_host_readback_on_replay_dispatch(graph_runner):
    frames = FRAME_BUCKETS[0]
    codes = torch.zeros(
        N_VQ, 1, frames, device=graph_runner._device, dtype=torch.long
    ).as_subclass(_NoHostReadbackTensor)
    out = graph_runner.decode_padded(codes, [frames])
    assert out is not None


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_capture_failure_fuse_stops_after_max_failures(codec_bundle):
    codec, nonstream_decoder, _ = codec_bundle
    runner = MossNonstreamVocoderGraphRunner(
        codec,
        nonstream_decoder,
        n_vq=N_VQ,
        batch_buckets=[1, 2, 4],
        frame_buckets=[4, 8, 16],  # 9 keys > 8-failure fuse
    )
    attempts: list = []

    def boom(batch_bucket, frame_bucket):
        attempts.append((batch_bucket, frame_bucket))
        raise RuntimeError("simulated capture failure")

    runner._capture = boom
    runner.warmup()
    assert runner.captured_keys() == []
    assert len(attempts) == 8, "fuse must stop captures after 8 distinct failures"
    codes = torch.zeros(N_VQ, 1, 4, device=next(codec.parameters()).device).long()
    assert runner.decode_padded(codes, [4]) is None


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_capture_failure_resets_graph_pool(codec_bundle, monkeypatch):
    codec, nonstream_decoder, _ = codec_bundle
    runner = MossNonstreamVocoderGraphRunner(
        codec,
        nonstream_decoder,
        n_vq=N_VQ,
        batch_buckets=[1],
        frame_buckets=[4],
    )
    resets: list = []
    original_reset = torch.cuda.CUDAGraph.reset

    def spy_reset(self, *args, **kwargs):
        resets.append(self)
        return original_reset(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "reset", spy_reset)

    class _Boom(RuntimeError):
        pass

    original_frame = codec._decode_frame
    calls = {"n": 0}

    def failing_decode_frame(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:  # let warmup pass, fail inside capture
            raise _Boom("simulated capture failure")
        return original_frame(*args, **kwargs)

    monkeypatch.setattr(codec, "_decode_frame", failing_decode_frame)
    runner.warmup()
    assert runner.captured_keys() == []
    assert resets, "a failed capture must release its graph pool via reset()"


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_vram_guard_skips_capture(codec_bundle):
    codec, nonstream_decoder, _ = codec_bundle
    runner = MossNonstreamVocoderGraphRunner(
        codec,
        nonstream_decoder,
        n_vq=N_VQ,
        batch_buckets=[1],
        frame_buckets=[4],
        min_free_gb=100000.0,
    )
    runner.warmup()
    assert runner.captured_keys() == []


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_scheduler_kill_switch_and_replay_failure(
    codec_bundle, graph_runner, monkeypatch
):
    """Scheduler-level wiring: env off bypasses the runner entirely; a replay
    failure disables the runner and serves eager afterwards, bit-identical."""
    from sglang_omni.models.moss_tts_local.streaming_vocoder import (
        MossTTSLocalStreamingVocoderScheduler,
    )

    codec, nonstream_decoder, vocab = codec_bundle
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        codec,
        n_vq=N_VQ,
        sample_rate=48000,
        cuda_graph=False,  # no streaming-session graphs; nonstream runner injected
    )
    scheduler._nonstream_decoder = nonstream_decoder
    torch.manual_seed(11)
    rows_list = [
        torch.randint(0, vocab, (17, N_VQ), device="cuda", dtype=torch.long),
        torch.randint(0, vocab, (12, N_VQ), device="cuda", dtype=torch.long),
    ]
    eager_ref = scheduler._decode_codes_rows([r.clone() for r in rows_list])

    scheduler._nonstream_cg_runner = graph_runner

    # Env off: the runner must not be consulted.
    monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, "0")
    consulted: list = []
    original_decode_padded = graph_runner.decode_padded

    def spying_decode_padded(*args, **kwargs):
        consulted.append(1)
        return original_decode_padded(*args, **kwargs)

    monkeypatch.setattr(graph_runner, "decode_padded", spying_decode_padded)
    off_out = scheduler._decode_codes_rows([r.clone() for r in rows_list])
    assert not consulted, "kill switch must bypass the nonstream graph runner"
    for a, b in zip(off_out, eager_ref):
        assert torch.equal(a, b)

    # Env on: the runner is consulted; output shape matches and the value
    # stays within the bucket-geometry family (bit-identity vs ragged eager
    # is not defined; see the geometry-delta test).
    monkeypatch.setenv(MOSSL_FRAME_GRAPH_ENV, "1")
    on_out = scheduler._decode_codes_rows([r.clone() for r in rows_list])
    assert consulted, "env on must consult the nonstream graph runner"
    for a, b in zip(on_out, eager_ref):
        assert a.shape == b.shape
        assert (a - b).abs().max().item() <= 0.1

    # Replay failure: disables the runner, raises, then serves eager.
    def boom(*args, **kwargs):
        raise RuntimeError("simulated replay failure")

    monkeypatch.setattr(graph_runner, "decode_padded", boom)
    with pytest.raises(RuntimeError):
        scheduler._decode_codes_rows([r.clone() for r in rows_list])
    assert scheduler._nonstream_cg_runner is None
    after = scheduler._decode_codes_rows([r.clone() for r in rows_list])
    for a, b in zip(after, eager_ref):
        assert torch.equal(a, b)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
