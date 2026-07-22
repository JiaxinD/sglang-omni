# SPDX-License-Identifier: Apache-2.0
"""Bit-identity and capture-safety gates for the Qwen3-TTS predictor CUDA graph.

The per-semantic-token code-predictor chain (lm_head + seeded sampling + codec
embedding + predictor stack, num_code_groups - 1 sub-iterations) is captured as
one CUDA graph per (batch bucket, sampling signature). Graphed output codes and
summed embeddings must equal the eager chain bit-for-bit (torch.equal) at
bucket-exact batch sizes, and the dispatch/replay path must never host-sync.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import sglang_omni.models.qwen3_tts.sglang_model as sglang_model_module
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.vendor.sglang.layers import RMSNorm

_HAS_CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _stub_qk_norm(monkeypatch: pytest.MonkeyPatch):
    # apply_qk_norm reads global server args (unset in unit tests); the norm
    # ops themselves are covered by the real RMSNorm layer norms.
    monkeypatch.setattr(sglang_model_module, "apply_qk_norm", lambda q, k, **_: (q, k))


HIDDEN = 8
NUM_HEADS = 2
NUM_KV_HEADS = 1
HEAD_DIM = 4
NUM_CODE_GROUPS = 4
PRED_VOCAB = 16
MAX_BS = 16
BUCKETS = (1, 2, 4, 8, 16)
DTYPE = torch.bfloat16


class _TupleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        return self.proj(hidden_states), None


class _IdentityRotary(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        fused_set_kv_buffer_arg=None,
    ):
        del positions, fused_set_kv_buffer_arg
        return q, k


def _build_talker(device: torch.device) -> Qwen3TTSTalker:
    torch.manual_seed(7)
    predictor_len = NUM_CODE_GROUPS + 1
    talker = object.__new__(Qwen3TTSTalker)
    talker.training = False
    talker.config = SimpleNamespace(
        num_code_groups=NUM_CODE_GROUPS,
        code_predictor_config=SimpleNamespace(
            vocab_size=PRED_VOCAB,
            hidden_size=HIDDEN,
        ),
    )
    talker._predictor_input_buffer = torch.zeros(
        MAX_BS, predictor_len, HIDDEN, device=device, dtype=DTYPE
    )
    positions = torch.arange(predictor_len, device=device, dtype=torch.long)
    talker._predictor_positions = positions
    talker._predictor_position_rows = (
        positions[:, None].expand(predictor_len, MAX_BS).contiguous()
    )
    talker._predictor_k_cache = torch.zeros(
        1, MAX_BS, NUM_KV_HEADS, predictor_len, HEAD_DIM, device=device, dtype=DTYPE
    )
    talker._predictor_v_cache = torch.zeros_like(talker._predictor_k_cache)
    talker._output_codes = torch.zeros(
        MAX_BS, NUM_CODE_GROUPS, dtype=torch.long, device=device
    )
    talker._output_embeds = torch.zeros(MAX_BS, HIDDEN, device=device, dtype=DTYPE)
    talker._sampled_token_ids = torch.zeros(MAX_BS, dtype=torch.long, device=device)

    talker._sub_batch_size = 0
    talker._sub_temperature_tensor = torch.full(
        (MAX_BS,), 0.9, device=device, dtype=torch.float32
    )
    talker._sub_top_p_tensor = torch.ones(MAX_BS, device=device, dtype=torch.float32)
    talker._sub_top_k_tensor = torch.full(
        (MAX_BS,), 50, device=device, dtype=torch.long
    )
    talker._semantic_sampling_seed_tensor = torch.zeros(
        MAX_BS, device=device, dtype=torch.long
    )
    talker._sub_sampling_seed_tensor = torch.zeros(
        MAX_BS, device=device, dtype=torch.long
    )
    talker._sub_sample_rows = []
    talker._sub_sample_row_indices_tensor = torch.zeros(
        MAX_BS, device=device, dtype=torch.long
    )
    talker._sub_sample_count = 0
    talker._sub_has_sampled_rows = False
    talker._sub_sampled_has_top_p = False
    talker._sub_sampled_max_top_k = 0
    talker._sub_sampled_has_unbounded_top_k = False

    layer = SimpleNamespace(
        input_layernorm=RMSNorm(HIDDEN, eps=1e-6).to(device, DTYPE),
        post_attention_layernorm=RMSNorm(HIDDEN, eps=1e-6).to(device, DTYPE),
        mlp=nn.Linear(HIDDEN, HIDDEN, bias=False).to(device, DTYPE),
    )
    layer.self_attn = SimpleNamespace(
        q_size=NUM_HEADS * HEAD_DIM,
        kv_size=NUM_KV_HEADS * HEAD_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        q_norm=RMSNorm(HEAD_DIM, eps=1e-6).to(device, DTYPE),
        k_norm=RMSNorm(HEAD_DIM, eps=1e-6).to(device, DTYPE),
        alt_stream=None,
        qkv_proj=_TupleLinear(HIDDEN, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM).to(
            device, DTYPE
        ),
        o_proj=_TupleLinear(NUM_HEADS * HEAD_DIM, HIDDEN).to(device, DTYPE),
        rotary_emb=_IdentityRotary(),
    )
    projection = nn.Linear(HIDDEN, HIDDEN, bias=True).to(device, DTYPE)
    talker.code_predictor = SimpleNamespace(
        model=SimpleNamespace(
            layers=[layer],
            norm=RMSNorm(HIDDEN, eps=1e-6).to(device, DTYPE),
            codec_embedding=nn.ModuleList(
                [
                    nn.Embedding(PRED_VOCAB, HIDDEN).to(device, DTYPE)
                    for _ in range(NUM_CODE_GROUPS - 1)
                ]
            ),
        ),
        lm_head=nn.ModuleList(
            [
                _TupleLinear(HIDDEN, PRED_VOCAB).to(device, DTYPE)
                for _ in range(NUM_CODE_GROUPS - 1)
            ]
        ),
        project_input=lambda hidden: projection(hidden),
    )
    layer0_embedding = nn.Embedding(PRED_VOCAB, HIDDEN).to(device, DTYPE)
    talker.get_input_embeddings = lambda: layer0_embedding

    talker._predictor_graphs = {}
    talker._predictor_graph_disabled = set()
    talker._predictor_graph_batch_sizes = BUCKETS
    talker._predictor_graph_enabled = True
    return talker


def _request(
    *,
    dosample: bool = True,
    temperature: float = 0.9,
    top_p: float = 1.0,
    top_k: int = 5,
    sub_seed: int = 1234,
    semantic_seed: int = 99,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            semantic_sampling_seed=semantic_seed,
            subtalker_dosample=dosample,
            subtalker_temperature=temperature,
            subtalker_top_p=top_p,
            subtalker_top_k=top_k,
            subtalker_sampling_seed=sub_seed,
        )
    )


def _uniform_requests(batch_size: int, **kwargs) -> list[SimpleNamespace]:
    return [
        _request(sub_seed=1000 + idx, semantic_seed=2000 + idx, **kwargs)
        for idx in range(batch_size)
    ]


def _step_inputs(batch_size: int, device: torch.device, *, step: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(31 * batch_size + step)
    layer0 = torch.randint(
        0, PRED_VOCAB, (batch_size, 1), generator=generator, dtype=torch.long
    ).to(device)
    hidden = torch.randn(
        batch_size, 1, HIDDEN, generator=generator, dtype=torch.float32
    ).to(device, DTYPE)
    positions = torch.arange(
        step * 3, step * 3 + batch_size, device=device, dtype=torch.long
    )
    return layer0, hidden, positions


def _run_eager(talker, layer0, hidden, positions):
    with torch.no_grad():
        codes, embeds = talker._code_predictor_forward_incremental(
            layer0, hidden, semantic_positions=positions
        )
    return codes.detach().clone(), embeds.detach().clone()


def _run_forward(talker, layer0, hidden, positions):
    with torch.no_grad():
        codes, embeds = talker.code_predictor_forward(
            layer0, hidden, semantic_positions=positions
        )
        torch.cuda.synchronize()
    return codes.detach().clone(), embeds.detach().clone()


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "sampling_kwargs",
    [
        {"top_k": 5, "top_p": 1.0},
        {"top_k": 5, "top_p": 0.9},
        {"top_k": 0, "top_p": 1.0},
    ],
    ids=["topk", "topk-topp", "fullsort"],
)
def test_graph_bit_identity_sampled(batch_size: int, sampling_kwargs: dict):
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(batch_size, **sampling_kwargs))
    layer0, hidden, positions = _step_inputs(batch_size, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert talker._predictor_graphs, "no predictor graph captured"
    assert torch.equal(graph_codes, eager_codes), (
        f"codes not bit-identical (bs={batch_size}): "
        f"mismatches={(graph_codes != eager_codes).sum().item()}"
    )
    assert torch.equal(graph_embeds, eager_embeds), (
        f"summed embeddings not bit-identical (bs={batch_size}): "
        f"max|delta|={(graph_embeds - eager_embeds).abs().max().item():.3e}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_graph_bit_identity_argmax(batch_size: int):
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(batch_size, dosample=False))
    layer0, hidden, positions = _step_inputs(batch_size, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert talker._predictor_graphs
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
def test_graph_bit_identity_argmax_none_positions():
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2, dosample=False))
    layer0, hidden, _ = _step_inputs(2, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, None)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, None)

    assert talker._predictor_graphs
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
def test_graph_padded_bucket_bit_identity():
    """Live bs=3 replays through the bucket-4 graph with padded rows."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(3))
    layer0, hidden, positions = _step_inputs(3, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert any(key[0] == 4 for key in talker._predictor_graphs)
    assert not any(key[0] == 3 for key in talker._predictor_graphs)
    assert graph_codes.shape == eager_codes.shape
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
def test_graph_multi_step_replay_bit_identity():
    """Consecutive steps reuse one captured graph and stay bit-identical."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(4))

    for step in range(3):
        layer0, hidden, positions = _step_inputs(4, device, step=step)
        eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
        graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)
        assert torch.equal(graph_codes, eager_codes), f"step={step}"
        assert torch.equal(graph_embeds, eager_embeds), f"step={step}"

    assert len(talker._predictor_graphs) == 1


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
def test_mixed_sampled_argmax_rows_fall_back_to_eager():
    device = torch.device("cuda")
    talker = _build_talker(device)
    requests = [_request(dosample=True), _request(dosample=False)]
    talker.prepare_decode_buffers(requests)
    layer0, hidden, positions = _step_inputs(2, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert not talker._predictor_graphs, "mixed batches must not capture a graph"
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
def test_kill_switch_disables_graph_path():
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker._predictor_graph_enabled = False
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert not talker._predictor_graphs
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


def test_env_switch_parsing(monkeypatch: pytest.MonkeyPatch):
    env = sglang_model_module.QTTS_PREDICTOR_GRAPH_ENV
    monkeypatch.delenv(env, raising=False)
    assert sglang_model_module._predictor_graph_env_enabled() is True
    monkeypatch.setenv(env, "0")
    assert sglang_model_module._predictor_graph_env_enabled() is False
    monkeypatch.setenv(env, "false")
    assert sglang_model_module._predictor_graph_env_enabled() is False
    monkeypatch.setenv(env, "1")
    assert sglang_model_module._predictor_graph_env_enabled() is True


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
def test_capture_failure_disables_key_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)

    calls = []

    class _BoomGraph:
        def __init__(self, *args, **kwargs) -> None:
            calls.append(1)
            raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(sglang_model_module, "_PredictorDecodeGraph", _BoomGraph)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert len(calls) == 1
    assert talker._predictor_graph_disabled, "failed key must be disabled"
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)

    _run_forward(talker, layer0, hidden, positions)
    assert len(calls) == 1, "disabled key must not retry capture"


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
def test_capture_failure_restores_live_sub_state(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)

    class _BoomGraph:
        def __init__(self, model, bucket_size, signature, **kwargs) -> None:
            del kwargs
            with model._predictor_graph_capture_state(bucket_size, signature):
                raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(sglang_model_module, "_PredictorDecodeGraph", _BoomGraph)
    _run_forward(talker, layer0, hidden, positions)

    assert talker._sub_batch_size == 2
    assert talker._sub_sample_count == 2
    assert talker._sub_sample_rows == [0, 1]


class _NoHostReadbackTensor(torch.Tensor):
    """Tensor whose host-materialization entry points fail the test."""

    def cpu(self, *args, **kwargs):
        raise RuntimeError("host readback (cpu) on the predictor graph path")

    def tolist(self):
        raise RuntimeError("host readback (tolist) on the predictor graph path")

    def numpy(self, *args, **kwargs):
        raise RuntimeError("host readback (numpy) on the predictor graph path")

    def item(self):
        raise RuntimeError("host readback (item) on the predictor graph path")

    def __float__(self):
        raise RuntimeError("host readback (float) on the predictor graph path")

    def __int__(self):
        raise RuntimeError("host readback (int) on the predictor graph path")

    def __bool__(self):
        raise RuntimeError("host readback (bool) on the predictor graph path")

    def __iter__(self):
        raise RuntimeError("host readback (iter) on the predictor graph path")

    def to(self, *args, **kwargs):
        if any(str(a) == "cpu" for a in args) or str(kwargs.get("device")) == "cpu":
            raise RuntimeError("host readback (to cpu) on the predictor graph path")
        return super().to(*args, **kwargs)


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
def test_no_host_readback_on_graph_dispatch_and_replay():
    """Live per-step inputs must reach the graph via device-side copies only."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)
    guarded_layer0 = layer0.as_subclass(_NoHostReadbackTensor)
    guarded_hidden = hidden.as_subclass(_NoHostReadbackTensor)
    guarded_positions = positions.as_subclass(_NoHostReadbackTensor)

    with torch.no_grad():
        talker.code_predictor_forward(
            guarded_layer0, guarded_hidden, semantic_positions=guarded_positions
        )
        assert talker._predictor_graphs, "expected graph capture"
        talker.code_predictor_forward(
            guarded_layer0, guarded_hidden, semantic_positions=guarded_positions
        )
        torch.cuda.synchronize()


@pytest.mark.skipif(not _HAS_CUDA, reason="predictor CUDA graph needs CUDA")
def test_no_host_readback_in_eager_chain():
    """The captured body itself must be free of host materialization."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)

    with torch.no_grad():
        talker._code_predictor_forward_incremental(
            layer0.as_subclass(_NoHostReadbackTensor),
            hidden.as_subclass(_NoHostReadbackTensor),
            semantic_positions=positions.as_subclass(_NoHostReadbackTensor),
        )
        torch.cuda.synchronize()


def test_capture_uses_thread_local_error_mode():
    source = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni"
        / "models"
        / "qwen3_tts"
        / "sglang_model.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    graph_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "graph"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
    ]
    assert graph_calls, "Qwen3-TTS predictor CUDA graph capture call not found"
    assert any(
        keyword.arg == "capture_error_mode"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "thread_local"
        for call in graph_calls
        for keyword in call.keywords
    )


def test_normalize_predictor_graph_batch_sizes():
    normalize = Qwen3TTSTalker._normalize_predictor_graph_batch_sizes
    assert normalize(SimpleNamespace(cuda_graph_bs=None), max_batch_size=16) == (
        1,
        2,
        4,
        8,
        12,
        16,
    )
    assert normalize(
        SimpleNamespace(cuda_graph_bs=[4, 2, 2, 64]), max_batch_size=16
    ) == (2, 4, 16)
    assert normalize(SimpleNamespace(), max_batch_size=2) == (1, 2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
