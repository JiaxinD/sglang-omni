# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from transformers import WhisperConfig, WhisperForConditionalGeneration  # noqa: E402

from sglang_omni.models.whisper_asr.mlx.model import (  # noqa: E402
    WhisperMlxModel,
    decoder_dimensions_from_hf_config,
)
from sglang_omni.models.whisper_asr.mlx.runner import (  # noqa: E402
    WhisperMlxModelRunner,
    WhisperSuppression,
)

MEL_BINS = 8
SOURCE_POSITIONS = 6
VOCAB_SIZE = 40
TIMESTAMP_BEGIN = 36


@pytest.fixture(autouse=True)
def exact_mlx_device() -> Iterator[None]:
    # Metal float32 kernels accumulate differently from Torch CPU; the CPU
    # device isolates graph parity from that precision difference.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny_hf_config() -> WhisperConfig:
    return WhisperConfig(
        vocab_size=VOCAB_SIZE,
        num_mel_bins=MEL_BINS,
        d_model=8,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_ffn_dim=16,
        max_source_positions=SOURCE_POSITIONS,
        max_target_positions=12,
        pad_token_id=1,
        bos_token_id=1,
        eos_token_id=1,
        decoder_start_token_id=2,
    )


def mlx_community_name(name: str) -> str:
    """Rename one HF Whisper tensor to the mlx-community checkpoint naming."""
    renames = (
        ("model.encoder.layers.", "encoder.blocks."),
        ("model.decoder.layers.", "decoder.blocks."),
        ("model.encoder.layer_norm.", "encoder.ln_post."),
        ("model.decoder.layer_norm.", "decoder.ln."),
        ("model.encoder.", "encoder."),
        ("model.decoder.embed_tokens.", "decoder.token_embedding."),
        ("model.decoder.embed_positions.weight", "decoder.positional_embedding"),
        (".self_attn_layer_norm.", ".attn_ln."),
        (".encoder_attn_layer_norm.", ".cross_attn_ln."),
        (".final_layer_norm.", ".mlp_ln."),
        (".self_attn.", ".attn."),
        (".encoder_attn.", ".cross_attn."),
        (".q_proj.", ".query."),
        (".k_proj.", ".key."),
        (".v_proj.", ".value."),
        (".out_proj.", ".out."),
        (".fc1.", ".mlp1."),
        (".fc2.", ".mlp2."),
    )
    for old, new in renames:
        name = name.replace(old, new)
    return name


def build_pair(
    *, naming: str
) -> tuple[WhisperForConditionalGeneration, WhisperMlxModel]:
    torch.manual_seed(0)
    reference = WhisperForConditionalGeneration(tiny_hf_config()).eval()
    weights: dict[str, mx.array] = {}
    for name, tensor in reference.state_dict().items():
        if name == "proj_out.weight":
            continue
        elif naming == "hf":
            weights[name] = mx.array(tensor.detach().float().numpy())
        elif name == "model.encoder.embed_positions.weight":
            continue
        elif name.endswith(("conv1.weight", "conv2.weight")):
            weights[mlx_community_name(name)] = mx.array(
                tensor.detach().float().numpy().transpose(0, 2, 1)
            )
        else:
            weights[mlx_community_name(name)] = mx.array(
                tensor.detach().float().numpy()
            )
    if naming == "mlx-community":
        weights["alignment_heads"] = mx.array([[1, 0]], dtype=mx.int64)
    else:
        pass
    model = WhisperMlxModel(decoder_dimensions_from_hf_config(tiny_hf_config()))
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    return reference, model


def mel_features(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, MEL_BINS, 2 * SOURCE_POSITIONS)).astype(np.float32)


@pytest.mark.parametrize("naming", ["hf", "mlx-community"])
def test_prefill_and_cached_decode_match_the_reference(naming: str) -> None:
    reference, model = build_pair(naming=naming)
    features = mel_features(1)
    prompt = [2, 5, 7]
    continuation = [9, 11]

    with torch.no_grad():
        expected = reference(
            input_features=torch.from_numpy(features),
            decoder_input_ids=torch.tensor([prompt + continuation]),
        ).logits[0]

    cross_states = model.cross_attention_states(model.encoder(mx.array(features)))
    cache = model.make_cache()
    prefill_logits = model.decode(
        mx.array([prompt], dtype=mx.int32), cache=cache, cross_states=cross_states
    )
    np.testing.assert_allclose(
        np.array(prefill_logits[0, -1]), expected[len(prompt) - 1].numpy(), atol=1e-5
    )
    for step, token_id in enumerate(continuation):
        step_logits = model.decode(
            mx.array([[token_id]], dtype=mx.int32),
            cache=cache,
            cross_states=cross_states,
        )
        np.testing.assert_allclose(
            np.array(step_logits[0, -1]),
            expected[len(prompt) + step].numpy(),
            atol=1e-5,
        )


def test_mlx_community_encoder_positions_are_whisper_sinusoids() -> None:
    reference, model = build_pair(naming="mlx-community")
    np.testing.assert_allclose(
        np.array(model.encoder.embed_positions.weight),
        reference.model.encoder.embed_positions.weight.detach().numpy(),
        atol=1e-6,
    )


def test_decoder_self_attention_meets_the_mlx_runner_contract() -> None:
    from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (
        find_attention_layers,
    )

    _reference, model = build_pair(naming="hf")
    layers, attention_names = find_attention_layers(model)

    assert len(layers) == 2
    assert attention_names == ["self_attn", "self_attn"]


def test_suppression_matches_the_swift_decode_rules() -> None:
    suppression = WhisperSuppression.build(
        vocab_size=VOCAB_SIZE,
        suppress_token_ids=[3, 4],
        begin_suppress_token_ids=[5, 1],
        timestamp_begin_token_id=TIMESTAMP_BEGIN,
    )
    logits = mx.zeros((1, VOCAB_SIZE))

    first_step = np.array(suppression.first_step(logits))[0]
    later_step = np.array(suppression.later_step(logits))[0]

    blocked_first = {int(i) for i in np.nonzero(first_step < -1e8)[0]}
    blocked_later = {int(i) for i in np.nonzero(later_step < -1e8)[0]}
    timestamps = set(range(TIMESTAMP_BEGIN, VOCAB_SIZE))
    assert blocked_first == {1, 3, 4, 5} | timestamps
    assert blocked_later == {3, 4} | timestamps


def test_runner_strips_encoder_placeholders_and_suppresses_first_token() -> None:
    _reference, model = build_pair(naming="hf")
    runner = object.__new__(WhisperMlxModelRunner)
    runner.model = model
    runner.disable_radix_cache = True
    runner.cross_states_by_request = {}
    runner.suppression = WhisperSuppression.build(
        vocab_size=VOCAB_SIZE,
        suppress_token_ids=[],
        begin_suppress_token_ids=[],
        timestamp_begin_token_id=TIMESTAMP_BEGIN,
    )
    runner._acquire_cache = (
        model.make_cache
    )  # noqa: leading-underscore  # SGLang hook name
    item = SimpleNamespace(
        feature=torch.from_numpy(mel_features(2)),
        model_specific_data={"num_audio_tokens": SOURCE_POSITIONS},
    )
    req = SimpleNamespace(multimodal_inputs=SimpleNamespace(mm_items=[item]))
    prompt = [2, 5, 7]
    full_token_ids = [1] * SOURCE_POSITIONS + prompt

    pending = runner.prefill_start(
        "speech", prompt, full_token_ids, [], list(range(3)), 0, req=req
    )

    assert pending.full_token_ids == prompt
    assert pending.cache[0].offset == len(prompt)
    assert int(pending.lazy_token.item()) < TIMESTAMP_BEGIN
    assert "speech" in runner.cross_states_by_request


def test_suppression_built_on_one_thread_evaluates_on_another() -> None:
    import threading

    mx.set_default_device(mx.gpu)
    suppression = WhisperSuppression.build(
        vocab_size=VOCAB_SIZE,
        suppress_token_ids=[3],
        begin_suppress_token_ids=[5],
        timestamp_begin_token_id=TIMESTAMP_BEGIN,
    )
    failures: list[BaseException] = []

    def scheduler_thread() -> None:
        try:
            with mx.stream(mx.new_thread_local_stream(mx.gpu)):
                mx.eval(suppression.first_step(mx.zeros((1, VOCAB_SIZE))))
        except RuntimeError as error:
            failures.append(error)

    thread = threading.Thread(target=scheduler_thread)
    thread.start()
    thread.join()

    assert failures == []
