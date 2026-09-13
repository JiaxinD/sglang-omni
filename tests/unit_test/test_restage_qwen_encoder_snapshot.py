# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.profiler.qwen_encoder_replay import (
    QwenEncoderCapture,
    QwenEncoderSnapshot,
)


def test_snapshot_owns_inputs_outputs_and_restores_native_fields(tmp_path):
    storage = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    first = SimpleNamespace(
        feature=storage[:, :, ::2],
        feature_attention_mask=torch.tensor([[1, 1, 0]]),
        model_specific_data={"num_audio_tokens": 2},
        audio_fingerprint="first",
    )
    second = SimpleNamespace(
        feature=torch.ones(1, 4, 5),
        model_specific_data={},
    )
    expected = first.feature.clone()
    output = torch.arange(12).reshape(1, 3, 4).float()
    snapshot = QwenEncoderSnapshot.capture(
        [first, second], output, metadata={"run_id": "capture", "attempt": 0}
    )
    storage.zero_()
    first.feature = None
    first.feature_attention_mask.zero_()
    first.model_specific_data["num_audio_tokens"] = 99
    output.zero_()
    snapshot.save(tmp_path / "batch.pt")
    loaded = QwenEncoderSnapshot.load(tmp_path / "batch.pt")
    items = loaded.restore_items()
    assert len(items) == 2
    torch.testing.assert_close(items[0].feature, expected)
    assert items[0].feature_attention_mask.tolist() == [[1, 1, 0]]
    assert items[0].model_specific_data == {"num_audio_tokens": 2}
    assert items[0].audio_fingerprint == "first"
    assert items[1].feature_attention_mask is None
    assert items[1].model_specific_data == {}
    torch.testing.assert_close(loaded.output, torch.arange(12).reshape(1, 3, 4).float())
    assert loaded.metadata == {"run_id": "capture", "attempt": 0}
    items[0].feature.zero_()
    items[0].model_specific_data["num_audio_tokens"] = 200
    again = loaded.restore_items()
    torch.testing.assert_close(again[0].feature, expected)
    assert again[0].model_specific_data["num_audio_tokens"] == 2


def test_snapshot_preserves_mask_and_token_dependent_replay(tmp_path):
    items = [
        SimpleNamespace(
            feature=torch.tensor([[[2.0, 9.0]]]),
            feature_attention_mask=torch.tensor([[1, 0]]),
            model_specific_data={"num_audio_tokens": 3},
        ),
        SimpleNamespace(feature=torch.tensor([[[4.0]]]), model_specific_data={}),
    ]

    def encode(batch):
        return torch.stack(
            [
                (
                    item.feature
                    * (
                        getattr(item, "feature_attention_mask", None)
                        if getattr(item, "feature_attention_mask", None) is not None
                        else 1
                    )
                ).sum()
                + (item.model_specific_data or {}).get("num_audio_tokens", 0)
                for item in batch
            ]
        )

    expected = encode(items)
    snapshot = QwenEncoderSnapshot.capture(items, expected, metadata={})
    snapshot.save(tmp_path / "batch.pt")
    restored = QwenEncoderSnapshot.load(tmp_path / "batch.pt")
    torch.testing.assert_close(encode(restored.restore_items()), expected)
    assert expected.tolist() == [5.0, 4.0]


def test_capture_stops_at_budget_and_does_not_mutate_output(tmp_path):
    capture = QwenEncoderCapture(tmp_path / "capture", max_batches=1, metadata={})
    item = SimpleNamespace(feature=torch.ones(1, 2, 3), model_specific_data={})
    output = torch.ones(1, 3, 4)
    capture.record([item], output)
    capture.record([item], output)
    assert capture.saved == 1
    assert not capture.failed
    assert len(list(capture.directory.glob("batch-*.pt"))) == 1
    status = json.loads((capture.directory / "status.json").read_text())
    assert status == {"state": "limit_reached", "saved_batches": 1, "max_batches": 1}
    capture.close()
    assert (
        json.loads((capture.directory / "status.json").read_text())["state"] == "closed"
    )
    torch.testing.assert_close(output, torch.ones(1, 3, 4))


def test_capture_failure_disables_session_without_retrying_encoder(
    tmp_path, monkeypatch
):
    capture = QwenEncoderCapture(tmp_path / "capture", max_batches=2, metadata={})
    calls = []

    def fail(*args):
        calls.append(1)
        raise OSError("disk full")

    monkeypatch.setattr(QwenEncoderSnapshot, "save", fail)
    item = SimpleNamespace(feature=torch.ones(1, 2, 3), model_specific_data={})
    capture.record([item], torch.ones(1))
    capture.record([item], torch.ones(1))
    assert capture.failed and capture.saved == 0
    assert len(calls) == 1
    assert (
        json.loads((capture.directory / "status.json").read_text())["state"] == "failed"
    )


def test_native_replay_checks_each_output_and_excludes_warmup():
    from sglang_omni.profiler.qwen_encoder_replay import replay_qwen_encoder

    class Model:
        audio_tower = torch.nn.Linear(1, 1)
        calls = 0
        wrong = False

        def get_audio_feature(self, items):
            assert not torch.is_grad_enabled()
            self.calls += 1
            return torch.stack([item.feature.sum() for item in items]) + int(self.wrong)

    model = Model()
    items = [SimpleNamespace(feature=torch.ones(1, 2, 3), model_specific_data={})]
    snapshot = QwenEncoderSnapshot.capture(items, torch.tensor([6.0]), metadata={})
    result = replay_qwen_encoder(model, snapshot, warmup=2, repeats=3, rtol=0, atol=0)
    assert model.calls == 5
    assert len(result["measurements"]) == 3
    assert result["output_verified"] is True
    assert all(row["cuda_stream_ms"] is None for row in result["measurements"])
    assert all(
        row["host_total_s"] >= row["host_submit_s"] >= 0
        for row in result["measurements"]
    )
    model.wrong = True
    with pytest.raises(AssertionError):
        replay_qwen_encoder(model, snapshot, warmup=0, repeats=1, rtol=0, atol=0)
