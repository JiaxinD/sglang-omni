# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared reference-audio path cache-key helpers."""

from sglang_omni.preprocessing import cache_key


def test_reference_path_cache_key_tracks_file_content(tmp_path) -> None:
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"a")
    first_key = cache_key.reference_path_cache_key(ref_audio)

    # Same content -> stable key (so repeat requests hit the cache).
    assert first_key == cache_key.reference_path_cache_key(ref_audio)

    ref_audio.write_bytes(b"longer")
    second_key = cache_key.reference_path_cache_key(ref_audio)

    # Different content -> different key (so a replaced file is not stale-served).
    assert first_key is not None and first_key.startswith("file:")
    assert second_key is not None and second_key.startswith("file:")
    assert first_key != second_key


def test_reference_path_cache_key_same_size_edit_and_non_files(tmp_path) -> None:
    # Same path, same size, same head/tail, different middle must not stale-hit.
    head, tail = b"H" * 8192, b"T" * 8192
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(head + b"a" * 4096 + tail)
    key_a = cache_key.reference_path_cache_key(ref_audio)
    ref_audio.write_bytes(head + b"b" * 4096 + tail)  # same size, middle differs
    assert key_a is not None
    assert key_a != cache_key.reference_path_cache_key(ref_audio)

    # URLs and missing files resolve to no key (callers bypass the cache).
    assert cache_key.reference_path_cache_key("https://example.com/ref.wav") is None
    assert cache_key.reference_path_cache_key(str(tmp_path / "missing.wav")) is None


def test_reference_path_cache_key_memoizes_stable_file_hash(
    monkeypatch, tmp_path
) -> None:
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake wav bytes")
    cache_key._REF_PATH_HASH_MEMO.clear()
    read_calls = 0
    original_read_bytes = cache_key.Path.read_bytes

    def counting_read_bytes(path):
        nonlocal read_calls
        if path == ref_audio:
            read_calls += 1
        return original_read_bytes(path)

    monkeypatch.setattr(cache_key.Path, "read_bytes", counting_read_bytes)

    first_key = cache_key.reference_path_cache_key(ref_audio)
    second_key = cache_key.reference_path_cache_key(ref_audio)

    assert first_key == second_key
    assert read_calls == 1
