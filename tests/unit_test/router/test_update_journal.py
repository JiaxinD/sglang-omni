from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from sglang_omni_router.update_journal import (
    JournalUnreadableError,
    JournalUnwritableError,
    UpdateJournal,
    default_journal_path,
)


def test_begin_keep_clear_round_trip(tmp_path: Path) -> None:
    journal = UpdateJournal(str(tmp_path / "j.json"))
    assert journal.pending() == []
    journal.begin("/update_weights_from_disk", ["w1", "w0"])
    assert journal.pending() == ["w0", "w1"]
    assert journal.has_pending() is True
    journal.keep(["w0"])
    assert journal.pending() == ["w0"]
    journal.keep([])
    assert journal.pending() == []
    assert not (tmp_path / "j.json").exists()


def test_unreadable_journal_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "j.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    journal = UpdateJournal(str(path))
    with pytest.raises(JournalUnreadableError):
        journal.pending()
    # has_pending must fail closed (assume a transaction is in progress)
    assert journal.has_pending() is True
    # discard must not silently clobber an unreadable journal
    journal.discard("w0")
    assert path.exists()


@pytest.mark.parametrize("payload", [[], {"worker_ids": [1]}])
def test_malformed_journal_schema_fails_closed(tmp_path: Path, payload) -> None:
    path = tmp_path / "j.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    journal = UpdateJournal(str(path))
    with pytest.raises(JournalUnreadableError):
        journal.pending()
    assert journal.has_pending() is True


def test_discard_removes_one_entry(tmp_path: Path) -> None:
    journal = UpdateJournal(str(tmp_path / "j.json"))
    journal.begin("/x", ["w0", "w1"])
    journal.discard("w0")
    assert journal.pending() == ["w1"]
    journal.discard("w1")
    assert journal.pending() == []


def test_begin_fails_closed_when_the_write_is_not_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a swallowed fsync error would let the update run with no recovery record
    journal = UpdateJournal(str(tmp_path / "j.json"))

    def _fail(fd: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", _fail)
    with pytest.raises(JournalUnwritableError):
        journal.begin("/update_weights_from_disk", ["w0"])


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_begin_fails_closed_when_the_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the rename is durable only once the parent directory entry is synced
    journal = UpdateJournal(str(tmp_path / "j.json"))
    real_fsync = os.fsync

    def _fail_on_directory(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fail_on_directory)
    with pytest.raises(JournalUnwritableError):
        journal.begin("/update_weights_from_disk", ["w0"])


def test_discard_reports_failure_when_the_rewrite_is_not_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = UpdateJournal(str(tmp_path / "j.json"))
    journal.begin("/x", ["w0", "w1"])

    def _fail(fd: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", _fail)
    assert journal.discard("w0") is False


def test_default_path_is_stable_by_endpoint() -> None:
    a = default_journal_path("0.0.0.0", 8000)
    b = default_journal_path("0.0.0.0", 8000)
    c = default_journal_path("0.0.0.0", 8001)
    assert a == b  # same endpoint -> same path across supervisor restarts
    assert a != c
    assert "8000" in a
