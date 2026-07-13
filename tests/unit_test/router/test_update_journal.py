from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni_router.update_journal import (
    JournalUnreadableError,
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


def test_discard_removes_one_entry(tmp_path: Path) -> None:
    journal = UpdateJournal(str(tmp_path / "j.json"))
    journal.begin("/x", ["w0", "w1"])
    journal.discard("w0")
    assert journal.pending() == ["w1"]
    journal.discard("w1")
    assert journal.pending() == []


def test_default_path_is_stable_by_endpoint() -> None:
    a = default_journal_path("0.0.0.0", 8000)
    b = default_journal_path("0.0.0.0", 8000)
    c = default_journal_path("0.0.0.0", 8001)
    assert a == b  # same endpoint -> same path across supervisor restarts
    assert a != c
    assert "8000" in a
