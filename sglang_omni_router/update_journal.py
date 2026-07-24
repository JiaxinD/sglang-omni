# SPDX-License-Identifier: Apache-2.0
"""Weight-update transaction journal (CP crash recovery).

A weight update is a distributed transaction: targets are disabled, the
barrier waits for the DPs, the broadcast runs, and disabled state is
restored. A CP crash mid-transaction loses the in-memory registry, and a
freshly built replacement would re-enable workers whose weight versions may
now differ (some accepted the update, some did not).

The journal makes that failure closed. Before disabling and publishing, the
target set is fsynced to a STABLE state path (keyed by host:port, so it
survives a full supervisor restart, not just a CP respawn inside one
supervisor). It is cleared only when the transaction reaches a KNOWN
outcome: the broadcast completed (the caller sees per-worker results) or the
ACK barrier aborted before anything was sent. If the CP dies after the
broadcast started but before results are known, the journal persists and a
restarting CP re-disables the journaled workers and logs; an operator
verifies weight versions and re-enables them (an admin-authenticated
PUT /workers {"disabled": false}), which discards the entry.

Fail-closed on read: any error other than a missing file is treated as
"a transaction may be in progress" so recovery keeps workers disabled
rather than silently enabling potentially mixed versions.

Fail-closed on write: a write that could not be made durable (including the
parent directory fsync) raises instead of returning, so the update refuses
rather than running with no recovery record. Non-POSIX hosts expose no
directory handle to sync, so there durability is file-level only (the
multi-process router is Linux-only in any case).
"""

from __future__ import annotations

import json
import os


class JournalUnreadableError(Exception):
    """The journal exists but could not be read; recovery must fail closed."""


class JournalUnwritableError(Exception):
    """The journal could not be made durable; the caller must fail closed.

    Reporting a write that never reached the disk would let a weight update
    run with no crash-recovery record, which is the mixed-weight re-enable
    this journal exists to prevent.
    """


class UpdateJournal:
    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    def exists(self) -> bool:
        return os.path.exists(self._path)

    def pending(self) -> list[str]:
        """Journaled target worker ids, or [] if there is no transaction.

        Raises JournalUnreadableError if the file is present but corrupt, so
        callers can fail closed instead of treating it as "no transaction".
        """
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            raise JournalUnreadableError(str(exc)) from exc
        if not isinstance(data, dict):
            raise JournalUnreadableError("journal is not an object")
        workers = data.get("worker_ids")
        if not isinstance(workers, list) or not all(
            isinstance(worker_id, str) for worker_id in workers
        ):
            raise JournalUnreadableError("worker_ids is not a list of strings")
        return workers

    def has_pending(self) -> bool:
        try:
            return bool(self.pending())
        except JournalUnreadableError:
            # Note (Jiaxin Deng): unreadable = assume a transaction is in
            # progress (fail closed).
            return True

    def begin(self, path: str, worker_ids: list[str]) -> None:
        self._write({"path": path, "worker_ids": sorted(worker_ids)})

    def keep(self, worker_ids: list[str]) -> None:
        """Persist the still-unresolved target set (empty clears)."""
        if worker_ids:
            self._write({"worker_ids": sorted(worker_ids)})
        else:
            self.clear()

    def discard(self, worker_id: str) -> bool:
        """Remove one id; False when the entry could not be durably resolved.

        A False return means the file needs operator inspection; callers must
        surface that instead of reporting success while the 409 update gate
        stays closed.
        """
        try:
            remaining = [wid for wid in self.pending() if wid != worker_id]
        except JournalUnreadableError:
            # Note (Jiaxin Deng): cannot safely edit an unreadable journal;
            # leave it for an operator rather than partially clearing it.
            return False
        try:
            self.keep(remaining)
        except JournalUnwritableError:
            return False
        return True

    def clear(self) -> None:
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise JournalUnwritableError(f"{self._path}: {exc}") from exc
        self._fsync_dir()

    def _write(self, data: dict) -> None:
        directory = os.path.dirname(self._path) or "."
        tmp_path = f"{self._path}.tmp"
        try:
            os.makedirs(directory, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(data))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError as exc:
            raise JournalUnwritableError(f"{self._path}: {exc}") from exc
        self._fsync_dir()

    def _fsync_dir(self) -> None:
        # Note (Jiaxin Deng): the rename/unlink is only durable across a host
        # crash once the parent directory entry is synced, so a failure here
        # must not be reported as a durable write. Windows exposes no
        # directory handle to sync; there the file fsync plus the atomic
        # replace is the strongest guarantee available.
        if os.name != "posix":
            return
        directory = os.path.dirname(self._path) or "."
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError as exc:
            raise JournalUnwritableError(f"{directory}: {exc}") from exc
        try:
            os.fsync(fd)
        except OSError as exc:
            raise JournalUnwritableError(f"{directory}: {exc}") from exc
        finally:
            os.close(fd)


def default_journal_path(host: str, port: int) -> str:
    """A stable per-endpoint journal path that survives supervisor restarts.

    Not the per-run workdir (which is random and lost on a full restart);
    keyed by host:port under a stable base so the same endpoint always finds
    its own in-progress transaction.
    """
    import tempfile

    safe_host = host.replace(":", "_").replace("/", "_")
    base = os.path.join(tempfile.gettempdir(), "sglang-omni-router-state")
    return os.path.join(base, f"{safe_host}_{port}", "update_journal.json")
