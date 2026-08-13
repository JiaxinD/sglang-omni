# SPDX-License-Identifier: Apache-2.0
"""Telling users, once, that a configuration spelling is on its way out.

The patches already carry the text: :attr:`ConfigPatch.deprecated` is populated
from the path's own visibility reason and from whatever the compat layer
attaches when it rewrites a legacy spelling. Until now nothing on the launch
path ever printed it -- ``deprecations()`` had a single reader in
``sgl-omni config``, so a user running ``serve`` with V1 syntax got no signal
at all.

Deduplication matters more than it looks: a broadcast alias can produce one
patch per stage from a single flag, and a Router launching eight workers in one
process would otherwise repeat every notice eight times. Users who see the same
warning scroll past learn to ignore warnings.
"""

from __future__ import annotations

import logging

from sglang_omni.config.patch import ConfigPatchSet

__all__ = ["warn_deprecations", "reset_deprecation_state"]

logger = logging.getLogger(__name__)

_SEEN: set[tuple[str, str, str]] = set()


def warn_deprecations(patches: ConfigPatchSet, *, context: str = "") -> list[str]:
    """Log one warning per distinct deprecated ``(path, reason)`` in ``patches``.

    Returns the messages emitted by *this* call, which is what lets a caller
    (``sgl-omni config explain``) present them itself instead of relying on the
    log. A message already emitted in this process is not returned again.
    """
    emitted: list[str] = []
    for patch in patches.deprecations():
        key = (patch.key, patch.deprecated, context)
        if key in _SEEN:
            continue
        _SEEN.add(key)
        where = f" ({context})" if context else ""
        message = f"deprecated config path {patch.key!r}{where}: {patch.deprecated}"
        logger.warning(message)
        emitted.append(message)
    return emitted


def reset_deprecation_state() -> None:
    """Forget what has been warned about. For tests and long-lived processes."""
    _SEEN.clear()
