"""Gracefully degrade when python-dotenv is not installed."""

from __future__ import annotations

try:  # pragma: no cover - prefer real implementation
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for tests

    def load_dotenv(*_args, **_kwargs):  # type: ignore[override]
        """No-op replacement that mirrors the signature of :func:`dotenv.load_dotenv`."""

        return False

__all__ = ["load_dotenv"]
