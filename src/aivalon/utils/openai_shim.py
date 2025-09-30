"""Provide a minimal stub for the OpenAI client in environments without the SDK."""

from __future__ import annotations

try:  # pragma: no cover - prefer the real SDK when available
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for tests

    class _Unavailable:
        def __getattr__(self, name: str) -> "_Unavailable":
            return self

        def __call__(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("openai package is not installed; network calls are disabled")

    class OpenAI:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple stub
            self.chat = _Unavailable()

        def close(self) -> None:
            return None

__all__ = ["OpenAI"]
