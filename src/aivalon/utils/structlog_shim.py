"""Fallback shim that mimics :mod:`structlog` when the dependency is unavailable."""

from __future__ import annotations

import logging
from typing import Any

try:  # pragma: no cover - prefer real structlog when available
    import structlog as _structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments

    class _BoundLogger:
        """Minimal bound logger implementing the subset of APIs we rely on."""

        def __init__(self, logger: logging.Logger, **bound: Any) -> None:
            self._logger = logger
            self._bound = dict(bound)

        def bind(self, **kwargs: Any) -> "_BoundLogger":
            merged = dict(self._bound)
            merged.update(kwargs)
            return _BoundLogger(self._logger, **merged)

        def _emit(self, level: int, event: str, **kwargs: Any) -> None:
            payload = dict(self._bound)
            payload.update(kwargs)
            self._logger.log(level, "%s %s", event, payload)

        def info(self, event: str, **kwargs: Any) -> None:
            self._emit(logging.INFO, event, **kwargs)

        def warning(self, event: str, **kwargs: Any) -> None:
            self._emit(logging.WARNING, event, **kwargs)

        def error(self, event: str, **kwargs: Any) -> None:
            self._emit(logging.ERROR, event, **kwargs)

        def debug(self, event: str, **kwargs: Any) -> None:
            self._emit(logging.DEBUG, event, **kwargs)

    class _StructlogShim:
        def __init__(self) -> None:
            logging.basicConfig(level=logging.INFO)

        def get_logger(self, name: str | None = None) -> _BoundLogger:
            return _BoundLogger(logging.getLogger(name or "aivalon"))

    structlog = _StructlogShim()  # type: ignore
else:  # pragma: no cover - re-export real structlog
    structlog = _structlog  # type: ignore

__all__ = ["structlog"]
