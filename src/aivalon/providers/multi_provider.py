"""Multi-provider client that routes requests to different providers based on seat."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, TypeVar

from pydantic import BaseModel

from ..utils.structlog_shim import structlog
from ..utils.dotenv_shim import load_dotenv
from .providers import (
    JsonCapability,
    ModelProvider,
    ProviderError,
    ProviderFactory,
    ProviderRequest,
    ProviderResponse,
)

load_dotenv()

LOGGER = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class MultiProvider(ModelProvider):
    """Multi-provider client that routes requests to different providers based on seat."""

    def __init__(
        self,
        *,
        seat_providers: Optional[Dict[int, str]] = None,
        default_provider: str = "openrouter",
        **kwargs: Any,
    ) -> None:
        self.seat_providers = seat_providers or {}
        self.default_provider = default_provider
        self._provider_instances: Dict[str, ModelProvider] = {}
        self.kwargs = kwargs

    @property
    def json_capability(self) -> JsonCapability:
        """Return the default provider's JSON capability."""
        default_client = self._get_provider_instance(self.default_provider)
        return default_client.json_capability

    def _get_provider_instance(self, provider_name: str) -> ModelProvider:
        """Get or create a provider instance."""
        if provider_name not in self._provider_instances:
            try:
                self._provider_instances[provider_name] = ProviderFactory.create(
                    provider_name, **self.kwargs
                )
            except ProviderError as exc:
                LOGGER.error("multi_provider.provider_creation_failed", 
                           provider=provider_name, error=str(exc))
                raise
        return self._provider_instances[provider_name]

    def _get_provider_for_seat(self, seat: Optional[int]) -> str:
        """Determine which provider to use for a given seat."""
        if seat is not None and seat in self.seat_providers:
            return self.seat_providers[seat]
        return self.default_provider

    def call_phase(self, request: ProviderRequest[T]) -> ProviderResponse[T]:
        """Route the request to the appropriate provider based on seat."""
        provider_name = self._get_provider_for_seat(request.seat)
        provider = self._get_provider_instance(provider_name)
        
        logger = LOGGER.bind(
            phase=request.phase,
            seat=request.seat,
            provider=provider_name,
            model=request.model
        )
        
        logger.info("multi_provider.routing_request", 
                   seat_providers=self.seat_providers,
                   selected_provider=provider_name)
        
        try:
            return provider.call_phase(request)
        except Exception as exc:
            logger.error("multi_provider.provider_call_failed", error=str(exc))
            raise

    def close(self) -> None:
        """Close all underlying provider connections."""
        for provider in self._provider_instances.values():
            try:
                provider.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                LOGGER.warning("multi_provider.close_failed", error=str(exc))
        self._provider_instances.clear()

    def __enter__(self) -> "MultiProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# Register MultiProvider with the factory
ProviderFactory.register("multi", MultiProvider)