"""Model provider abstraction layer for Avalon."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonCapability(Enum):
    """Enum representing different JSON capabilities of model providers."""

    JSON_SCHEMA = "json_schema"  # Full schema enforcement (e.g., OpenRouter)
    JSON_OBJECT = "json_object"  # JSON object mode with schema-to-prompt conversion
    NONE = "none"  # No JSON support


@dataclass(slots=True)
class ProviderRequest(Generic[T]):
    """Definition of a single phase call to a model provider."""

    phase: str
    model: str
    messages: List[Dict[str, Any]]
    schema_name: str
    schema: Dict[str, Any]
    expected_phase: str
    seat: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    speech_field: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[Dict[str, Any]] = None
    tool_name: Optional[str] = None
    tool_version: Optional[str] = None


@dataclass(slots=True)
class ProviderResponse(Generic[T]):
    """Result of a provider request."""

    payload: T
    raw_response: Dict[str, Any]
    usage: Optional[Dict[str, Any]]
    retries: int
    reasoning: Optional[Dict[str, Any]] = None
    violations: Optional[List[str]] = None
    json_capability_used: Optional[JsonCapability] = None


class ProviderError(RuntimeError):
    """Raised when a model provider encounters an error."""


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    @property
    @abstractmethod
    def json_capability(self) -> JsonCapability:
        """Return the JSON capability of this provider."""

    @abstractmethod
    def call_phase(self, request: ProviderRequest[T]) -> ProviderResponse[T]:
        """Call the provider API for a phase and validate the response."""

    @abstractmethod
    def close(self) -> None:
        """Close any underlying connections."""

    def __enter__(self) -> "ModelProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ProviderFactory:
    """Factory for creating model provider instances."""

    _providers: Dict[str, type[ModelProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[ModelProvider]) -> None:
        """Register a provider class with the factory."""
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, provider_name: str, **kwargs: Any) -> ModelProvider:
        """Create a provider instance by name."""
        if provider_name not in cls._providers:
            raise ProviderError(f"Unknown provider: {provider_name}")

        provider_class = cls._providers[provider_name]
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())