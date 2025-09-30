"""
Centralized registry of AI model providers and their available models.

This file serves as the single source of truth for all supported model configurations.
Add new providers or models here to make them available in the seat configuration UI.
"""

from typing import Dict, List, TypedDict


class ProviderInfo(TypedDict):
    """Information about an AI provider."""
    name: str
    display_name: str
    models: List[str]


# Registry of all available providers and their models
MODEL_REGISTRY: Dict[str, ProviderInfo] = {
    "deepseek": {
        "name": "deepseek",
        "display_name": "DeepSeek",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    },
    "openrouter": {
        "name": "openrouter",
        "display_name": "OpenRouter",
        "models": [
            "google/gemini-2.5-flash-lite-preview-09-2025",
            "google/gemini-2.5-pro",
            "moonshotai/kimi-k2-0905",
            "openai/gpt-4-turbo",
            "anthropic/claude-sonnet-4.5",
            "x-ai/grok-4",
            "anthropic/claude-sonnet-4",
        ],
    },
    "dashscope": {
        "name": "dashscope",
        "display_name": "DashScope (Alibaba)",
        "models": [
            "qwen-max",
            "qwen-plus",
            "qwen3-coder-plus",
        ],
    },
    "glm": {
        "name": "glm",
        "display_name": "GLM (Zhipu AI)",
        "models": [
            "glm-4.6",
            "glm-4.5",
            "glm-3-turbo",
        ],
    },
    "kimi": {
        "name": "kimi",
        "display_name": "Kimi (Moonshot)",
        "models": [
            "kimi-k2-0905-preview",
        ],
    },
}


def get_all_providers() -> List[ProviderInfo]:
    """Get list of all available providers with their models."""
    return list(MODEL_REGISTRY.values())


def get_provider_models(provider_name: str) -> List[str]:
    """Get list of available models for a specific provider."""
    provider = MODEL_REGISTRY.get(provider_name)
    return provider["models"] if provider else []


def get_provider_display_name(provider_name: str) -> str:
    """Get display name for a provider."""
    provider = MODEL_REGISTRY.get(provider_name)
    return provider["display_name"] if provider else provider_name