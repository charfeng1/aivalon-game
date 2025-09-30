"""Core package for the Avalon AI simulator MVP."""

# Re-export modules from the new structure for backward compatibility
from . import agents, human_player
from .core import fsm, context, roles, schemas, transcript
from .utils import leak_filter, rng
from .providers import openrouter, providers
from .services import cli

# Direct re-exports for backward compatibility
from .core.fsm import *  # noqa: F403, F401

__all__ = [
    "agents",
    "human_player",
    "cli",
    "fsm",
    "context",
    "roles",
    "schemas",
    "openrouter",
    "transcript",
    "leak_filter",
    "rng",
    "providers",
]
