"""Avalon tool descriptors used for OpenAI-compatible tool calls.

The project historically relied on structured JSON responses that were parsed
with Pydantic.  As part of the migration to the tool-calling APIs we keep the
same payload definitions but expose them as function tools.  Providers and
the FSM can now dispatch on ``tool_calls`` while still validating against the
original Pydantic models.
"""

from __future__ import annotations

from typing import Dict, List

from .core.rulesets import AvalonRuleset, get_ruleset

#: Version tag included with every :class:`ProviderRequest` so the orchestrator
#: and the frontend can coordinate rollouts.  Bump this whenever the tool
#: contract changes in a breaking way.
TOOLSET_VERSION = "2024-09-01"

def build_proposal_tool(ruleset: AvalonRuleset, round_num: int) -> Dict[str, object]:
    """Build proposal tool definition for a specific ruleset and round.
    
    Args:
        ruleset: The AvalonRuleset to build tools for
        round_num: Current round number (1-5)
        
    Returns:
        Proposal tool definition with exact team size for the round
    """
    team_size = ruleset.team_size_for_round(round_num)
    
    return {
        "type": "function",
        "function": {
            "name": "proposal_payload",
            "description": f"Propose a mission team of exactly {team_size} players for round {round_num}",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "object",
                        "description": f"Mission proposal for round {round_num} (team size: {team_size})",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["PROPOSE"]
                            },
                            "members": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": ruleset.players,
                                    "description": f"玩家座位号，范围 1–{ruleset.players}",
                                },
                                "minItems": team_size,
                                "maxItems": team_size,
                                "description": f"本轮必须提名 {team_size} 名玩家。",
                            }
                        },
                        "required": ["type", "members"]
                    }
                },
                "required": ["action"]
            }
        }
    }

# Legacy support: Default to 5-player ruleset for backward compatibility
_DEFAULT_RULESET = get_ruleset(5)

# Create individual tool builders for the static tools that don't need ruleset adaptation
def _build_discussion_tool() -> Dict[str, object]:
    """Build discussion tool definition (ruleset-independent)."""
    return {
        "type": "function",
        "function": {
            "name": "discussion_payload",
            "description": "Provide private thinking FIRST, then public speech for discussion/summary phases. Order matters: thinking must come before speech.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "【第一步：必须先填写】私人分析，不会展示给其他玩家。请在 100 字以内总结你的推理、怀疑对象、当前局势或战略计划，应当理性且与游戏相关，避免闲聊。",
                        "maxLength": 100
                    },
                    "speech": {
                        "type": "string",
                        "description": "【第二步：在 thinking 之后填写】仅返回公开发言，所有玩家可见，不得为空。在 200 字以内表达你的立场、说服或辩护理由。尽量避免泄露你的隐藏身份或角色信息。将 DSL 语言转为自然语言，不要使用 A，P，S 等 DSL 语言发言",
                        "maxLength": 200
                    }
                },
                "required": ["thinking", "speech"],
                "additionalProperties": False
            }
        }
    }

# Static tool definitions that are independent of ruleset
discussion_tool = _build_discussion_tool()

def _build_vote_tool() -> Dict[str, object]:
    """Build vote tool definition (ruleset-independent)."""
    return {
        "type": "function",
        "function": {
            "name": "vote_payload",
            "description": "Vote to approve or reject a proposed team",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "object",
                        "description": "Vote decision for the pending proposal",
                        "properties": {
                            "type": {"type": "string", "enum": ["VOTE"]},
                            "value": {
                                "type": "string",
                                "enum": ["APPROVE", "REJECT"],
                                "description": "Set to APPROVE (赞成) 或 REJECT (反对) based on whether you support the current mission team"
                            },
                            "reflection": {
                                "type": "string",
                                "maxLength": 100,
                                "description": (
                                    "反思字段，≤100 字。请说明你的身份信息或已知线索，"
                                    "总结本轮玩家发言值得关注的点，并为下轮准备策略。"
                                )
                            }
                        },
                        "required": ["type", "value", "reflection"]
                    }
                },
                "required": ["action"]
            }
        }
    }


def _build_mission_tool() -> Dict[str, object]:
    """Build mission tool definition (ruleset-independent)."""
    return {
        "type": "function",
        "function": {
            "name": "mission_payload",
            "description": "Submit success or fail for a mission attempt",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "object",
                        "description": "Mission execution decision",
                        "properties": {
                            "type": {"type": "string", "enum": ["MISSION"]},
                            "value": {
                                "type": "string",
                                "enum": ["SUCCESS", "FAIL"],
                                "description": "SUCCESS 表示完成任务；FAIL 仅在邪恶阵营需要翻车时选用"
                            }
                        },
                        "required": ["type", "value"]
                    }
                },
                "required": ["action"]
            }
        }
    }


def _build_evil_conclave_tool() -> Dict[str, object]:
    """Build evil conclave tool definition (ruleset-independent)."""
    return {
        "type": "function",
        "function": {
            "name": "evil_conclave_payload",
            "description": "Evil team assassination coordination after good wins 3 missions. Provide thinking FIRST, then speech.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": (
                            "【第一步：必须先填写】邪阵营的刺杀分析，≤400 字。分析谁是梅林，总结游戏中的线索和行为模式。"
                        ),
                        "maxLength": 400
                    },
                    "speech": {
                        "type": "string",
                        "description": (
                            "【第二步：在 thinking 之后填写】发送给其他邪恶玩家的密谈内容，≤300 字。讨论暗杀目标和策略，协调最终猜测。"
                        ),
                        "maxLength": 300
                    }
                },
                "required": ["thinking", "speech"],
                "additionalProperties": False
            }
        }
    }

# Create static tool instances
vote_tool = _build_vote_tool()
mission_tool = _build_mission_tool()
evil_conclave_tool = _build_evil_conclave_tool()


def build_assassin_guess_tool(ruleset: AvalonRuleset) -> Dict[str, object]:
    """Build assassin guess tool definition for a specific ruleset."""
    return {
        "type": "function",
        "function": {
            "name": "assassin_guess_payload",
            "description": "Assassin guesses the identity of Merlin to end the game",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "object",
                        "description": "Assassin's final guess to identify Merlin",
                        "properties": {
                            "type": {"type": "string", "enum": ["ASSASSIN_GUESS"]},
                            "target": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": ruleset.players,
                                "description": f"座位编号，指向你认为是梅林的玩家 (1-{ruleset.players})"
                            }
                        },
                        "required": ["type", "target"]
                    }
                },
                "required": ["action"]
            }
        }
    }

def build_tools_for_game(ruleset: AvalonRuleset, round_num: int = 1) -> List[Dict[str, object]]:
    """Build complete toolset for a specific game ruleset and round.
    
    Args:
        ruleset: The AvalonRuleset to build tools for
        round_num: Current round number (1-5), defaults to 1 for backward compatibility
        
    Returns:
        List of tool definitions adapted to the ruleset and round
    """
    return [
        build_proposal_tool(ruleset, round_num),
        discussion_tool,
        vote_tool,
        mission_tool,
        evil_conclave_tool,
        build_assassin_guess_tool(ruleset),
    ]


def build_tools_for_round(ruleset: AvalonRuleset, round_num: int) -> List[Dict[str, object]]:
    """Build tools specifically for a given round (explicit round-aware version).
    
    Args:
        ruleset: The AvalonRuleset to build tools for  
        round_num: Current round number (1-5)
        
    Returns:
        List of tool definitions with round-specific proposal tool
    """
    return [
        build_proposal_tool(ruleset, round_num),
        discussion_tool,
        vote_tool, 
        mission_tool,
        evil_conclave_tool,
        build_assassin_guess_tool(ruleset),
    ]


def get_static_tools_for_game(ruleset: AvalonRuleset) -> Dict[str, Dict[str, object]]:
    """Get the static tools that don't change per round.
    
    Args:
        ruleset: The AvalonRuleset to build tools for
        
    Returns:
        Dictionary of static tool definitions (excludes proposal tool)
    """
    return {
        "discussion_payload": discussion_tool,
        "vote_payload": vote_tool,
        "mission_payload": mission_tool,
        "evil_conclave_payload": evil_conclave_tool,
        "assassin_guess_payload": build_assassin_guess_tool(ruleset),
    }


def build_tool_definitions_for_game(ruleset: AvalonRuleset, round_num: int = 1) -> Dict[str, Dict[str, object]]:
    """Build tool definitions mapping for a specific game ruleset and round.
    
    Args:
        ruleset: The AvalonRuleset to build tools for
        round_num: Current round number (1-5), defaults to 1 for backward compatibility
        
    Returns:
        Dictionary mapping tool names to definitions
    """
    return {
        "proposal_payload": build_proposal_tool(ruleset, round_num),
        "discussion_payload": discussion_tool,
        "vote_payload": vote_tool,
        "mission_payload": mission_tool,
        "evil_conclave_payload": evil_conclave_tool,
        "assassin_guess_payload": build_assassin_guess_tool(ruleset),
    }


# Legacy support: Create default tools using 5-player ruleset for backward compatibility
# Note: Uses round 1 by default since legacy code doesn't specify rounds
TOOL_DEFINITIONS: Dict[str, Dict[str, object]] = build_tool_definitions_for_game(_DEFAULT_RULESET, round_num=1)


def _tool_list() -> List[Dict[str, object]]:
    """Return tool definitions as a list.

    The helper keeps ``ALL_TOOLS`` immutable in tests and at runtime while
    still exposing the canonical mapping.
    """
    return list(TOOL_DEFINITIONS.values())


# Complete list of all tools (legacy support for 5-player)
# Note: Uses round 1 by default since legacy code doesn't specify rounds
ALL_TOOLS: List[Dict[str, object]] = build_tools_for_game(_DEFAULT_RULESET, round_num=1)


# Map tool names to their original payload types for validation
TOOL_NAME_TO_PAYLOAD: Dict[str, str] = {
    "proposal_payload": "ProposalPayload",
    "discussion_payload": "DiscussionPayload",
    "vote_payload": "VotePayload",
    "mission_payload": "MissionPayload",
    "evil_conclave_payload": "EvilConclavePayload",
    "assassin_guess_payload": "AssassinGuessPayload",
}


PAYLOAD_TO_TOOL_NAME: Dict[str, str] = {
    value: key for key, value in TOOL_NAME_TO_PAYLOAD.items()
}


__all__ = [
    "ALL_TOOLS",
    "build_tools_for_game",
    "build_tools_for_round",
    "build_tool_definitions_for_game",
    "build_proposal_tool",
    "build_assassin_guess_tool",
    "get_static_tools_for_game",
    "PAYLOAD_TO_TOOL_NAME",
    "TOOL_DEFINITIONS",
    "TOOL_NAME_TO_PAYLOAD",
    "TOOLSET_VERSION",
]
