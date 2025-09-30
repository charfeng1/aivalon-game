"""Avalon game rulesets for different player counts.

This module defines the official Avalon rules for different numbers of players,
including team composition, mission sizes, and special rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AvalonRuleset:
    """Defines the rules for an Avalon game with a specific number of players."""
    
    players: int
    good: int
    evil: int
    mission_sizes: List[int]
    double_fail_round: Optional[int] = None  # Round that requires 2 FAIL votes to fail
    good_win_threshold: int = 3
    evil_win_threshold: int = 3
    max_failed_proposals: int = 3
    
    def __post_init__(self) -> None:
        """Validate the ruleset configuration."""
        if self.good + self.evil != self.players:
            raise ValueError(f"Good ({self.good}) + Evil ({self.evil}) must equal players ({self.players})")
        
        if len(self.mission_sizes) != 5:
            raise ValueError("Mission sizes must contain exactly 5 rounds")
        
        if self.double_fail_round is not None and not (1 <= self.double_fail_round <= 5):
            raise ValueError("Double fail round must be between 1 and 5")
    
    @property
    def all_seats(self) -> tuple[int, ...]:
        """Return all valid seat numbers for this game."""
        return tuple(range(1, self.players + 1))
    
    def team_size_for_round(self, round_num: int) -> int:
        """Get the required team size for a specific round."""
        if 1 <= round_num <= len(self.mission_sizes):
            return self.mission_sizes[round_num - 1]
        return self.mission_sizes[-1]  # Default to last round's size
    
    def requires_double_fail(self, round_num: int) -> bool:
        """Check if this round requires 2 FAIL votes to fail the mission."""
        return self.double_fail_round == round_num
    
    def get_mission_sizes_description(self) -> str:
        """Get a human-readable description of mission sizes."""
        return "-".join(map(str, self.mission_sizes))


# Official Avalon rulesets based on player count
RULESETS: Dict[int, AvalonRuleset] = {
    5: AvalonRuleset(
        players=5, 
        good=3, 
        evil=2, 
        mission_sizes=[2, 3, 2, 3, 3]
    ),
    6: AvalonRuleset(
        players=6, 
        good=4, 
        evil=2, 
        mission_sizes=[2, 3, 4, 3, 4]
    ),
    7: AvalonRuleset(
        players=7, 
        good=4, 
        evil=3, 
        mission_sizes=[2, 3, 3, 4, 4], 
        double_fail_round=4
    ),
    8: AvalonRuleset(
        players=8, 
        good=5, 
        evil=3, 
        mission_sizes=[3, 4, 4, 5, 5], 
        double_fail_round=4
    ),
    9: AvalonRuleset(
        players=9, 
        good=6, 
        evil=3, 
        mission_sizes=[3, 4, 4, 5, 5], 
        double_fail_round=4
    ),
    10: AvalonRuleset(
        players=10, 
        good=6, 
        evil=4, 
        mission_sizes=[3, 4, 4, 5, 5], 
        double_fail_round=4
    ),
}


def get_ruleset(players: int) -> AvalonRuleset:
    """Get the ruleset for a specific number of players.
    
    Args:
        players: Number of players in the game
        
    Returns:
        The appropriate AvalonRuleset
        
    Raises:
        ValueError: If no ruleset is defined for the given number of players
    """
    if players not in RULESETS:
        available = ", ".join(map(str, sorted(RULESETS.keys())))
        raise ValueError(f"No ruleset defined for {players} players. Available: {available}")
    
    return RULESETS[players]


def get_available_player_counts() -> List[int]:
    """Get all supported player counts."""
    return sorted(RULESETS.keys())


def format_rules_description(ruleset: AvalonRuleset) -> str:
    """Format a human-readable description of the ruleset in Chinese."""
    rules_text = (
        f"{ruleset.players}人局：{ruleset.good}名正义，{ruleset.evil}名邪恶。\n"
        f"任务队伍人数：{ruleset.get_mission_sizes_description()}。\n"
    )
    
    if ruleset.double_fail_round:
        rules_text += f"注意：第{ruleset.double_fail_round}轮任务需要至少两张失败票才判定失败。\n"
    
    return rules_text