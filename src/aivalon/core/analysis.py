"""Real-time game analysis data generation.

This module provides a writer that generates analysis data as the game progresses,
capturing prompts, thinking, speech, and actions for each seat in real-time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import orjson  # type: ignore
except ModuleNotFoundError:
    import json as _json

    class _OrjsonShim:
        @staticmethod
        def loads(data: Any) -> Any:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            return _json.loads(data)

        @staticmethod
        def dumps(obj: Any) -> bytes:
            return _json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

    orjson = _OrjsonShim()  # type: ignore

from ..utils.structlog_shim import structlog

LOGGER = structlog.get_logger(__name__)

RUNS_DIR = Path("runs")
TIMESTAMP_FMT = "%Y%m%d-%H%M%S"


@dataclass
class SeatRecord:
    """Single action record for a seat."""

    round: int
    phase: str
    seat: int
    role: str
    prompt: str = ""
    system_prompt: str = ""
    thinking: Optional[str] = None
    speech: Optional[str] = None
    action: Optional[Any] = None
    reflection: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {
            "round": self.round,
            "phase": self.phase,
            "seat": self.seat,
            "role": self.role,
        }
        if self.prompt:
            result["prompt"] = self.prompt
        if self.system_prompt:
            result["system_prompt"] = self.system_prompt
        if self.thinking is not None:
            result["thinking"] = self.thinking
        if self.speech is not None:
            result["speech"] = self.speech
        if self.action is not None:
            result["action"] = self.action
        if self.reflection is not None:
            result["reflection"] = self.reflection
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        return result


@dataclass
class AnalysisWriter:
    """Real-time game analysis data writer.

    Captures prompts, thinking, speech, and actions for each seat as the game progresses.
    """

    seed: str
    ruleset_id: str = "avalon-5p-v1"
    runs_dir: Path = RUNS_DIR

    roles: Dict[int, str] = field(default_factory=dict)
    leader_order: List[int] = field(default_factory=list)
    models: Dict[int, str] = field(default_factory=dict)
    records: List[SeatRecord] = field(default_factory=list)
    winner: Optional[str] = None
    _file_path: Optional[Path] = None
    _md_file_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Ensure runs directory exists and create analysis files."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Create the analysis files immediately
        timestamp = datetime.now(UTC).strftime(TIMESTAMP_FMT)
        base_name = f"{timestamp}_{self.seed}_analysis"

        self._file_path = self.runs_dir / f"{base_name}.json"
        self._md_file_path = self.runs_dir / f"{base_name}.md"

        # Write initial structures
        self._write_to_file()
        self._init_markdown_file()

    # Metadata setters --------------------------------------------------------

    def set_roles(self, roles: Dict[int, Any]) -> None:
        """Set player roles."""
        self.roles = {
            seat: getattr(role, "value", str(role))
            for seat, role in roles.items()
        }

    def set_leader_order(self, leader_order: List[int]) -> None:
        """Set leader rotation order."""
        self.leader_order = list(leader_order)

    def set_models(self, seat_models: Dict[int, str]) -> None:
        """Set model assignments per seat."""
        self.models = dict(seat_models)

    def mark_winner(self, outcome: Any) -> None:
        """Mark the game winner."""
        self.winner = getattr(outcome, "value", str(outcome))

    # Record helpers ----------------------------------------------------------

    def _get_role(self, seat: int) -> str:
        """Get role for a seat, returning Unknown if not found."""
        return self.roles.get(seat, "Unknown")

    def _init_markdown_file(self) -> None:
        """Initialize markdown file with header."""
        if not self._md_file_path:
            return

        header = f"""# Game Analysis - {self.seed}

## Metadata
- **Seed**: {self.seed}
- **Ruleset**: {self.ruleset_id}
- **Status**: In Progress

---

"""
        try:
            self._md_file_path.write_text(header, encoding='utf-8')
        except Exception as e:
            LOGGER.warning("analysis.md_init_failed", error=str(e))

    def _write_to_file(self) -> None:
        """Write current state to file (incremental updates)."""
        if not self._file_path:
            return

        analysis_data = {
            "metadata": {
                "seed": self.seed,
                "ruleset_id": self.ruleset_id,
                "winner": self.winner,
                "roles": [
                    {"seat": seat, "role": role}
                    for seat, role in sorted(self.roles.items())
                ],
                "models": [
                    {"seat": seat, "model": model}
                    for seat, model in sorted(self.models.items())
                ],
                "leader_order": self.leader_order,
            },
            "records": [record.to_dict() for record in self.records],
            "total_records": len(self.records),
        }

        try:
            self._file_path.write_bytes(orjson.dumps(analysis_data))
        except Exception as e:
            LOGGER.warning("analysis.write_failed", error=str(e))

    def _append_to_markdown(self, record: SeatRecord) -> None:
        """Append a record to the markdown file."""
        if not self._md_file_path:
            return

        entry = f"""## 🎮 R{record.round} | {record.phase} | Seat {record.seat} ({record.role})

"""

        if record.system_prompt:
            entry += f"""**📋 System Prompt:**
```
{record.system_prompt}
```

"""

        if record.prompt:
            entry += f"""**📝 User Message:**
```
{record.prompt}
```

"""

        if record.thinking:
            entry += f"""**💭 Thinking:**
> {record.thinking}

"""

        if record.speech:
            entry += f"""**💬 Speech:**
> {record.speech}

"""

        if record.action:
            entry += f"""**⚡ Action:**
```json
{orjson.dumps(record.action, option=orjson.OPT_INDENT_2).decode('utf-8') if not isinstance(record.action, str) else record.action}
```

"""

        if record.reflection:
            entry += f"""**🤔 Reflection:**
> {record.reflection}

"""

        entry += "---\n\n"

        try:
            with open(self._md_file_path, 'a', encoding='utf-8') as f:
                f.write(entry)
        except Exception as e:
            LOGGER.warning("analysis.md_append_failed", error=str(e))

    def record_proposal(
        self,
        *,
        round_num: int,
        phase: str,
        seat: int,
        prompt: str = "",
        system_prompt: str = "",
        action: Optional[Any] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a proposal action."""
        record = SeatRecord(
            round=round_num,
            phase=phase,
            seat=seat,
            role=self._get_role(seat),
            prompt=prompt,
            system_prompt=system_prompt,
            action=action,
            reasoning=reasoning,
        )
        self.records.append(record)
        self._append_to_markdown(record)
        self._write_to_file()

    def record_discussion(
        self,
        *,
        round_num: int,
        phase: str,
        seat: int,
        prompt: str = "",
        system_prompt: str = "",
        thinking: str = "",
        speech: str = "",
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a discussion/summary entry."""
        record = SeatRecord(
            round=round_num,
            phase=phase,
            seat=seat,
            role=self._get_role(seat),
            prompt=prompt,
            system_prompt=system_prompt,
            thinking=thinking,
            speech=speech,
            reasoning=reasoning,
        )
        self.records.append(record)
        self._append_to_markdown(record)
        self._write_to_file()

    def record_vote(
        self,
        *,
        round_num: int,
        seat: int,
        prompt: str = "",
        system_prompt: str = "",
        action: Any,
        reflection: str = "",
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a vote action."""
        record = SeatRecord(
            round=round_num,
            phase="VOTE",
            seat=seat,
            role=self._get_role(seat),
            prompt=prompt,
            system_prompt=system_prompt,
            action=action,
            reflection=reflection,
            reasoning=reasoning,
        )
        self.records.append(record)
        self._append_to_markdown(record)
        self._write_to_file()

    def record_mission(
        self,
        *,
        round_num: int,
        seat: int,
        prompt: str = "",
        system_prompt: str = "",
        action: Any,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a mission action."""
        record = SeatRecord(
            round=round_num,
            phase="MISSION",
            seat=seat,
            role=self._get_role(seat),
            prompt=prompt,
            system_prompt=system_prompt,
            action=action,
            reasoning=reasoning,
        )
        self.records.append(record)
        self._append_to_markdown(record)
        self._write_to_file()

    def record_evil_conclave(
        self,
        *,
        round_num: int,
        seat: int,
        prompt: str = "",
        system_prompt: str = "",
        thinking: str = "",
        speech: str = "",
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record evil conclave message."""
        record = SeatRecord(
            round=round_num,
            phase="EVIL_CONCLAVE",
            seat=seat,
            role=self._get_role(seat),
            prompt=prompt,
            system_prompt=system_prompt,
            thinking=thinking,
            speech=speech,
            reasoning=reasoning,
        )
        self.records.append(record)
        self._append_to_markdown(record)
        self._write_to_file()

    def record_assassin_guess(
        self,
        *,
        seat: int,
        prompt: str = "",
        system_prompt: str = "",
        action: Any,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record assassin guess."""
        record = SeatRecord(
            round=0,
            phase="ASSASSIN_GUESS",
            seat=seat,
            role=self._get_role(seat),
            prompt=prompt,
            system_prompt=system_prompt,
            action=action,
            reasoning=reasoning,
        )
        self.records.append(record)
        self._append_to_markdown(record)
        self._write_to_file()

    # Persistence -------------------------------------------------------------

    def flush(self) -> Path:
        """Final flush with winner marked."""
        self._write_to_file()

        # Update markdown with final status
        if self._md_file_path and self.winner:
            try:
                content = self._md_file_path.read_text(encoding='utf-8')
                content = content.replace("**Status**: In Progress", f"**Status**: Complete - Winner: {self.winner}")
                self._md_file_path.write_text(content, encoding='utf-8')
            except Exception as e:
                LOGGER.warning("analysis.md_finalize_failed", error=str(e))

        LOGGER.info("analysis.written", path=str(self._file_path), records=len(self.records))
        return self._file_path

    @staticmethod
    def from_game_state(
        state: Any,
        *,
        seed: Optional[str] = None,
        seat_models: Optional[Dict[int, str]] = None,
    ) -> "AnalysisWriter":
        """Create an analysis writer from game state."""
        seed_value = seed or str(getattr(state, "seed", "unknown"))
        writer = AnalysisWriter(seed=seed_value)
        writer.set_roles(getattr(state, "roles", {}) or {})
        writer.set_leader_order(getattr(state, "leader_order", []) or [])
        if seat_models:
            writer.set_models(seat_models)
        return writer


__all__ = ["AnalysisWriter", "SeatRecord"]