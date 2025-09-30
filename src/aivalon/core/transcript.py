"""Transcript generation and persistence utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import orjson  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for CI
    import json as _json

    class _OrjsonShim:
        @staticmethod
        def loads(data: Any) -> Any:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            return _json.loads(data)

        @staticmethod
        def dumps(obj: Any) -> bytes:
            return _json.dumps(obj).encode("utf-8")

    orjson = _OrjsonShim()  # type: ignore
from ..utils.structlog_shim import structlog

from .fsm import AssassinGuess, EvilConclaveMessage, Mission, Outcome, Proposal, Vote

LOGGER = structlog.get_logger(__name__)

RUNS_DIR = Path("runs")
DEFAULT_RULESET_ID = "avalon-5p-v1"
TIMESTAMP_FMT = "%Y%m%d-%H%M%S"


@dataclass(slots=True)
class PhaseRecord:
    """Single phase entry in the transcript."""

    phase: str
    leader: Optional[int] = None
    output: Optional[Dict[str, Any]] = None
    outputs: Optional[List[Dict[str, Any]]] = None
    cost: Optional[Dict[str, Any]] = None
    tally: Optional[Dict[str, Any]] = None
    reveal: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None  # Reasoning information


@dataclass(slots=True)
class RoundRecord:
    """Collection of phases for a round."""

    round: int
    phases: List[PhaseRecord] = field(default_factory=list)


@dataclass(slots=True)
class Transcript:
    """Complete transcript structure before serialization."""

    seed: str
    ruleset_id: str
    roles: List[Dict[str, Any]]
    leader_order: List[int]
    models: List[Dict[str, Any]]
    rounds: List[RoundRecord] = field(default_factory=list)
    evil_conclave: List[Dict[str, Any]] = field(default_factory=list)
    assassin_guess: Optional[Dict[str, Any]] = None
    winner: Optional[str] = None

    def to_json(self) -> bytes:
        """Serialize the transcript using orjson."""
        return orjson.dumps(self, default=_dataclass_to_dict, option=orjson.OPT_INDENT_2)


@dataclass
class TranscriptWriter:
    """Helper that accumulates transcript entries and writes them to disk."""

    seed: str
    ruleset_id: str = DEFAULT_RULESET_ID
    runs_dir: Path = RUNS_DIR

    roles: List[Dict[str, Any]] = field(default_factory=list)
    leader_order: List[int] = field(default_factory=list)
    models: List[Dict[str, Any]] = field(default_factory=list)
    rounds: List[RoundRecord] = field(default_factory=list)
    current_round: Optional[RoundRecord] = None
    evil_conclave: List[Dict[str, Any]] = field(default_factory=list)
    assassin_guess: Optional[Dict[str, Any]] = None
    winner: Optional[str] = None

    def __post_init__(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # Basic metadata setters -----------------------------------------------------

    def set_roles(self, roles: Dict[int, Any]) -> None:
        self.roles = [
            {"seat": seat, "role": getattr(role, "value", role)} for seat, role in sorted(roles.items())
        ]

    def set_leader_order(self, leader_order: List[int]) -> None:
        self.leader_order = list(leader_order)

    def set_models(self, seat_models: Dict[int, str]) -> None:
        self.models = [
            {"seat": seat, "model": model}
            for seat, model in sorted(seat_models.items())
        ]

    def mark_winner(self, outcome: Outcome) -> None:
        self.winner = outcome.value

    # Round/phase management -----------------------------------------------------

    def start_round(self, round_num: int) -> None:
        if self.current_round is not None:
            self.rounds.append(self.current_round)
        self.current_round = RoundRecord(round=round_num)

    def append_phase(self, record: PhaseRecord) -> None:
        if self.current_round is None:
            raise RuntimeError("No active round when appending phase")
        self.current_round.phases.append(record)

    def commit_current_round(self) -> None:
        if self.current_round is not None:
            self.rounds.append(self.current_round)
            self.current_round = None

    # Specific phase helpers -----------------------------------------------------

    def record_proposal_phase(
        self,
        *,
        phase: str,
        leader: int,
        output: Dict[str, Any],
        cost: Optional[Dict[str, Any]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.append_phase(
            PhaseRecord(
                phase=phase,
                leader=leader,
                output=output,
                cost=cost,
                reasoning=reasoning,
            )
        )

    def record_discussion_phase(
        self, 
        outputs: List[Dict[str, Any]], 
        reasoning: Optional[Dict[str, Any]] = None
    ) -> None:
        self.append_phase(PhaseRecord(phase="DISCUSSION", outputs=outputs, reasoning=reasoning))

    def record_summary_phase(
        self,
        outputs: List[Dict[str, Any]],
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.append_phase(PhaseRecord(phase="SUMMARY", outputs=outputs, reasoning=reasoning))

    def record_vote_phase(
        self,
        outputs: List[Dict[str, Any]],
        tally: Dict[str, Any],
        cost: Optional[Dict[str, Any]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.append_phase(
            PhaseRecord(
                phase="VOTE",
                outputs=outputs,
                tally=tally,
                cost=cost,
                reasoning=reasoning,
            )
        )

    def record_mission_phase(
        self,
        outputs: List[Dict[str, Any]],
        reveal: Dict[str, Any],
        cost: Optional[Dict[str, Any]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.append_phase(
            PhaseRecord(
                phase="MISSION",
                outputs=outputs,
                reveal=reveal,
                cost=cost,
                reasoning=reasoning,
            )
        )

    def record_round_end_phase(self, score: Dict[str, Any]) -> None:
        self.append_phase(PhaseRecord(phase="ROUND_END", output=score))

    def record_evil_conclave(self, messages: List[EvilConclaveMessage]) -> None:
        for message in messages:
            self.evil_conclave.append(
                {
                    "round": message.round_num,
                    "seat": message.seat,
                    "speech": message.speech,
                    "thinking": message.thinking,
                    "reasoning": message.reasoning,
                }
            )

    def record_assassin_guess(self, guess: AssassinGuess) -> None:
        self.assassin_guess = {
            "seat": guess.seat,
            "target": guess.target,
            "correct": guess.correct,
            "reasoning": guess.reasoning,
        }

    # Persistence ----------------------------------------------------------------

    def flush(self) -> Path:
        if self.current_round is not None:
            self.rounds.append(self.current_round)
            self.current_round = None

        self._validate_before_flush()

        transcript = Transcript(
            seed=self.seed,
            ruleset_id=self.ruleset_id,
            roles=self.roles,
            leader_order=self.leader_order,
            models=self.models,
            rounds=self.rounds,
            evil_conclave=self.evil_conclave,
            assassin_guess=self.assassin_guess,
            winner=self.winner,
        )

        timestamp = datetime.now(UTC).strftime(TIMESTAMP_FMT)
        file_name = f"{timestamp}_{self.seed}.json"
        file_path = self.runs_dir / file_name

        file_path.write_bytes(transcript.to_json())
        LOGGER.info("transcript.written", path=str(file_path))
        return file_path

    # Utility --------------------------------------------------------------------

    def _validate_before_flush(self) -> None:
        if not self.roles:
            raise ValueError("Transcript missing roles")
        if not self.rounds:
            raise ValueError("Transcript missing round data")
        if self.winner is None:
            raise ValueError("Transcript missing winner")

    @staticmethod
    def from_game_state(state: Any, *, seat_models: Optional[Dict[int, str]] = None) -> "TranscriptWriter":
        """Create a writer and seed it with information from the GameState."""

        seed = str(getattr(state, "seed", "unknown"))
        writer = TranscriptWriter(seed=seed)
        writer.set_roles(getattr(state, "roles", {}) or {})
        writer.set_leader_order(getattr(state, "leader_order", []) or [])
        if seat_models:
            writer.set_models(seat_models)
        return writer


def _dataclass_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {field: getattr(obj, field) for field in obj.__dataclass_fields__}
    if isinstance(obj, Outcome):
        return obj.value
    raise TypeError(f"Type {type(obj).__name__} is not JSON serializable")
