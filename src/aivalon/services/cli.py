"""Typer CLI entry point for running AI-driven Avalon matches."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from ..utils.structlog_shim import structlog
import typer
from ..utils.dotenv_shim import load_dotenv

from ..agents import AgentRegistry, BaseSeatAgent, HumanSeatAgent, ModelSeatAgent
from ..core.context import ContextBuilder, load_context_config
from ..core.fsm import (
    AssassinGuess,
    EvilConclaveMessage,
    MissionOutcome,
    Outcome,
    Phase,
    PhaseServices,
    TOTAL_SEATS,
    run_game,
)
from ..providers.openrouter import OpenRouterClient
from ..providers.providers import ModelProvider, ProviderError, ProviderFactory
from ..core.transcript import TranscriptWriter
from ..core.analysis import AnalysisWriter
from ..human_player import DEFAULT_HUMAN_SEAT, GameNarrator, HumanPlayerService, HumanRecorder

LOGGER = structlog.get_logger(__name__)
DEFAULT_CONFIG_PATH = Path("config/config.local.json")

app = typer.Typer(help="Run Avalon AI simulations.", invoke_without_command=False)
_configured_logging = False


def _attach_reasoning(reasoning: Optional[Dict[str, Any]], *, seat: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Return a shallow copy of reasoning info with seat metadata."""

    if not reasoning:
        return None
    enriched = dict(reasoning)
    if seat is not None:
        enriched.setdefault("seat", seat)
    return enriched


@dataclass(slots=True)
class TranscriptPaths:
    json_path: Path
    markdown_path: Path
    context_markdown_path: Optional[Path] = None


class SimulationRecorder:
    """Collects phase outputs during the FSM run and emits transcript files."""

    def __init__(self, *, total_seats: int = TOTAL_SEATS) -> None:
        self.total_seats = total_seats
        self.events: List[Dict[str, Any]] = []
        self._sequence = 0
        self.context_entries: List[Dict[str, Any]] = []
        self.analysis_writer: Optional[AnalysisWriter] = None

        self._discussion_round: Optional[int] = None
        self._discussion_buffer: List[Dict[str, Any]] = []

        self._vote_round: Optional[int] = None
        self._vote_buffer: List[Dict[str, Any]] = []
        self._vote_seen_seats: Set[int] = set()
        self._seat_range: Set[int] = set(range(1, total_seats + 1))

        self._mission_round: Optional[int] = None
        self._mission_buffer: List[Dict[str, Any]] = []
        self._mission_expected: Optional[int] = None

        self._conclave_round: Optional[int] = None
        self._conclave_expected: Optional[int] = None
        self._conclave_buffer: List[Dict[str, Any]] = []
        self.evil_conclave_messages: List[Dict[str, Any]] = []

        self.assassin_guess: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_members(members: Iterable[Any]) -> List[int]:
        try:
            normalized = {int(seat) for seat in members}
        except Exception:  # pragma: no cover - defensive
            return list(members)
        return list(sorted(normalized))

    def _canonicalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action = payload.get("action")
        if isinstance(action, dict) and "members" in action:
            action = dict(action)
            action["members"] = self._canonical_members(action.get("members", []))
            payload = dict(payload)
            payload["action"] = action
        return payload

    def init_analysis_writer(self, state: Any, seed: Optional[str] = None, seat_models: Optional[Dict[int, str]] = None) -> None:
        """Initialize the analysis writer from game state."""
        self.analysis_writer = AnalysisWriter.from_game_state(state, seed=seed, seat_models=seat_models)

    def _get_context(self, round_num: int, phase: str, seat: int) -> tuple[str, str]:
        """Get prompt and system_prompt for given round/phase/seat."""
        for entry in reversed(self.context_entries):
            if entry["round"] == round_num and entry["phase"] == phase and entry["seat"] == seat:
                return entry.get("user_message", ""), entry.get("system_prompt", "")
        return "", ""

    def record_context(
        self,
        *,
        round_num: int,
        phase: str,
        seat: int,
        request: Any,
        prompt_payload: Optional[Dict[str, Any]] = None,
        prompt_format: str = "json",
    ) -> None:
        """Persist the raw prompt seen by an agent for later inspection."""

        messages = list(getattr(request, "messages", []) or [])
        system_prompt = ""
        user_message = ""
        for message in messages:
            role = message.get("role") if isinstance(message, dict) else None
            content = message.get("content", "") if isinstance(message, dict) else ""
            if role == "system" and not system_prompt:
                system_prompt = str(content)
            elif role == "user" and not user_message:
                user_message = str(content)

        entry = {
            "order": len(self.context_entries) + 1,
            "round": round_num,
            "phase": phase,
            "seat": seat,
            "model": getattr(request, "model", "unknown"),
            "system_prompt": system_prompt,
            "user_message": user_message,
            "metadata": getattr(request, "metadata", {}) or {},
            "format": prompt_format,
        }
        if prompt_payload is not None:
            entry["prompt_payload"] = prompt_payload

        self.context_entries.append(entry)

    def record_proposal(
        self,
        *,
        round_num: int,
        phase: str,
        leader: int,
        payload: Dict[str, Any],
        usage: Optional[Dict[str, Any]],
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        canonical_payload = self._canonicalize_payload(payload)
        members = canonical_payload.get("action", {}).get("members", [])
        proposal_id = f"P{round_num}-s{leader}"
        self._add_event(
            {
                "round": round_num,
                "phase": phase,
                "leader": leader,
                "output": canonical_payload,
                "proposal_id": proposal_id,
                "cost": usage or {},
                "reasoning": _attach_reasoning(reasoning, seat=leader),
            }
        )

        # Also record in analysis writer
        if self.analysis_writer:
            prompt, system_prompt = self._get_context(round_num, phase, leader)
            self.analysis_writer.record_proposal(
                round_num=round_num,
                phase=phase,
                seat=leader,
                prompt=prompt,
                system_prompt=system_prompt,
                action=canonical_payload.get("action"),
                reasoning=reasoning,
            )

    def record_discussion(
        self,
        *,
        round_num: int,
        seat: int,
        payload: Dict[str, Any],
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._discussion_round != round_num:
            self._discussion_round = round_num
            self._discussion_buffer = []
        entry = {"seat": seat, "speech": payload.get("speech", "")}
        thinking = payload.get("thinking")
        if thinking:
            entry["thinking"] = thinking
        if reasoning:
            entry["reasoning"] = _attach_reasoning(reasoning, seat=seat)
        self._discussion_buffer.append(entry)

        # Also record in analysis writer immediately
        if self.analysis_writer:
            prompt, system_prompt = self._get_context(round_num, "DISCUSSION", seat)
            self.analysis_writer.record_discussion(
                round_num=round_num,
                phase="DISCUSSION",
                seat=seat,
                prompt=prompt,
                system_prompt=system_prompt,
                thinking=thinking or "",
                speech=payload.get("speech", ""),
                reasoning=reasoning,
            )

        if len(self._discussion_buffer) == self.total_seats:
            reasoning_entries = [item["reasoning"] for item in self._discussion_buffer if "reasoning" in item]
            self._add_event(
                {
                    "round": round_num,
                    "phase": "DISCUSSION",
                    "outputs": list(self._discussion_buffer),
                    "reasoning": {"entries": reasoning_entries} if reasoning_entries else None,
                }
            )
            self._discussion_round = None
            self._discussion_buffer = []

    def record_summary(
        self,
        *,
        round_num: int,
        seat: int,
        payload: Dict[str, Any],
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {"seat": seat, "speech": payload.get("speech", "")}
        thinking = payload.get("thinking")
        if thinking:
            entry["thinking"] = thinking
        if reasoning:
            entry["reasoning"] = _attach_reasoning(reasoning, seat=seat)
        self._add_event(
            {
                "round": round_num,
                "phase": "SUMMARY",
                "outputs": [entry],
                "reasoning": {"entries": [entry["reasoning"]]} if "reasoning" in entry else None,
            }
        )

    def record_vote(
        self,
        *,
        round_num: int,
        seat: int,
        value: str,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._vote_round is None:
            if self._vote_buffer:
                raise ValueError("Vote buffer was not cleared before starting a new vote phase")
            self._vote_round = round_num
            self._vote_buffer.clear()
            self._vote_seen_seats = set()
        elif self._vote_round != round_num:
            raise ValueError(
                f"Votes already recording for round {self._vote_round}; got vote for round {round_num}"
            )
        if seat not in self._seat_range:
            raise ValueError(f"Seat {seat} out of bounds for vote recording")
        if seat in self._vote_seen_seats:
            raise ValueError(f"Duplicate vote detected for seat {seat} in round {round_num}")
        vote_entry = {"seat": seat, "value": value}
        if reasoning:
            vote_entry["reasoning"] = _attach_reasoning(reasoning, seat=seat)
        self._vote_buffer.append(vote_entry)
        self._vote_seen_seats.add(seat)

        # Also record in analysis writer immediately
        if self.analysis_writer:
            prompt, system_prompt = self._get_context(round_num, "VOTE", seat)
            self.analysis_writer.record_vote(
                round_num=round_num,
                seat=seat,
                prompt=prompt,
                system_prompt=system_prompt,
                action=value,
                reasoning=reasoning,
            )
        if len(self._vote_buffer) == self.total_seats:
            normalized_votes = sorted(self._vote_buffer, key=lambda item: item["seat"])
            seats = [item["seat"] for item in normalized_votes]
            if seats != list(range(1, self.total_seats + 1)):
                raise ValueError(
                    "Vote collection produced mismatched seat ordering: "
                    f"expected {list(range(1, self.total_seats + 1))} got {seats}"
                )
            approve_count = 0
            reject_count = 0
            for item in normalized_votes:
                vote_value = item.get("value", "")
                # Handle both enum and string values
                if hasattr(vote_value, 'value'):
                    label = vote_value.value  # Extract enum value
                else:
                    label = str(vote_value).upper()

                if label == "APPROVE":
                    approve_count += 1
                elif label == "REJECT":
                    reject_count += 1
                else:
                    raise ValueError(f"Unexpected vote label '{label}' for seat {item['seat']}")
                item["value"] = label
            tally = {
                "approve": approve_count,
                "reject": reject_count,
                "approved": approve_count >= (self.total_seats // 2 + 1),
            }
            reasoning_entries = [item["reasoning"] for item in normalized_votes if "reasoning" in item]
            self._add_event(
                {
                    "round": round_num,
                    "phase": "VOTE",
                    "outputs": normalized_votes,
                    "tally": tally,
                    "reasoning": {"entries": reasoning_entries} if reasoning_entries else None,
                }
            )
            self._vote_round = None
            self._vote_buffer = []
            self._vote_seen_seats = set()

    def record_mission(
        self,
        *,
        round_num: int,
        seat: int,
        value: str,
        expected: int,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._mission_round != round_num:
            self._mission_round = round_num
            self._mission_expected = expected
            self._mission_buffer = []
        mission_entry = {"seat": seat, "value": value}
        if reasoning:
            mission_entry["reasoning"] = _attach_reasoning(reasoning, seat=seat)
        self._mission_buffer.append(mission_entry)
        if self._mission_expected and len(self._mission_buffer) >= self._mission_expected:
            fails = sum(1 for item in self._mission_buffer if item["value"] == "FAIL")
            reveal = {"fails": fails, "result": "FAIL" if fails > 0 else "SUCCESS"}
            reasoning_entries = [item["reasoning"] for item in self._mission_buffer if "reasoning" in item]
            self._add_event(
                {
                    "round": round_num,
                    "phase": "MISSION",
                    "outputs": list(self._mission_buffer),
                    "reveal": reveal,
                    "reasoning": {"entries": reasoning_entries} if reasoning_entries else None,
                }
            )
            self._mission_round = None
            self._mission_expected = None
            self._mission_buffer = []

    def record_conclave(
        self,
        *,
        round_num: int,
        seat: int,
        speech: str,
        expected: int,
        thinking: Optional[str] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._conclave_round != round_num:
            self._conclave_round = round_num
            self._conclave_expected = expected
            self._conclave_buffer = []
        entry = {"round": round_num, "seat": seat, "speech": speech}
        if thinking:
            entry["thinking"] = thinking
        if reasoning:
            entry["reasoning"] = _attach_reasoning(reasoning, seat=seat)
        self._conclave_buffer.append(entry)
        if self._conclave_expected and len(self._conclave_buffer) >= self._conclave_expected:
            self.evil_conclave_messages.extend(self._conclave_buffer)
            self._conclave_round = None
            self._conclave_expected = None
            self._conclave_buffer = []

    def record_assassin_guess(
        self,
        *,
        seat: int,
        target: int,
        correct: bool,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.assassin_guess = {
            "seat": seat,
            "target": target,
            "correct": correct,
            "reasoning": _attach_reasoning(reasoning, seat=seat) if reasoning else None,
        }

    def ensure_mission_events(self, missions: Iterable[Any]) -> None:
        existing = {(event["round"], event["phase"]) for event in self.events}
        for mission in missions:
            key = (mission.round_num, "MISSION")
            if key not in existing:
                reveal = {
                    "fails": getattr(mission, "fails", 0),
                    "result": getattr(getattr(mission, "result", None), "value", getattr(mission, "result", "UNKNOWN")),
                }
                self._add_event(
                    {
                        "round": mission.round_num,
                        "phase": "MISSION",
                        "outputs": [],
                        "reveal": reveal,
                    }
                )
                existing.add(key)

    # ------------------------------------------------------------------
    # Transcript emission
    # ------------------------------------------------------------------

    def build_transcript(
        self,
        *,
        state: Any,
        seat_models: Dict[int, str],
        seed: Optional[int],
        default_model: str,
    ) -> TranscriptPaths:
        writer = TranscriptWriter.from_game_state(state, seat_models=seat_models)

        # Ensure every mission has a transcript entry (including auto-fail missions).
        self.ensure_mission_events(getattr(state, "missions", []) or [])

        events_by_round: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for event in sorted(self.events, key=lambda item: item["order"]):
            events_by_round[event["round"]].append(event)

        missions = sorted((getattr(state, "missions", []) or []), key=lambda mission: mission.round_num)
        cumulative_good = 0
        cumulative_evil = 0
        round_scores: Dict[int, Dict[str, int]] = {}
        for mission in missions:
            result = getattr(mission.result, "value", mission.result)
            if result == MissionOutcome.SUCCESS.value:
                cumulative_good += 1
            else:
                cumulative_evil += 1
            round_scores[mission.round_num] = {"good": cumulative_good, "evil": cumulative_evil}

        for round_num in sorted(events_by_round.keys()):
            writer.start_round(round_num)
            for event in events_by_round[round_num]:
                phase = event["phase"]
                if phase in {"PROPOSAL_DRAFT", "PROPOSAL_FINAL"}:
                    writer.record_proposal_phase(
                        phase=phase,
                        leader=event["leader"],
                        output=event["output"],
                        cost=event.get("cost"),
                        reasoning=event.get("reasoning"),
                    )
                elif phase == "DISCUSSION":
                    writer.record_discussion_phase(event["outputs"], reasoning=event.get("reasoning"))
                elif phase == "SUMMARY":
                    writer.record_summary_phase(event["outputs"], reasoning=event.get("reasoning"))
                elif phase == "VOTE":
                    writer.record_vote_phase(
                        event["outputs"],
                        event["tally"],
                        reasoning=event.get("reasoning"),
                    )
                elif phase == "MISSION":
                    writer.record_mission_phase(
                        event.get("outputs", []),
                        event.get("reveal", {}),
                        reasoning=event.get("reasoning"),
                    )
            score = round_scores.get(round_num, {"good": cumulative_good, "evil": cumulative_evil})
            writer.record_round_end_phase(score)

        if self.evil_conclave_messages:
            writer.record_evil_conclave(
                [
                    EvilConclaveMessage(
                        round_num=msg["round"],
                        seat=msg["seat"],
                        speech=msg["speech"],
                        thinking=msg.get("thinking"),
                        reasoning=msg.get("reasoning"),
                    )
                    for msg in self.evil_conclave_messages
                ]
            )
        elif getattr(state, "evil_conclave_messages", None):
            writer.record_evil_conclave(getattr(state, "evil_conclave_messages"))

        assassin = self.assassin_guess or (
            getattr(state, "assassin_guess", None) and {
                "seat": state.assassin_guess.seat,
                "target": state.assassin_guess.target,
                "correct": state.assassin_guess.correct,
            }
        )
        if assassin:
            writer.record_assassin_guess(
                AssassinGuess(
                    seat=assassin["seat"],
                    target=assassin["target"],
                    correct=assassin["correct"],
                    reasoning=assassin.get("reasoning"),
                )
            )

        writer.mark_winner(getattr(state, "winner", Outcome.EVIL_WIN))
        transcript_path = writer.flush()

        # Flush analysis writer if initialized
        analysis_path = None
        if self.analysis_writer:
            self.analysis_writer.mark_winner(getattr(state, "winner", Outcome.EVIL_WIN))
            analysis_path = self.analysis_writer.flush()
            LOGGER.info("analysis.saved", path=str(analysis_path))

        markdown_path = self._write_markdown_report(
            state=state,
            seat_models=seat_models,
            seed=seed,
            transcript_path=transcript_path,
            events_by_round=events_by_round,
            round_scores=round_scores,
            default_model=default_model,
        )
        context_markdown_path = self._write_context_markdown(
            state=state,
            seed=seed,
            transcript_path=transcript_path,
        )
        LOGGER.info("transcript.saved", path=str(transcript_path), markdown=str(markdown_path))
        return TranscriptPaths(
            json_path=transcript_path,
            markdown_path=markdown_path,
            context_markdown_path=context_markdown_path,
        )

    # ------------------------------------------------------------------

    def _add_event(self, event: Dict[str, Any]) -> None:
        event["order"] = self._sequence
        self._sequence += 1
        self.events.append(event)

    def _write_markdown_report(
        self,
        *,
        state: Any,
        seat_models: Dict[int, str],
        seed: Optional[int],
        transcript_path: Path,
        events_by_round: Dict[int, List[Dict[str, Any]]],
        round_scores: Dict[int, Dict[str, int]],
        default_model: str,
    ) -> Path:
        def seat_model(seat: int) -> str:
            return seat_models.get(seat, default_model)

        winner = getattr(getattr(state, "winner", None), "value", getattr(state, "winner", "Unknown"))
        lines: List[str] = []
        lines.append("# Avalon Match Report")
        lines.append("")
        lines.append(f"- Seed: {seed if seed is not None else getattr(state, 'seed', 'unknown')}")
        lines.append(f"- Winner: {winner}")
        lines.append(f"- JSON Transcript: {transcript_path.name}")
        lines.append("")

        max_round = max(events_by_round.keys(), default=0)
        for round_num in range(1, max_round + 1):
            events = events_by_round.get(round_num, [])
            lines.append(f"## Round {round_num}")
            leader_event = next((ev for ev in events if "leader" in ev), None)
            if leader_event:
                leader = leader_event.get("leader")
                lines.append(f"- Leader: Seat {leader} ({seat_model(leader)})")
            prev_score = round_scores.get(round_num - 1, {"good": 0, "evil": 0})
            lines.append(
                f"- Score entering round: Good {prev_score['good']} vs Evil {prev_score['evil']}"
            )

            for event in events:
                phase = event["phase"]
                if phase in {"PROPOSAL_DRAFT", "PROPOSAL_FINAL"}:
                    members = event["output"].get("action", {}).get("members", [])
                    lines.append(
                        f"  - {phase}: Leader seat {event['leader']} proposes {members}"
                    )
                elif phase == "DISCUSSION":
                    lines.append("  - DISCUSSION:")
                    for speech in event.get("outputs", []):
                        seat = speech.get("seat")
                        text = speech.get("speech", "")
                        lines.append(
                            f"    - Seat {seat} ({seat_model(seat)}): {text}"
                        )
                        thinking = speech.get("thinking")
                        if thinking:
                            lines.append(
                                f"      [thinking] {thinking}"
                            )
                elif phase == "SUMMARY":
                    lines.append("  - SUMMARY:")
                    for summary in event.get("outputs", []):
                        seat = summary.get("seat")
                        text = summary.get("speech", "")
                        lines.append(
                            f"    - Seat {seat} ({seat_model(seat)}): {text}"
                        )
                        thinking = summary.get("thinking")
                        if thinking:
                            lines.append(
                                f"      [thinking] {thinking}"
                            )
                elif phase == "VOTE":
                    tally = event.get("tally", {})
                    lines.append(
                        f"  - VOTE: tally approve={tally.get('approve', 0)} reject={tally.get('reject', 0)} approved={tally.get('approved')}"
                    )
                    for vote in event.get("outputs", []):
                        seat = vote.get("seat")
                        value = vote.get("value")
                        lines.append(
                            f"    - Seat {seat} ({seat_model(seat)}): {value}"
                        )
                elif phase == "MISSION":
                    reveal = event.get("reveal", {})
                    lines.append(
                        f"  - MISSION: result={reveal.get('result')} fails={reveal.get('fails')}"
                    )
                    for mission_entry in event.get("outputs", []):
                        seat = mission_entry.get("seat")
                        value = mission_entry.get("value")
                        lines.append(
                            f"    - Seat {seat} ({seat_model(seat)}): {value}"
                        )
                elif phase == "ROUND_END":
                    score = event.get("output", {})
                    lines.append(
                        f"  - ROUND_END: Good {score.get('good')} vs Evil {score.get('evil')}"
                    )

            lines.append("")

        if self.evil_conclave_messages:
            lines.append("## Evil Conclave")
            for message in self.evil_conclave_messages:
                seat = message.get("seat")
                text = message.get("speech", "")
                lines.append(
                    f"- Round {message.get('round')}: Seat {seat} ({seat_model(seat)}): {text}"
                )
                thinking = message.get("thinking")
                if thinking:
                    lines.append(f"  [thinking] {thinking}")
            lines.append("")

        assassin = self.assassin_guess or (
            getattr(state, "assassin_guess", None)
            and {
                "seat": state.assassin_guess.seat,
                "target": state.assassin_guess.target,
                "correct": state.assassin_guess.correct,
            }
        )
        if assassin:
            lines.append("## Assassin Guess")
            lines.append(
                f"- Seat {assassin['seat']} ({seat_model(assassin['seat'])}) targeted seat {assassin['target']} -> {'correct' if assassin['correct'] else 'incorrect'}"
            )

        markdown_path = transcript_path.with_suffix(".md")
        markdown_path.write_text("\n".join(lines).strip() + "\n")
        return markdown_path

    def _write_context_markdown(
        self,
        *,
        state: Any,
        seed: Optional[int],
        transcript_path: Path,
    ) -> Path:
        lines: List[str] = []
        lines.append("# Agent Request Context")
        lines.append("")
        lines.append(f"- Seed: {seed if seed is not None else getattr(state, 'seed', 'unknown')}")
        lines.append(f"- Total calls captured: {len(self.context_entries)}")
        truncated = getattr(state, "truncated", False)
        round_limit = getattr(state, "round_limit", None)
        if truncated or round_limit:
            lines.append(
                f"- Truncated run: {'yes' if truncated else 'no'}"
                + (f" (round limit={round_limit})" if round_limit is not None else "")
            )
        lines.append("")

        for entry in self.context_entries:
            lines.append(
                f"## Call {entry['order']}: Round {entry['round']} – {entry['phase']} – Seat {entry['seat']}"
            )
            lines.append(f"- Model: {entry.get('model', 'unknown')}")
            prompt_format = entry.get("format", "json")
            lines.append(f"- Prompt format: {prompt_format}")
            meta_context = (entry.get("metadata", {}) or {}).get("context", {})
            reasoning_enabled = meta_context.get("reasoning_enabled")
            if reasoning_enabled is not None:
                lines.append(f"- Reasoning enabled: {bool(reasoning_enabled)}")
            options = (entry.get("metadata", {}) or {}).get("options")
            if options:
                options_json = json.dumps(options, indent=2, sort_keys=True)
                lines.append("- Options:")
                lines.append("```json")
                lines.extend(options_json.splitlines())
                lines.append("```")

            lines.append("### System Prompt")
            lines.append("```text")
            lines.extend((entry.get("system_prompt") or "").splitlines())
            lines.append("```")
            lines.append("### User Payload")
            user_message = entry.get("user_message") or ""
            block_lang = "json" if prompt_format == "json" else "text"
            lines.append(f"```{block_lang}")
            lines.extend(user_message.splitlines())
            lines.append("```")
            lines.append("")

        context_path = transcript_path.with_name(f"{transcript_path.stem}_contexts.md")
        context_path.write_text("\n".join(lines).strip() + "\n")
        return context_path


def configure_logging() -> None:
    global _configured_logging
    if _configured_logging:
        return
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )
    _configured_logging = True


def make_phase_services(
    *,
    client: ModelProvider,
    builder: ContextBuilder,
    recorder: SimulationRecorder,
    seed: Optional[int],
    human_service: Optional[HumanPlayerService] = None,
    agent_overrides: Optional[Dict[int, BaseSeatAgent]] = None,
) -> PhaseServices:
    def _phase_value(phase: Phase | str) -> str:
        return phase.value if hasattr(phase, "value") else str(phase)

    human_seat_set = set(human_service.human_seats) if human_service else set()
    seat_agents: AgentRegistry = {}

    # Calculate total seats dynamically
    total_seats = TOTAL_SEATS  # default fallback
    if agent_overrides:
        total_seats = max(agent_overrides.keys())
    elif builder.config.seat_models:
        total_seats = max(builder.config.seat_models.keys())

    for seat in range(1, total_seats + 1):
        if agent_overrides and seat in agent_overrides:
            seat_agents[seat] = agent_overrides[seat]
            continue
        if seat in human_seat_set and human_service is not None:
            seat_agents[seat] = HumanSeatAgent(seat, service=human_service, recorder=recorder)
        else:
            seat_agents[seat] = ModelSeatAgent(
                seat,
                client=client,
                builder=builder,
                recorder=recorder,
                seed=seed,
                logger=LOGGER,
                human_service=human_service,
            )

    async def _call_agent(agent: BaseSeatAgent, method: str, *args: Any) -> Any:
        async_method = getattr(agent, f"{method}_async", None)
        if async_method is not None:
            return await async_method(*args)  # type: ignore[misc]
        func = getattr(agent, method)
        return await asyncio.to_thread(func, *args)

    async def proposal(state: Any, stage: Phase) -> Any:
        return await _call_agent(seat_agents[state.leader], "proposal", state, stage)

    async def proposal_summary(state: Any, seat: int) -> Any:
        return await _call_agent(seat_agents[seat], "proposal_summary", state, seat)

    async def discussion(state: Any, seat: int) -> Any:
        return await _call_agent(seat_agents[seat], "discussion", state, seat)

    async def vote(state: Any, seat: int) -> Any:
        return await _call_agent(seat_agents[seat], "vote", state, seat)

    async def mission(state: Any, seat: int) -> Any:
        return await _call_agent(seat_agents[seat], "mission", state, seat)

    async def evil_conclave(state: Any, seat: int) -> Any:
        return await _call_agent(seat_agents[seat], "evil_conclave", state, seat)

    async def assassin_guess(state: Any, seat: int) -> Any:
        return await _call_agent(seat_agents[seat], "assassin_guess", state, seat)

    services = PhaseServices(
        proposal=proposal,
        proposal_summary=proposal_summary,
        discussion=discussion,
        vote=vote,
        mission=mission,
        evil_conclave=evil_conclave,
        assassin_guess=assassin_guess,
        agents=seat_agents,
    )

    if human_service:
        services._human_service = human_service  # type: ignore[attr-defined]
        services._recorder = recorder  # type: ignore[attr-defined]

    return services


def configure_human_logging() -> None:
    """Configure minimal logging for human player mode."""
    import logging
    # Reduce log level to WARNING to hide info messages
    logging.getLogger().setLevel(logging.WARNING)
    # Disable structlog's console output for human mode but keep basic processors
    # to handle bound context properly
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
        cache_logger_on_first_use=True,
    )


@app.command("simulate")
def simulate(
    seed: Optional[int] = typer.Option(None, help="Seed for deterministic simulation"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to seat/model configuration JSON"),
    max_rounds: Optional[int] = typer.Option(
        2,
        help="Maximum number of rounds to play before stopping early (use 0 for a full game)",
    ),
    human_mode: bool = typer.Option(False, "--human", help="Enable human player mode"),
    human_seat: int = typer.Option(DEFAULT_HUMAN_SEAT, "--human-seat", help="Seat number for human player (1-5)"),
) -> None:
    """Run a full Avalon match and store the transcript under runs/."""

    load_dotenv()
    configure_logging()

    round_limit = max_rounds if max_rounds and max_rounds > 0 else None

    LOGGER.info("simulation.start", seed=seed, config=str(config), max_rounds=round_limit, human_mode=human_mode, human_seat=human_seat if human_mode else None)

    context_config = load_context_config(config)

    # Validate human_seat is within valid range
    max_seats = max(context_config.seat_models.keys()) if context_config.seat_models else TOTAL_SEATS
    if human_mode and not (1 <= human_seat <= max_seats):
        typer.echo(f"Error: human-seat must be between 1 and {max_seats}")
        raise typer.Exit(code=1)
    builder = ContextBuilder(config=context_config)
    base_recorder = SimulationRecorder()

    # Setup human player components if enabled
    narrator = None
    recorder = base_recorder
    human_seats = (human_seat,) if human_mode else tuple()
    if human_mode:
        narrator = GameNarrator(human_seats=human_seats, total_seats=max_seats)
        recorder = HumanRecorder(base_recorder, narrator, human_seats=human_seats, total_seats=max_seats)
        configure_human_logging()

    # Log reasoning configuration status
    reasoning_config = context_config.reasoning
    if reasoning_config.enabled:
        enabled_models = [model for model, cfg in reasoning_config.per_model_config.items() if cfg.get("enabled")]
        LOGGER.info(
            "reasoning.enabled",
            total_models=len(context_config.seat_models),
            reasoning_models=len(enabled_models),
            enabled_models=enabled_models,
            phases=list(reasoning_config.phases.keys()) if reasoning_config.phases else "all"
        )
    else:
        LOGGER.info("reasoning.disabled")

    try:
        # Check if we have seat_providers configuration for multi-provider support
        if context_config.seat_providers:
            client = ProviderFactory.create(
                "multi",
                seat_providers=context_config.seat_providers,
                default_provider=context_config.default_provider
            )
        else:
            # Use the default provider for all seats
            client = ProviderFactory.create(context_config.default_provider)
        
        with client:
            human_service = HumanPlayerService(narrator, human_seats=human_seats) if narrator else None

            # Initialize analysis writer BEFORE creating services
            from ..core.fsm import GameState
            from ..utils.rng import build_rng
            initial_state = GameState(seed=seed, rng=build_rng(seed=seed))
            base_recorder.init_analysis_writer(initial_state, seed=str(seed) if seed else None, seat_models=context_config.seat_models)

            services = make_phase_services(client=client, builder=builder, recorder=recorder, seed=seed, human_service=human_service)
            state = run_game(seed=seed, services=services, max_rounds=round_limit)
    except ProviderError as exc:  # pragma: no cover - I/O failure path
        LOGGER.error("simulation.failed", error=str(exc))
        raise typer.Exit(code=1) from exc

    recorder.ensure_mission_events(getattr(state, "missions", []) or [])

    transcript_paths = recorder.build_transcript(
        state=state,
        seat_models=context_config.seat_models,
        seed=seed,
        default_model=context_config.default_model,
    )

    winner = getattr(state.winner, "value", state.winner)
    # Show final game summary for human players
    if human_mode and narrator:
        narrator.game_end(winner, getattr(state, 'roles', {}))

    if not human_mode:
        if getattr(state, "truncated", False):
            LOGGER.info("simulation.truncated", rounds_played=getattr(state, "round_num", None) - 1, round_limit=round_limit)
        LOGGER.info(
            "simulation.complete",
            winner=winner,
            transcript=str(transcript_paths.json_path),
            markdown=str(transcript_paths.markdown_path),
            contexts=str(transcript_paths.context_markdown_path) if transcript_paths.context_markdown_path else None,
        )
        typer.echo(
            "\n".join(
                [
                    f"Simulation complete: winner={winner}.",
                    f"Transcript JSON saved to {transcript_paths.json_path}",
                    f"Markdown summary saved to {transcript_paths.markdown_path}",
                    (
                        f"Context capture saved to {transcript_paths.context_markdown_path}"
                        if transcript_paths.context_markdown_path
                        else ""
                    ),
                ]
            ).strip()
        )
    else:
        # Minimal output for human mode
        typer.echo(f"\nTranscript saved to {transcript_paths.json_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
