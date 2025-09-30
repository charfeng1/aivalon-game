"""Abstractions and implementations for seat agents interacting with the engine."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Protocol

from .core.fsm import Phase
from .core.schemas import (
    AssassinGuessPayload,
    DiscussionPayload,
    EvilConclavePayload,
    MissionPayload,
    ProposalPayload,
    VotePayload,
    create_default_mission_payload,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from .core.context import ContextBuilder
    from .human_player import HumanPlayerService
    from .providers.providers import ModelProvider, ProviderResponse
    from .core.schemas import PhasePayload


class SeatAgent(Protocol):
    """Minimal protocol describing a seat actor."""

    seat: int

    def add_memory(self, key: str, value: str) -> None:
        """Store a persistent memory snippet for the agent."""

    def remove_memory(self, key: str) -> None:
        """Remove a previously stored memory snippet."""

    def clear_memory(self) -> None:
        """Remove all stored memory snippets."""

    def memory_snapshot(self) -> List[str]:
        """Return the stored memory snippets in insertion order."""


@dataclass(slots=True)
class AgentMemory:
    """Container that tracks persistent snippets for a seat agent."""

    entries: MutableMapping[str, str]

    def __init__(self) -> None:
        self.entries = OrderedDict()

    def add(self, key: str, value: str) -> None:
        self.entries[str(key)] = str(value)

    def remove(self, key: str) -> None:
        self.entries.pop(str(key), None)

    def clear(self) -> None:
        self.entries.clear()

    def snapshot(self) -> List[str]:
        return list(self.entries.values())


class BaseSeatAgent:
    """Concrete helper implementing the memory protocol for agents."""

    def __init__(self, seat: int) -> None:
        self.seat = seat
        self._memory = AgentMemory()

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def add_memory(self, key: str, value: str) -> None:
        """Store a memory snippet available for future prompts."""

        self._memory.add(key, value)

    def remove_memory(self, key: str) -> None:
        """Remove a memory snippet if present."""

        self._memory.remove(key)

    def clear_memory(self) -> None:
        """Remove all stored memory snippets."""

        self._memory.clear()

    def memory_snapshot(self) -> List[str]:
        """Expose stored snippets in insertion order."""

        return self._memory.snapshot()


class ModelSeatAgent(BaseSeatAgent):
    """LLM-backed seat agent that produces payloads via the transport client."""

    def __init__(
        self,
        seat: int,
        *,
        client: "ModelProvider",
        builder: "ContextBuilder",
        recorder: Any,
        seed: Optional[int],
        logger: Any,
        human_service: Optional["HumanPlayerService"] = None,
    ) -> None:
        super().__init__(seat)
        self._client = client
        self._builder = builder
        self._recorder = recorder
        self._seed = seed
        self._logger = logger
        self._skip_vote_recording = bool(human_service and hasattr(recorder, "narrator"))
        self.records_vote = not self._skip_vote_recording

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _phase_name(self, phase: Phase | str) -> str:
        return phase.value if hasattr(phase, "value") else str(phase)

    def _build_context(self, state: Any, phase: Phase) -> Any:
        return self._builder.build_phase_context(
            state=state,
            phase=phase,
            seat=self.seat,
            seed=self._seed,
            memory=self.memory_snapshot(),
        )

    def _record_context(self, state: Any, phase: Phase, context: Any) -> None:
        self._recorder.record_context(
            round_num=getattr(state, "round_num", 0),
            phase=self._phase_name(phase),
            seat=self.seat,
            request=context.request,
            prompt_payload=context.prompt_payload,
            prompt_format="dsl",
        )

    def _call_client(self, context: Any) -> "ProviderResponse[Any]":
        return self._client.call_phase(context.request)

    def _handle_violations(self, response: "ProviderResponse[Any]", phase: Phase) -> None:
        if not response.violations:
            return
        phase_name = self._phase_name(phase)
        for violation in response.violations:
            self._builder.register_violation(phase=phase_name, seat=self.seat, code=violation)

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def proposal(self, state: Any, stage: Phase) -> ProposalPayload:
        context = self._build_context(state, stage)
        self._record_context(state, stage, context)
        response = self._call_client(context)
        self._handle_violations(response, stage)
        # Extract reasoning info if available (OpenRouter-specific)
        reasoning_info = {}
        if hasattr(self._client, '_extract_reasoning_info'):
            reasoning_info = self._client._extract_reasoning_info(response.raw_response)
        if reasoning_info.get("reasoning_tokens"):
            self._logger.info(
                "reasoning.used",
                phase=self._phase_name(stage),
                seat=self.seat,
                reasoning_tokens=reasoning_info.get("reasoning_tokens"),
                has_reasoning_text=bool(reasoning_info.get("reasoning")),
            )
        payload = response.payload
        self._recorder.record_proposal(
            round_num=getattr(state, "round_num", 0),
            phase=self._phase_name(stage),
            leader=self.seat,
            payload=payload.model_dump(),
            usage=response.usage,
            reasoning=response.reasoning,
        )
        return payload

    def proposal_summary(self, state: Any, seat: int) -> DiscussionPayload:
        context = self._build_context(state, Phase.PROPOSAL_SUMMARY)
        self._record_context(state, Phase.PROPOSAL_SUMMARY, context)
        response = self._call_client(context)
        self._handle_violations(response, Phase.PROPOSAL_SUMMARY)
        payload = response.payload
        self._recorder.record_summary(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            payload=payload.model_dump(),
            reasoning=response.reasoning,
        )
        return payload

    def discussion(self, state: Any, seat: int) -> DiscussionPayload:
        context = self._build_context(state, Phase.DISCUSSION)
        self._record_context(state, Phase.DISCUSSION, context)
        response = self._call_client(context)
        self._handle_violations(response, Phase.DISCUSSION)
        payload = response.payload
        self._recorder.record_discussion(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            payload=payload.model_dump(),
            reasoning=response.reasoning,
        )
        return payload

    def vote(self, state: Any, seat: int) -> VotePayload:
        context = self._build_context(state, Phase.VOTE)
        self._record_context(state, Phase.VOTE, context)
        response = self._call_client(context)
        self._handle_violations(response, Phase.VOTE)
        payload = response.payload
        if not self._skip_vote_recording:
            payload_dict = payload.model_dump()
            value = payload_dict.get("action", {}).get("value", "ABSTAIN")
            self._recorder.record_vote(
                round_num=getattr(state, "round_num", 0),
                seat=seat,
                value=value,
                reasoning=response.reasoning,
            )
        return payload

    def mission(self, state: Any, seat: int) -> MissionPayload:
        context = self._build_context(state, Phase.MISSION)
        self._record_context(state, Phase.MISSION, context)
        log = self._logger.bind(phase=Phase.MISSION.value, seat=seat)

        def _handle_mission_fallback(exc: Exception, *, code: str, level: str = "warning") -> MissionPayload:
            log_method = getattr(log, level, log.warning)
            log_method("mission.fallback", code=code, error=str(exc))
            self._builder.register_violation(phase=Phase.MISSION.value, seat=seat, code=code)
            fallback_payload = create_default_mission_payload()
            payload_dict = fallback_payload.model_dump()
            value = payload_dict.get("action", {}).get("value", "SUCCESS")
            expected = len(getattr(getattr(state, "current_mission", None), "members", [])) or state.current_team_size()
            self._recorder.record_mission(
                round_num=getattr(state, "round_num", 0),
                seat=seat,
                value=value,
                expected=expected,
                reasoning=None,
            )
            return fallback_payload

        try:
            response = self._call_client(context)
        except Exception as exc:  # pragma: no cover - defensive guard mirrors CLI behaviour
            from .providers.providers import ProviderError
            from .core.schemas import SchemaValidationError

            if isinstance(exc, SchemaValidationError):
                return _handle_mission_fallback(exc, code="mission_schema_error")
            if isinstance(exc, ProviderError):
                return _handle_mission_fallback(exc, code="mission_transport_error")
            if isinstance(exc, KeyboardInterrupt):
                raise
            return _handle_mission_fallback(exc, code="mission_unexpected_error", level="error")

        self._handle_violations(response, Phase.MISSION)
        payload = response.payload
        payload_dict = payload.model_dump()
        value = payload_dict.get("action", {}).get("value", "SUCCESS")
        expected = len(getattr(getattr(state, "current_mission", None), "members", [])) or state.current_team_size()
        self._recorder.record_mission(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            value=value,
            expected=expected,
            reasoning=response.reasoning,
        )
        return payload

    def evil_conclave(self, state: Any, seat: int) -> EvilConclavePayload:
        context = self._build_context(state, Phase.EVIL_CONCLAVE)
        self._record_context(state, Phase.EVIL_CONCLAVE, context)
        response = self._call_client(context)
        self._handle_violations(response, Phase.EVIL_CONCLAVE)
        payload = response.payload
        expected = len(state.evil_seats()) if hasattr(state, "evil_seats") else 0
        self._recorder.record_conclave(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            speech=payload.model_dump().get("speech", ""),
            thinking=payload.model_dump().get("thinking"),
            expected=expected,
            reasoning=response.reasoning,
        )
        return payload

    def assassin_guess(self, state: Any, seat: int) -> AssassinGuessPayload:
        context = self._build_context(state, Phase.ASSASSIN_GUESS)
        self._record_context(state, Phase.ASSASSIN_GUESS, context)
        response = self._call_client(context)
        self._handle_violations(response, Phase.ASSASSIN_GUESS)
        payload = response.payload
        payload_dict = payload.model_dump()
        target = payload_dict.get("action", {}).get("target", -1)
        correct = False
        if hasattr(state, "merlin_seat"):
            try:
                correct = target == state.merlin_seat()
            except Exception:  # pragma: no cover - defensive
                correct = False
        self._recorder.record_assassin_guess(
            seat=seat,
            target=target,
            correct=correct,
            reasoning=response.reasoning,
        )
        return payload

    async def proposal_async(self, state: Any, stage: Phase) -> ProposalPayload:
        return await asyncio.to_thread(self.proposal, state, stage)

    async def proposal_summary_async(self, state: Any, seat: int) -> DiscussionPayload:
        return await asyncio.to_thread(self.proposal_summary, state, seat)

    async def discussion_async(self, state: Any, seat: int) -> DiscussionPayload:
        return await asyncio.to_thread(self.discussion, state, seat)

    async def vote_async(self, state: Any, seat: int) -> VotePayload:
        return await asyncio.to_thread(self.vote, state, seat)

    async def mission_async(self, state: Any, seat: int) -> MissionPayload:
        return await asyncio.to_thread(self.mission, state, seat)

    async def evil_conclave_async(self, state: Any, seat: int) -> EvilConclavePayload:
        return await asyncio.to_thread(self.evil_conclave, state, seat)

    async def assassin_guess_async(self, state: Any, seat: int) -> AssassinGuessPayload:
        return await asyncio.to_thread(self.assassin_guess, state, seat)


class HumanSeatAgent(BaseSeatAgent):
    """Seat agent wrapping the CLI human player service."""

    def __init__(
        self,
        seat: int,
        *,
        service: "HumanPlayerService",
        recorder: Any,
    ) -> None:
        super().__init__(seat)
        self._service = service
        self._recorder = recorder

    def proposal(self, state: Any, stage: Phase) -> ProposalPayload:
        payload = self._service.proposal(state, stage, seat=self.seat)
        self._recorder.record_proposal(
            round_num=getattr(state, "round_num", 0),
            phase=stage.value if hasattr(stage, "value") else str(stage),
            leader=self.seat,
            payload=payload.model_dump(),
            usage={},
            reasoning=None,
        )
        return payload

    def proposal_summary(self, state: Any, seat: int) -> DiscussionPayload:
        payload = self._service.summary(state, seat)
        self._recorder.record_summary(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            payload=payload.model_dump(),
            reasoning=None,
        )
        return payload

    def discussion(self, state: Any, seat: int) -> DiscussionPayload:
        payload = self._service.discussion(state, seat)
        self._recorder.record_discussion(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            payload=payload.model_dump(),
            reasoning=None,
        )
        return payload

    def vote(self, state: Any, seat: int) -> VotePayload:
        return self._service.vote(state, seat)

    async def vote_async(self, state: Any, seat: int) -> VotePayload:
        if hasattr(self._service, "vote_async"):
            return await self._service.vote_async(state, seat)
        return await asyncio.to_thread(self.vote, state, seat)

    def mission(self, state: Any, seat: int) -> MissionPayload:
        payload = self._service.mission(state, seat)
        payload_dict = payload.model_dump()
        value = payload_dict.get("action", {}).get("value", "SUCCESS")
        expected = len(getattr(getattr(state, "current_mission", None), "members", [])) or state.current_team_size()
        self._recorder.record_mission(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            value=value,
            expected=expected,
            reasoning=None,
        )
        return payload

    async def proposal_async(self, state: Any, stage: Phase) -> ProposalPayload:
        return await asyncio.to_thread(self.proposal, state, stage)

    async def proposal_summary_async(self, state: Any, seat: int) -> DiscussionPayload:
        return await asyncio.to_thread(self.proposal_summary, state, seat)

    async def discussion_async(self, state: Any, seat: int) -> DiscussionPayload:
        return await asyncio.to_thread(self.discussion, state, seat)

    async def mission_async(self, state: Any, seat: int) -> MissionPayload:
        return await asyncio.to_thread(self.mission, state, seat)

    async def evil_conclave_async(self, state: Any, seat: int) -> EvilConclavePayload:
        return await asyncio.to_thread(self.evil_conclave, state, seat)

    async def assassin_guess_async(self, state: Any, seat: int) -> AssassinGuessPayload:
        return await asyncio.to_thread(self.assassin_guess, state, seat)

    def evil_conclave(self, state: Any, seat: int) -> EvilConclavePayload:
        payload = self._service.evil_conclave(state, seat)
        payload_dict = payload.model_dump()
        expected = len(state.evil_seats()) if hasattr(state, "evil_seats") else 0
        self._recorder.record_conclave(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            speech=payload_dict.get("speech", ""),
            thinking=payload_dict.get("thinking"),
            expected=expected,
            reasoning=None,
        )
        return payload

    def assassin_guess(self, state: Any, seat: int) -> AssassinGuessPayload:
        payload = self._service.assassin_guess(state, seat)
        payload_dict = payload.model_dump()
        target = payload_dict.get("action", {}).get("target", -1)
        correct = False
        if hasattr(state, "merlin_seat"):
            try:
                correct = target == state.merlin_seat()
            except Exception:  # pragma: no cover - defensive
                correct = False
        self._recorder.record_assassin_guess(
            seat=seat,
            target=target,
            correct=correct,
            reasoning=None,
        )
        return payload


AgentRegistry = Dict[int, SeatAgent]
"""Convenience alias used for storing seat-to-agent mappings."""
