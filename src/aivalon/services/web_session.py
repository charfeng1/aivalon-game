"""Interactive web session primitives for the Avalon web dashboard."""

from __future__ import annotations

import re
import time
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents import BaseSeatAgent, ModelSeatAgent
from ..services.cli import SimulationRecorder, LOGGER, TranscriptPaths
from ..core.context import ContextBuilder, load_context_config
from ..core.fsm import Phase, PhaseServices, run_game_async
from ..core.roles import ROLE_NAMES_ZH, Role, default_role_card_provider, is_evil_player
from ..core.schemas import (
    AssassinGuessPayload,
    AssassinGuessAction,
    DiscussionPayload,
    EvilConclavePayload,
    MissionPayload,
    MissionOutcome,
    MissionAction,
    Phase as SchemaPhase,
    ProposalPayload,
    ProposeAction,
    VotePayload,
    VoteAction,
    VoteValue,
    validate_payload,
)
from ..providers.providers import ProviderResponse, ProviderError


@dataclass(slots=True)
class PendingRequest:
    """Represents a human decision that is awaiting frontend input."""

    request_id: str
    seat: int
    phase: str
    options: Dict[str, Any]
    instructions: str
    state_snapshot: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class SessionModelSeatAgent(ModelSeatAgent):
    """Model-backed agent that mirrors :class:`ModelSeatAgent` but reports state updates."""

    def __init__(
        self,
        seat: int,
        *,
        session: "GameSession",
        client: Any,
        builder: ContextBuilder,
        recorder: SimulationRecorder,
        seed: Optional[int],
    ) -> None:
        super().__init__(
            seat,
            client=client,
            builder=builder,
            recorder=recorder,
            seed=seed,
            logger=LOGGER,
            human_service=None,
        )
        self._session = session

    def _record_context(self, state: Any, phase: Phase, context: Any) -> None:  # type: ignore[override]
        self._session.update_state(state)
        super()._record_context(state, phase, context)

    async def discussion_async(self, state: Any, seat: int) -> DiscussionPayload:
        # Track that this seat is currently thinking (waiting for API call)
        self._session.set_thinking_seat(seat)
        try:
            return await super().discussion_async(state, seat)
        finally:
            # Clear the thinking seat after the API call completes
            self._session.clear_thinking_seat()

    async def proposal_summary_async(self, state: Any, seat: int) -> DiscussionPayload:
        # Track that this seat is currently thinking (waiting for API call)
        self._session.set_thinking_seat(seat)
        try:
            return await super().proposal_summary_async(state, seat)
        finally:
            # Clear the thinking seat after the API call completes
            self._session.clear_thinking_seat()

    async def evil_conclave_async(self, state: Any, seat: int) -> EvilConclavePayload:
        # Track that this seat is currently thinking (waiting for API call)
        self._session.set_thinking_seat(seat)
        try:
            return await super().evil_conclave_async(state, seat)
        finally:
            # Clear the thinking seat after the API call completes
            self._session.clear_thinking_seat()

    async def proposal_async(self, state: Any, stage: Phase) -> ProposalPayload:
        # Track that this seat is currently thinking (waiting for API call)
        self._session.set_thinking_seat(self.seat)
        try:
            return await super().proposal_async(state, stage)
        finally:
            # Clear the thinking seat after the API call completes
            self._session.clear_thinking_seat()

    async def vote_async(self, state: Any, seat: int) -> VotePayload:
        # Track that this seat is currently thinking (waiting for API call)
        self._session.set_thinking_seat(seat)
        try:
            return await super().vote_async(state, seat)
        finally:
            # Clear the thinking seat after the API call completes
            self._session.clear_thinking_seat()

    async def mission_async(self, state: Any, seat: int) -> MissionPayload:
        # Track that this seat is currently thinking (waiting for API call)
        self._session.set_thinking_seat(seat)
        try:
            return await super().mission_async(state, seat)
        finally:
            # Clear the thinking seat after the API call completes
            self._session.clear_thinking_seat()

    async def assassin_guess_async(self, state: Any, seat: int) -> AssassinGuessPayload:
        # Track that this seat is currently thinking (waiting for API call)
        self._session.set_thinking_seat(seat)
        try:
            return await super().assassin_guess_async(state, seat)
        finally:
            # Clear the thinking seat after the API call completes
            self._session.clear_thinking_seat()


class WebHumanSeatAgent(BaseSeatAgent):
    """Seat agent that coordinates with :class:`GameSession` via HTTP."""

    def __init__(self, seat: int, *, session: "GameSession", recorder: SimulationRecorder) -> None:
        super().__init__(seat)
        self._session = session
        self._recorder = recorder
        self.records_vote = True  # Record own votes like AI agents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _await_payload_async(self, state: Any, phase: Phase) -> Dict[str, Any]:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        options = self._session.build_options(state=state, phase=phase, seat=self.seat)
        instructions = self._session.build_instructions(phase=phase, seat=self.seat)
        await self._session.update_state_async(state)
        return await self._session.wait_for_action_async(
            seat=self.seat,
            phase=phase_name,
            options=options,
            instructions=instructions,
        )

    def _validate_payload(self, phase: Phase, payload: Dict[str, Any]) -> Any:
        schema_phase = SchemaPhase(phase.value if hasattr(phase, "value") else str(phase))
        return validate_payload(phase=schema_phase, payload=payload)

    # ------------------------------------------------------------------
    # Phase handlers mirroring :class:`HumanSeatAgent`
    # ------------------------------------------------------------------

    async def proposal_async(self, state: Any, stage: Phase) -> ProposalPayload:
        payload_model = self._validate_payload(stage, await self._await_payload_async(state, stage))
        self._recorder.record_proposal(
            round_num=getattr(state, "round_num", 0),
            phase=stage.value if hasattr(stage, "value") else str(stage),
            leader=self.seat,
            payload=payload_model.model_dump(),
            usage={},
            reasoning=None,
        )
        return payload_model

    async def proposal_summary_async(self, state: Any, seat: int) -> DiscussionPayload:
        payload_model = self._validate_payload(Phase.PROPOSAL_SUMMARY, await self._await_payload_async(state, Phase.PROPOSAL_SUMMARY))
        self._recorder.record_summary(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            payload=payload_model.model_dump(),
            reasoning=None,
        )
        return payload_model

    async def discussion_async(self, state: Any, seat: int) -> DiscussionPayload:
        payload_model = self._validate_payload(Phase.DISCUSSION, await self._await_payload_async(state, Phase.DISCUSSION))
        self._recorder.record_discussion(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            payload=payload_model.model_dump(),
            reasoning=None,
        )
        return payload_model

    async def vote_async(self, state: Any, seat: int) -> VotePayload:
        payload_model = self._validate_payload(Phase.VOTE, await self._await_payload_async(state, Phase.VOTE))

        # Record vote like AI agents do
        vote_value = payload_model.action.value
        value_str = vote_value.value if hasattr(vote_value, 'value') else str(vote_value)
        self._recorder.record_vote(
            round_num=state.round_num,
            seat=seat,
            value=value_str,
            reasoning=None,  # No reasoning for human players
        )

        return payload_model

    async def mission_async(self, state: Any, seat: int) -> MissionPayload:
        payload_model = self._validate_payload(Phase.MISSION, await self._await_payload_async(state, Phase.MISSION))
        # Note: FSM handles mission recording - don't duplicate record here
        return payload_model

    async def evil_conclave_async(self, state: Any, seat: int) -> EvilConclavePayload:
        payload_model = self._validate_payload(Phase.EVIL_CONCLAVE, await self._await_payload_async(state, Phase.EVIL_CONCLAVE))
        payload_dict = payload_model.model_dump()
        expected = len(state.evil_seats()) if hasattr(state, "evil_seats") else 0
        self._recorder.record_conclave(
            round_num=getattr(state, "round_num", 0),
            seat=seat,
            speech=payload_dict.get("speech", ""),
            thinking=payload_dict.get("thinking"),
            expected=expected,
            reasoning=None,
        )
        return payload_model

    async def assassin_guess_async(self, state: Any, seat: int) -> AssassinGuessPayload:
        payload_model = self._validate_payload(Phase.ASSASSIN_GUESS, await self._await_payload_async(state, Phase.ASSASSIN_GUESS))
        payload_dict = payload_model.model_dump()
        target = payload_dict.get("action", {}).get("target", -1)
        correct = False
        if hasattr(state, "merlin_seat"):
            try:
                correct = target == state.merlin_seat()
            except Exception:  # pragma: no cover - defensive safeguard
                correct = False
        self._recorder.record_assassin_guess(
            seat=seat,
            target=target,
            correct=correct,
            reasoning=None,
        )
        return payload_model

    # ------------------------------------------------------------------
    # Synchronous interface is not supported
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name in {"proposal", "proposal_summary", "discussion", "vote", "mission", "evil_conclave", "assassin_guess"}:
            raise AttributeError(
                "WebHumanSeatAgent must be used via its async methods"
            )
        raise AttributeError(name)


class GameSession:
    """Coordinates a running game and exposes state for a web frontend."""

    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
        max_rounds: Optional[int] = None,
        human_seat: Optional[int] = None,
        seat_models: Optional[Dict[str, str]] = None,
        seat_providers: Optional[Dict[str, str]] = None,
        builder: Optional[ContextBuilder] = None,
        client: Any = None,
        recorder: Optional[SimulationRecorder] = None,
        agent_overrides: Optional[Dict[int, BaseSeatAgent]] = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        self.seed = seed
        self.max_rounds = max_rounds
        self.human_seat = human_seat
        self.seat_models = seat_models
        self.seat_providers = seat_providers
        self.builder = builder or ContextBuilder(config=load_context_config())
        self.client = client

        # Determine total players from seat configuration
        # Note: seat_models only contains AI seats, so we need to find the max seat number
        # and include the human seat if present
        total_players = 5  # default
        if seat_models:
            max_ai_seat = max(int(seat) for seat in seat_models.keys())
            total_players = max_ai_seat
        if human_seat is not None:
            total_players = max(total_players, human_seat)
        self.recorder = recorder or SimulationRecorder(total_seats=total_players)
        self._agent_overrides = agent_overrides or {}

        self._state: Optional[Any] = None
        self._pending: Optional[PendingRequest] = None
        self._pending_response: Optional[Dict[str, Any]] = None
        self._pending_id: Optional[str] = None
        self._thinking_seat: Optional[int] = None  # Track which agent is currently waiting for API response
        self._completed = False
        self._error: Optional[str] = None
        self._transcript_paths: Optional[TranscriptPaths] = None

        self._condition = asyncio.Condition()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task[Any]] = None

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background FSM coroutine."""

        if self._task and not self._task.done():  # pragma: no cover - defensive
            return

        services = self._make_services()
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run_game(services))

    async def _run_game(self, services: PhaseServices) -> None:
        try:
            # Wait 1 second to allow avatar images to load from DiceBear
            await asyncio.sleep(1.0)

            # Determine player count from configured seats
            # Note: seat_models only contains AI seats, so we need to find the max seat number
            # and include the human seat if present
            players = 5  # default
            if self.builder.config.seat_models:
                max_ai_seat = max(self.builder.config.seat_models.keys())
                players = max_ai_seat
            if self.human_seat is not None:
                players = max(players, self.human_seat)
            state = await run_game_async(seed=self.seed, services=services, max_rounds=self.max_rounds, players=players)
            await self.update_state_async(state)
            transcript_paths = self._build_transcript_paths(state)
            async with self._condition:
                self._completed = True
                if transcript_paths is not None:
                    self._transcript_paths = transcript_paths
                self._condition.notify_all()
        except Exception as exc:  # pragma: no cover - propagate error for visibility
            async with self._condition:
                self._error = str(exc)
                self._completed = True
                self._condition.notify_all()

    def _make_services(self) -> PhaseServices:
        from ..services.cli import make_phase_services
        from ..providers.openrouter import OpenRouterClient

        # If custom seat configuration provided, override builder config
        if self.seat_models or self.seat_providers:
            # Convert string keys to int keys for seat_models and seat_providers
            seat_models_int = {int(k): v for k, v in (self.seat_models or {}).items()}
            seat_providers_int = {int(k): v for k, v in (self.seat_providers or {}).items()}

            # Update the builder's config with dynamic seat configuration
            self.builder.config.seat_models = seat_models_int
            if seat_providers_int:
                self.builder.config.seat_providers = seat_providers_int

        client = self.client
        if client is None:
            try:
                from ..providers.providers import ProviderFactory
                # Check if we have seat_providers configuration for multi-provider support
                if self.builder.config.seat_providers:
                    client = ProviderFactory.create(
                        "multi",
                        seat_providers=self.builder.config.seat_providers,
                        default_provider=self.builder.config.default_provider
                    )
                else:
                    # Use the default provider for all seats
                    client = ProviderFactory.create(self.builder.config.default_provider)
            except ProviderError:
                client = LocalDeterministicClient()
        self.client = client

        overrides = dict(self._agent_overrides)

        # Determine total players from seat configuration
        # Note: seat_models only contains AI seats, so we need to find the max seat number
        # and include the human seat if present
        total_players = 5  # default
        if self.builder.config.seat_models:
            max_ai_seat = max(self.builder.config.seat_models.keys())
            total_players = max_ai_seat
        if self.human_seat is not None:
            total_players = max(total_players, self.human_seat)

        for seat in range(1, total_players + 1):
            if seat == self.human_seat:
                overrides[seat] = WebHumanSeatAgent(
                    seat,
                    session=self,
                    recorder=self.recorder,
                )
            else:
                overrides.setdefault(
                    seat,
                    SessionModelSeatAgent(
                        seat,
                        session=self,
                        client=client,
                        builder=self.builder,
                        recorder=self.recorder,
                        seed=self.seed,
                    ),
                )

        # Initialize analysis writer
        from ..core.fsm import GameState
        from ..utils.rng import build_rng
        initial_state = GameState(seed=self.seed, rng=build_rng(seed=self.seed))
        self.recorder.init_analysis_writer(
            initial_state,
            seed=str(self.seed) if self.seed else None,
            seat_models=self.builder.config.seat_models
        )

        services = make_phase_services(
            client=client,
            builder=self.builder,
            recorder=self.recorder,
            seed=self.seed,
            human_service=None,
            agent_overrides=overrides,
        )
        return services

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _build_transcript_paths(self, state: Any) -> Optional[TranscriptPaths]:
        try:
            config = getattr(self.builder, "config", None)
            seat_models = getattr(config, "seat_models", {}) if config else {}
            default_model = getattr(config, "default_model", "qwen2.5-instruct") if config else "qwen2.5-instruct"
            self.recorder.ensure_mission_events(getattr(state, "missions", []) or [])
            return self.recorder.build_transcript(
                state=state,
                seat_models=seat_models,
                seed=self.seed,
                default_model=default_model,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("transcript.generate_failed", error=str(exc))
            return None

    def _transcript_metadata(self) -> Optional[Dict[str, Dict[str, str]]]:
        paths = self._transcript_paths
        if paths is None:
            return None
        data: Dict[str, Dict[str, str]] = {
            "json": {
                "path": str(paths.json_path),
                "filename": paths.json_path.name,
            },
            "markdown": {
                "path": str(paths.markdown_path),
                "filename": paths.markdown_path.name,
            },
        }
        if paths.context_markdown_path is not None:
            data["context"] = {
                "path": str(paths.context_markdown_path),
                "filename": paths.context_markdown_path.name,
            }
        return data

    async def transcript_metadata_async(self) -> Optional[Dict[str, Dict[str, str]]]:
        async with self._condition:
            return self._transcript_metadata()

    def transcript_metadata(self) -> Optional[Dict[str, Dict[str, str]]]:
        return self._transcript_metadata()

    async def update_state_async(self, state: Any) -> None:
        async with self._condition:
            self._state = state
            self._condition.notify_all()

    def update_state(self, state: Any) -> None:
        loop = self._loop
        if loop and loop.is_running():
            try:
                # Check if we're already in the event loop
                current_loop = asyncio.get_running_loop()
                if current_loop is loop:
                    # We're in the same loop, schedule the coroutine
                    asyncio.create_task(self.update_state_async(state))
                else:
                    # We're in a different loop, use run_coroutine_threadsafe
                    asyncio.run_coroutine_threadsafe(self.update_state_async(state), loop)
            except RuntimeError:
                # No running loop, use run_coroutine_threadsafe
                asyncio.run_coroutine_threadsafe(self.update_state_async(state), loop)
        else:  # pragma: no cover - fallback for tests before loop starts
            self._state = state

    def _serialize_state(self, state: Any, perspective: Optional[int]) -> Dict[str, Any]:
        current_proposal = getattr(state, "current_proposal", None)
        proposals = [
            {
                "round": proposal.round_num,
                "leader": proposal.leader,
                "members": list(proposal.members),
                "approved": proposal.approved,
            }
            for proposal in getattr(state, "proposals", [])
        ]
        missions = [
            {
                "round": mission.round_num,
                "members": list(mission.members),
                "fails": mission.fails,
                "result": mission.result.value if hasattr(mission.result, "value") else mission.result,
            }
            for mission in getattr(state, "missions", [])
        ]
        votes = []
        for vote_data in getattr(state, "votes", []):
            # Handle both old format (round, votes) and new format (round, votes, leader)
            if len(vote_data) == 2:
                round_num, vote_list = vote_data
                leader = None  # No leader info in old format
            elif len(vote_data) == 3:
                round_num, vote_list, leader = vote_data
            else:
                continue  # Skip malformed vote data

            votes.append({
                "round": round_num,
                "leader": leader,
                "votes": [
                    {
                        "seat": vote.seat,
                        "value": vote.value.value if hasattr(vote.value, "value") else vote.value,
                    }
                    for vote in vote_list
                ],
            })

        speeches: List[Dict[str, Any]] = []
        raw_speeches = getattr(state, "speeches", {}) or {}
        for proposal_id in sorted(raw_speeches.keys()):
            proposal_entries = raw_speeches.get(proposal_id, {})
            for seat_id, text in proposal_entries.items():
                # Extract round number from proposal ID (e.g., "P2-s3" -> round 2)
                try:
                    round_num = int(proposal_id.split('-')[0][1:])  # Remove 'P' and get number
                except (ValueError, IndexError):
                    round_num = 1  # fallback
                
                speeches.append(
                    {
                        "round": round_num,
                        "seat": seat_id,
                        "speech": text,
                        "kind": "DISCUSSION",
                        "isYou": perspective == seat_id if perspective is not None else False,
                    }
                )

        raw_summaries = getattr(state, "summaries", {}) or {}
        for round_id in sorted(raw_summaries.keys()):
            round_entries = raw_summaries.get(round_id, {})
            for seat_id, text in round_entries.items():
                speeches.append(
                    {
                        "round": round_id,
                        "seat": seat_id,
                        "speech": text,
                        "kind": "SUMMARY",
                        "isYou": perspective == seat_id if perspective is not None else False,
                    }
                )

        players: List[Dict[str, Any]] = []
        all_roles = getattr(state, "roles", {}) or {}

        def _to_role(value: Any) -> Optional[Role]:
            if isinstance(value, Role):
                return value
            if value is None:
                return None
            try:
                return Role(value)
            except Exception:  # pragma: no cover - defensive conversion
                return None

        def _role_value(value: Any) -> Optional[str]:
            if value is None:
                return None
            if hasattr(value, "value"):
                return str(value.value)
            return str(value)

        visible_map: Dict[int, Dict[str, Optional[str]]] = {}
        if perspective is None or self._completed:
            for seat, role in all_roles.items():
                visible_map[seat] = {
                    "role": _role_value(role),
                    "role_name": ROLE_NAMES_ZH.get(_to_role(role), _role_value(role) or ""),
                }
        else:
            perspective_role = _to_role(all_roles.get(perspective))
            if perspective_role is not None:
                if perspective_role == Role.MERLIN:
                    ambiguous_evil_label = "未知坏人"
                    for seat, role in all_roles.items():
                        concrete = _to_role(role)
                        if concrete in {Role.ASSASSIN, Role.MORGANA, Role.OBERON}:
                            visible_map[seat] = {
                                "role": None,
                                "role_name": ambiguous_evil_label,
                            }
                elif perspective_role in {Role.ASSASSIN, Role.MORGANA}:
                    for seat, role in all_roles.items():
                        concrete = _to_role(role)
                        # Assassin and Morgana see each other but not Oberon
                        if concrete in {Role.ASSASSIN, Role.MORGANA}:
                            visible_map[seat] = {
                                "role": _role_value(role),
                                "role_name": ROLE_NAMES_ZH.get(concrete, _role_value(role) or ""),
                            }
                elif perspective_role == Role.OBERON:
                    # Oberon doesn't see any other evil players, only knows his own role
                    pass
                elif perspective_role == Role.PERCIVAL:
                    ambiguous_label = f"{ROLE_NAMES_ZH.get(Role.MERLIN, 'Merlin')} / {ROLE_NAMES_ZH.get(Role.MORGANA, 'Morgana')}"
                    for seat, role in all_roles.items():
                        concrete = _to_role(role)
                        if concrete in {Role.MERLIN, Role.MORGANA}:
                            visible_map[seat] = {
                                "role": None,
                                "role_name": ambiguous_label,
                            }

            # Always ensure the human sees their own role
            if perspective_role is not None:
                own_role = all_roles.get(perspective, perspective_role)
                visible_map[perspective] = {
                    "role": _role_value(own_role),
                    "role_name": ROLE_NAMES_ZH.get(perspective_role, _role_value(own_role) or ""),
                }

        # Get total players from state
        total_players = getattr(state, "ruleset", None)
        if total_players:
            total_players = total_players.players
        else:
            total_players = len(all_roles) if all_roles else 5

        for seat in range(1, total_players + 1):
            role = all_roles.get(seat)
            visible = visible_map.get(seat)
            if visible is None and (perspective is None or self._completed or seat == perspective):
                visible = {
                    "role": _role_value(role),
                    "role_name": ROLE_NAMES_ZH.get(_to_role(role), _role_value(role) or ""),
                }

            # Get model and provider info for this seat
            seat_str = str(seat)
            model_name = self.seat_models.get(seat_str) if self.seat_models else None
            provider_name = self.seat_providers.get(seat_str) if self.seat_providers else None

            # If this is the human seat, mark it as "Human"
            if seat == self.human_seat:
                model_name = "Human"
                provider_name = None

            players.append(
                {
                    "seat": seat,
                    "role": visible.get("role") if visible else None,
                    "role_name": visible.get("role_name") if visible else None,
                    "model": model_name,
                    "provider": provider_name,
                }
            )

        summary: Dict[str, Any] = {
            "round": getattr(state, "round_num", 0),
            "phase": getattr(getattr(state, "state", None), "value", str(getattr(state, "state", ""))),
            "leader": getattr(state, "leader", 1),
            "teamSize": state.current_team_size(),
            "scores": {"good": getattr(state, "good_missions", 0), "evil": getattr(state, "evil_missions", 0)},
            "failedProposals": getattr(state, "failed_proposals", 0),
            "currentProposal": {
                "leader": getattr(current_proposal, "leader", None),
                "members": list(getattr(current_proposal, "members", []) or []),
                "approved": getattr(current_proposal, "approved", None),
            },
            "proposals": proposals,
            "missions": missions,
            "votes": votes,
            "speeches": speeches,
            "players": players,
            "winner": getattr(getattr(state, "winner", None), "value", None),
        }

        if perspective is not None:
            summary["privateCard"] = default_role_card_provider(state, perspective)

        return summary

    def serialize_state(self, perspective: Optional[int] = None) -> Dict[str, Any]:
        state = self._state
        if state is None:
            return {}
        return self._serialize_state(state, perspective)

    async def serialize_state_async(self, perspective: Optional[int] = None) -> Dict[str, Any]:
        async with self._condition:
            state = self._state
        if state is None:
            return {}
        return self._serialize_state(state, perspective)

    # ------------------------------------------------------------------
    # Human request coordination
    # ------------------------------------------------------------------

    async def wait_for_action_async(
        self,
        *,
        seat: int,
        phase: str,
        options: Dict[str, Any],
        instructions: str,
    ) -> Dict[str, Any]:
        request_id = uuid.uuid4().hex
        snapshot = await self.serialize_state_async(perspective=seat)
        pending = PendingRequest(
            request_id=request_id,
            seat=seat,
            phase=phase,
            options=options,
            instructions=instructions,
            state_snapshot=snapshot,
        )

        async with self._condition:
            self._pending = pending
            self._pending_id = request_id
            self._condition.notify_all()
            while True:
                await self._condition.wait()
                if self._pending_response is not None and self._pending_id == request_id:
                    payload = self._pending_response
                    self._pending = None
                    self._pending_response = None
                    self._pending_id = None
                    return payload

    async def submit_action_async(self, request_id: str, payload: Dict[str, Any]) -> None:
        async with self._condition:
            if not self._pending or self._pending.request_id != request_id:
                raise ValueError("No pending request matches the provided identifier")
            self._pending_response = payload
            self._pending_id = request_id
            self._condition.notify_all()

    def wait_for_action(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - legacy sync path
        loop = self._loop
        if not loop or not loop.is_running():
            raise RuntimeError("Session event loop is not running")
        future = asyncio.run_coroutine_threadsafe(self.wait_for_action_async(**kwargs), loop)
        return future.result()

    def submit_action(self, request_id: str, payload: Dict[str, Any]) -> None:
        loop = self._loop
        if not loop or not loop.is_running():
            raise RuntimeError("Session event loop is not running")
        future = asyncio.run_coroutine_threadsafe(
            self.submit_action_async(request_id, payload), loop
        )
        future.result()

    async def current_pending_async(self) -> Optional[PendingRequest]:
        async with self._condition:
            return self._pending

    def current_pending(self) -> Optional[PendingRequest]:
        loop = self._loop
        if loop and loop.is_running():
            return asyncio.run_coroutine_threadsafe(self.current_pending_async(), loop).result()
        return self._pending

    def set_thinking_seat(self, seat: int) -> None:
        """Set the seat that is currently waiting for an API response."""
        loop = self._loop
        if loop and loop.is_running():
            # Schedule on the event loop if it's running
            asyncio.run_coroutine_threadsafe(self._set_thinking_seat_async(seat), loop)
        else:
            # Directly set if no event loop (e.g., for testing)
            self._thinking_seat = seat

    async def _set_thinking_seat_async(self, seat: int) -> None:
        """Asynchronously set the thinking seat."""
        async with self._condition:
            self._thinking_seat = seat
            self._condition.notify_all()

    def clear_thinking_seat(self) -> None:
        """Clear the current thinking seat."""
        loop = self._loop
        if loop and loop.is_running():
            # Schedule on the event loop if it's running
            asyncio.run_coroutine_threadsafe(self._clear_thinking_seat_async(), loop)
        else:
            # Directly clear if no event loop (e.g., for testing)
            self._thinking_seat = None

    async def _clear_thinking_seat_async(self) -> None:
        """Asynchronously clear the thinking seat."""
        async with self._condition:
            self._thinking_seat = None
            self._condition.notify_all()

    async def status_async(self) -> Dict[str, Any]:
        async with self._condition:
            pending = self._pending
            completed = self._completed
            error = self._error
            state = self._state
            transcript = self._transcript_metadata()
        serialized = self._serialize_state(state, self.human_seat) if state is not None else {}
        return {
            "sessionId": self.session_id,
            "completed": completed,
            "error": error,
            "state": serialized,
            "pending": pending,
            "thinkingSeat": self._thinking_seat,  # Add the thinking seat info
            "transcript": transcript,
        }

    def status(self) -> Dict[str, Any]:
        loop = self._loop
        if loop and loop.is_running():
            return asyncio.run_coroutine_threadsafe(self.status_async(), loop).result()
        return {
            "sessionId": self.session_id,
            "completed": self._completed,
            "error": self._error,
            "state": self.serialize_state(perspective=self.human_seat),
            "pending": self._pending,
            "thinkingSeat": self._thinking_seat,  # Add the thinking seat info
            "transcript": self._transcript_metadata(),
        }

    async def stop_async(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover - expected during shutdown
                pass
        async with self._condition:
            self._completed = True
            self._condition.notify_all()
        self._task = None

    def stop(self) -> None:
        loop = self._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.stop_async(), loop).result()
        else:
            self._task = None

    # ------------------------------------------------------------------
    # Frontend utilities
    # ------------------------------------------------------------------

    def build_options(self, *, state: Any, phase: Phase, seat: int) -> Dict[str, Any]:
        team_size = state.current_team_size()
        current_members = list(getattr(getattr(state, "current_proposal", None), "members", []) or [])
        mission_members = list(getattr(getattr(state, "current_mission", None), "members", []) or current_members)
        role = getattr(state, "roles", {}).get(seat)
        on_mission = seat in mission_members
        # Get total players from state
        total_players = getattr(state, "ruleset", None)
        if total_players:
            total_players = total_players.players
        else:
            total_players = 5

        return {
            "phase": phase.value if hasattr(phase, "value") else str(phase),
            "teamSize": team_size,
            "availableSeats": list(range(1, total_players + 1)),
            "currentMembers": current_members,
            "missionMembers": mission_members,
            "onMission": on_mission,
            "canFailMission": bool(role and is_evil_player(role)),
        }

    def build_instructions(self, *, phase: Phase, seat: int) -> str:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        base = {
            Phase.PROPOSAL_DRAFT.value: "选择团队成员并提交提案。",
            Phase.PROPOSAL_FINAL.value: "最终确定任务阵容并提交。",
            Phase.PROPOSAL_SUMMARY.value: "撰写简短总结并公开说明提案理由。",
            Phase.DISCUSSION.value: "输入你的思考（thinking）与公开发言（speech）。",
            Phase.VOTE.value: "选择是否赞成当前提案，并写下不公开的反思。",
            Phase.MISSION.value: "如果你在任务队伍中，请决定任务成功或失败。",
            Phase.EVIL_CONCLAVE.value: "和其他坏人密谈，给出你的思考与发言。",
            Phase.ASSASSIN_GUESS.value: "如果你是刺客，请选择你认为是梅林的座位。",
        }
        return base.get(phase_name, f"完成 {phase_name} 阶段的动作。")


class SessionManager:
    """Registry for multiple concurrent :class:`GameSession` instances."""

    def __init__(self) -> None:
        self._sessions: Dict[str, GameSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, **kwargs: Any) -> GameSession:
        session = GameSession(**kwargs)
        async with self._lock:
            self._sessions[session.session_id] = session
        await session.start()
        return session

    async def get(self, session_id: str) -> GameSession:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session {session_id}")
            return self._sessions[session_id]

    async def list_sessions(self) -> List[GameSession]:
        async with self._lock:
            return list(self._sessions.values())

    async def stop(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise KeyError(f"Unknown session {session_id}")
            del self._sessions[session_id]
        await session.stop_async()


SESSION_MANAGER = SessionManager()
"""Default registry used by the FastAPI app."""


TEAM_PATTERN = re.compile(r"team\s*[:：]\s*(\d)")


class LocalDeterministicClient:
    """Fallback transport that produces deterministic payloads without API access."""

    def __init__(self) -> None:
        self._call_count = 0

    @staticmethod
    def _extract_team_size(request: Any) -> int:
        messages = getattr(request, "messages", []) or []
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                match = TEAM_PATTERN.search(content)
                if match:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        continue
        return 2

    def call_phase(self, request: Any) -> ProviderResponse[Any]:
        phase = str(getattr(request, "phase", "")).upper()
        seat = int(getattr(request, "seat", 1) or 1)
        self._call_count += 1

        if phase in {Phase.PROPOSAL_DRAFT.value, Phase.PROPOSAL_FINAL.value}:
            team_size = self._extract_team_size(request)
            # Get total players from state
            state = self._state
            total_players = getattr(state, "ruleset", None)
            if total_players:
                total_players = total_players.players
            else:
                total_players = 5
            members = [((seat + idx - 1) % total_players) + 1 for idx in range(team_size)]
            payload = ProposalPayload(action=ProposeAction(members=sorted(dict.fromkeys(members))[:team_size]))
        elif phase == Phase.DISCUSSION.value or phase == Phase.PROPOSAL_SUMMARY.value:
            payload = DiscussionPayload(
                thinking="保持冷静分析局势。",
                speech=f"座位{seat} 认为当前提案值得讨论。",
            )
        elif phase == Phase.VOTE.value:
            payload = VotePayload(
                action=VoteAction(
                    value=VoteValue.APPROVE if (self._call_count + seat) % 2 == 0 else VoteValue.REJECT,
                    reflection="观察队伍历史，做出谨慎判断。",
                )
            )
        elif phase == Phase.MISSION.value:
            value = MissionOutcome.FAIL if (seat % 2 == 0 and self._call_count % 3 == 0) else MissionOutcome.SUCCESS
            payload = MissionPayload(action=MissionAction(value=value))
        elif phase == Phase.EVIL_CONCLAVE.value:
            payload = EvilConclavePayload(
                thinking="与队友协调下一步行动。",
                speech="保持隐蔽，等待好机会出手。",
            )
        elif phase == Phase.ASSASSIN_GUESS.value:
            # Get total players from state
            state = self._state
            total_players = getattr(state, "ruleset", None)
            if total_players:
                total_players = total_players.players
            else:
                total_players = 5
            target = ((seat + 1) % total_players) + 1
            payload = AssassinGuessPayload(action=AssassinGuessAction(target=target))
        else:  # pragma: no cover - defensive default
            payload = DiscussionPayload(thinking="", speech="")

        return ProviderResponse(
            payload=payload,
            raw_response={},
            usage={},
            retries=0,
            reasoning=None,
            violations=None,
        )

    @staticmethod
    def _extract_reasoning_info(_raw: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - compatibility shim
        return {}
