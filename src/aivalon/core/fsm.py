"""Finite state machine orchestrating Avalon game flow."""

from __future__ import annotations

import random
import asyncio
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

from pydantic import BaseModel

from ..providers.providers import ProviderError
from ..utils.rng import build_rng, shuffle_seats
from .rulesets import AvalonRuleset, get_ruleset
from .schemas import (
    AssassinGuessPayload,
    DiscussionPayload,
    EvilConclavePayload,
    MissionOutcome,
    MissionPayload,
    ProposalPayload,
    SchemaValidationError,
    VotePayload,
    VoteValue,
    create_default_assassin_guess_payload,
    create_default_discussion_payload,
    create_default_evil_conclave_payload,
    create_default_mission_payload,
    create_default_proposal_payload,
    create_default_vote_payload,
    validate_payload,
)

TOTAL_SEATS = 5
ALL_SEATS: Tuple[int, ...] = tuple(range(1, TOTAL_SEATS + 1))

T = TypeVar("T", bound=BaseModel)


class State(str, Enum):
    """Game states."""

    SETUP = "SETUP"
    PROPOSAL_DRAFT = "PROPOSAL_DRAFT"
    DISCUSSION = "DISCUSSION"
    PROPOSAL_FINAL = "PROPOSAL_FINAL"
    VOTE = "VOTE"
    MISSION = "MISSION"
    ROUND_END = "ROUND_END"
    EVIL_CONCLAVE = "EVIL_CONCLAVE"
    ASSASSIN_GUESS = "ASSASSIN_GUESS"
    END_GAME = "END_GAME"


class Phase(str, Enum):
    """Phase enum used when requesting model outputs."""

    PROPOSAL_DRAFT = "PROPOSAL_DRAFT"
    PROPOSAL_SUMMARY = "PROPOSAL_SUMMARY"
    DISCUSSION = "DISCUSSION"
    PROPOSAL_FINAL = "PROPOSAL_FINAL"
    VOTE = "VOTE"
    MISSION = "MISSION"
    ROUND_END = "ROUND_END"
    EVIL_CONCLAVE = "EVIL_CONCLAVE"
    ASSASSIN_GUESS = "ASSASSIN_GUESS"


class Outcome(str, Enum):
    """Game outcomes."""

    GOOD_WIN = "Good"
    EVIL_WIN = "Evil"


class Role(str, Enum):
    """Player roles."""

    MERLIN = "Merlin"
    PERCIVAL = "Percival"
    ASSASSIN = "Assassin"
    MORGANA = "Morgana"
    OBERON = "Oberon"
    LOYAL_SERVANT = "Loyal Servant"


MISSION_TEAM_SIZES: Tuple[int, ...] = (2, 3, 2, 3, 3)
GOOD_WIN_THRESHOLD = 3
EVIL_WIN_THRESHOLD = 3
MAX_FAILED_PROPOSALS = 3


def canonical_team(members: Iterable[int]) -> Tuple[int, ...]:
    """Return a sorted, duplicate-free tuple of seat numbers."""

    try:
        normalized = {int(seat) for seat in members}
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Team members must be integers") from exc
    if not normalized:
        return tuple()
    return tuple(sorted(normalized))



@dataclass
class Proposal:
    """A proposed team."""

    leader: int
    members: Tuple[int, ...]
    round_num: int
    approved: Optional[bool] = None
    proposal_id: str = ""


@dataclass
class Vote:
    """A vote on a proposal."""

    seat: int
    value: VoteValue


@dataclass
class Mission:
    """A mission execution record."""

    members: Tuple[int, ...]
    round_num: int
    fails: int = 0
    result: Optional[MissionOutcome] = None


@dataclass
class EvilConclaveMessage:
    """A single evil conclave message."""

    round_num: int
    seat: int
    speech: str
    thinking: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None


@dataclass
class AssassinGuess:
    """Assassin's final guess outcome."""

    seat: int
    target: int
    correct: bool
    reasoning: Optional[Dict[str, Any]] = None


PhaseCallable = Callable[..., BaseModel | dict | Awaitable[BaseModel | dict]]


@dataclass
class PhaseServices:
    """Callables that produce phase payloads for the FSM."""

    proposal: Optional[PhaseCallable] = None
    proposal_summary: Optional[PhaseCallable] = None
    discussion: Optional[PhaseCallable] = None
    vote: Optional[PhaseCallable] = None
    mission: Optional[PhaseCallable] = None
    evil_conclave: Optional[PhaseCallable] = None
    assassin_guess: Optional[PhaseCallable] = None
    agents: Dict[int, Any] = field(default_factory=dict)
    _human_service: Optional[Any] = None  # Reference to human player service for async operations
    _recorder: Optional[Any] = None  # Reference to recorder for human vote recording


@dataclass
class GameState:
    """Container for tracking game state."""

    seed: Optional[int]
    rng: random.Random
    ruleset: Optional[AvalonRuleset] = None
    players: int = 5  # Default to 5 players for backward compatibility

    state: State = State.SETUP
    round_num: int = 1
    leader: int = 1
    failed_proposals: int = 0

    good_missions: int = 0
    evil_missions: int = 0

    proposals: List[Proposal] = field(default_factory=list)
    votes: List[Tuple[int, List[Vote]]] = field(default_factory=list)
    stance_history: Dict[int, Dict[Tuple[int, ...], Dict[int, VoteValue]]] = field(default_factory=dict)
    missions: List[Mission] = field(default_factory=list)
    speeches: Dict[str, Dict[int, str]] = field(default_factory=dict)
    summaries: Dict[int, Dict[int, str]] = field(default_factory=dict)
    reflections: Dict[Tuple[int, int], str] = field(default_factory=dict)  # (round, seat) -> reflection
    evil_conclave_messages: List[EvilConclaveMessage] = field(default_factory=list)
    assassin_guess: Optional[AssassinGuess] = None
    winner: Optional[Outcome] = None

    round_limit: Optional[int] = None
    truncated: bool = False

    roles: Dict[int, Role] = field(default_factory=dict)
    leader_order: List[int] = field(default_factory=list)
    current_proposal: Optional[Proposal] = None
    current_mission: Optional[Mission] = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = build_rng(seed=self.seed)
        if self.ruleset is None:
            self.ruleset = get_ruleset(self.players)
        if not self.leader_order:
            self.leader_order.append(self.leader)

    def current_team_size(self) -> int:
        return self.ruleset.team_size_for_round(self.round_num)

    def is_good_win(self) -> bool:
        return self.good_missions >= self.ruleset.good_win_threshold

    def is_evil_win(self) -> bool:
        return self.evil_missions >= self.ruleset.evil_win_threshold

    def next_leader(self) -> int:
        return (self.leader % self.ruleset.players) + 1

    def advance_leader(self) -> None:
        self.leader = self.next_leader()
        self.leader_order.append(self.leader)

    def record_proposal(self, proposal: Proposal) -> None:
        proposal.proposal_id = f"P{proposal.round_num}-s{proposal.leader}"
        self.current_proposal = proposal
        self.proposals.append(proposal)

    def record_vote_round(self, votes: List[Vote]) -> None:
        expected_seats = set(self.ruleset.all_seats)
        if len(votes) != len(expected_seats):
            raise ValueError(
                f"Vote round must include exactly one vote per seat; expected {len(expected_seats)} got {len(votes)}"
            )

        seen: set[int] = set()
        for vote in votes:
            if vote.seat not in expected_seats:
                raise ValueError(f"Vote recorded for invalid seat {vote.seat}")
            if vote.seat in seen:
                raise ValueError(f"Duplicate vote detected for seat {vote.seat}")
            seen.add(vote.seat)
            if vote.value not in {VoteValue.APPROVE, VoteValue.REJECT}:
                raise ValueError(f"Unexpected vote value {vote.value} for seat {vote.seat}")

        canonical_votes = sorted(votes, key=lambda item: item.seat)
        # Include leader info to distinguish between multiple proposals in same round
        leader = self.current_proposal.leader if self.current_proposal else self.leader
        self.votes.append((self.round_num, canonical_votes, leader))

        if self.current_proposal is not None:
            round_history = self.stance_history.setdefault(self.round_num, {})
            team_history = round_history.setdefault(self.current_proposal.members, {})
            for vote in canonical_votes:
                team_history[vote.seat] = vote.value

    def record_mission(self, mission: Mission) -> None:
        self.current_mission = mission
        self.missions.append(mission)
        if mission.result == MissionOutcome.SUCCESS:
            self.good_missions += 1
        else:
            self.evil_missions += 1

        # Validate counter accuracy after each mission recording
        self._validate_mission_counters()

    def _validate_mission_counters(self) -> None:
        """Validate that mission counters match the actual mission list."""
        expected_good = sum(1 for m in self.missions if m.result == MissionOutcome.SUCCESS)
        expected_evil = sum(1 for m in self.missions if m.result == MissionOutcome.FAIL)

        if self.good_missions != expected_good:
            raise ValueError(
                f"Good missions counter mismatch: counter={self.good_missions}, "
                f"actual={expected_good}, missions={len(self.missions)}"
            )

        if self.evil_missions != expected_evil:
            raise ValueError(
                f"Evil missions counter mismatch: counter={self.evil_missions}, "
                f"actual={expected_evil}, missions={len(self.missions)}"
            )

    def record_speech(self, seat: int, speech: str, *, proposal_id: Optional[str] = None) -> None:
        # Use current proposal ID if not provided
        if proposal_id is None:
            if self.current_proposal and self.current_proposal.proposal_id:
                proposal_id = self.current_proposal.proposal_id
            else:
                # Fallback: generate proposal ID from current state
                proposal_id = f"P{self.round_num}-s{self.leader}"
        
        proposal_map = self.speeches.setdefault(proposal_id, {})
        proposal_map[seat] = speech[:300]

        # Keep only recent proposals (last 3 proposals)
        tracked_proposals = sorted(self.speeches.keys())
        if len(tracked_proposals) > 3:
            for obsolete_proposal in tracked_proposals[:-3]:
                self.speeches.pop(obsolete_proposal, None)

    def record_summary(self, seat: int, summary: str, *, round_num: Optional[int] = None) -> None:
        round_key = int(round_num if round_num is not None else self.round_num)
        summaries_for_round = self.summaries.setdefault(round_key, {})
        summaries_for_round[seat] = summary[:300]

    def record_reflection(self, seat: int, round_num: int, reflection: str) -> None:
        """Record a vote reflection for a specific seat and round."""
        self.reflections[(round_num, seat)] = reflection[:200]

    def get_previous_reflection(self, seat: int, round_num: int) -> Optional[str]:
        """Get the most recent reflection for a seat before the given round."""
        for r in range(round_num - 1, 0, -1):  # Go backwards from round_num-1 to 1
            if (r, seat) in self.reflections:
                return self.reflections[(r, seat)]
        return None

    def evil_seats(self) -> List[int]:
        """Return all evil seats including Oberon."""
        return [seat for seat, role in self.roles.items() if role in {Role.ASSASSIN, Role.MORGANA, Role.OBERON}]

    def evil_conclave_seats(self) -> List[int]:
        """Return evil seats that participate in evil conclave (excludes Oberon)."""
        return [seat for seat, role in self.roles.items() if role in {Role.ASSASSIN, Role.MORGANA}]

    def assassin_seat(self) -> int:
        for seat, role in self.roles.items():
            if role == Role.ASSASSIN:
                return seat
        raise RuntimeError("Assassin role not assigned")

    def merlin_seat(self) -> int:
        for seat, role in self.roles.items():
            if role == Role.MERLIN:
                return seat
        raise RuntimeError("Merlin role not assigned")


class FSMError(RuntimeError):
    """Raised when the FSM cannot make progress."""


async def run_game_async(
    *,
    seed: Optional[int] = None,
    services: Optional[PhaseServices] = None,
    max_rounds: Optional[int] = None,
    players: int = 5,
) -> GameState:
    """Run the Avalon game FSM and return the final state asynchronously."""

    services = services or PhaseServices()
    rng = build_rng(seed=seed)
    state = GameState(seed=seed, rng=rng, players=players, round_limit=max_rounds)

    _setup_game(state)

    max_iterations = 500
    iterations = 0

    while state.state != State.END_GAME and iterations < max_iterations:
        if max_rounds is not None and state.round_num > max_rounds:
            state.truncated = True
            state.state = State.END_GAME
            break

        iterations += 1

        if state.state == State.PROPOSAL_DRAFT:
            await _handle_proposal_draft(state, services)
        elif state.state == State.DISCUSSION:
            await _handle_discussion(state, services)
        elif state.state == State.PROPOSAL_FINAL:
            await _handle_proposal_final(state, services)
        elif state.state == State.VOTE:
            await _handle_vote(state, services)
        elif state.state == State.MISSION:
            await _handle_mission(state, services)
        elif state.state == State.ROUND_END:
            _handle_round_end(state)
        elif state.state == State.EVIL_CONCLAVE:
            await _handle_evil_conclave(state, services)
        elif state.state == State.ASSASSIN_GUESS:
            await _handle_assassin_guess(state, services)
        else:
            raise FSMError(f"Unhandled state: {state.state}")

    if iterations >= max_iterations:
        raise FSMError("FSM reached iteration cap without ending the game")

    _handle_end_game(state)
    return state


def run_game(
    *,
    seed: Optional[int] = None,
    services: Optional[PhaseServices] = None,
    max_rounds: Optional[int] = None,
    players: int = 5,
) -> GameState:
    """Synchronous convenience wrapper for :func:`run_game_async`."""

    return asyncio.run(
        run_game_async(seed=seed, services=services, max_rounds=max_rounds, players=players)
    )


def get_role_composition(players: int) -> str:
    """Get human-readable role composition for a game in Chinese.

    Returns a string describing all roles in the game.
    """
    from .roles import ROLE_NAMES_ZH
    from collections import Counter

    roles = _get_role_assignment(players)
    role_counts = Counter(roles)

    role_parts = []
    for role, count in role_counts.items():
        role_name = ROLE_NAMES_ZH.get(role, role.value)
        if count > 1:
            role_parts.append(f"{role_name}×{count}")
        else:
            role_parts.append(role_name)

    return "、".join(role_parts)


def _get_role_assignment(players: int) -> List[Role]:
    """Get role assignment based on player count.

    5 players: Merlin, Percival, Assassin, Morgana, Loyal Servant
    6 players: Merlin, Percival, Assassin, Morgana, Loyal Servant x2
    7 players: Merlin, Percival, Assassin, Morgana, Oberon, Loyal Servant x2
    """
    if players == 5:
        return [
            Role.MERLIN,
            Role.PERCIVAL,
            Role.ASSASSIN,
            Role.MORGANA,
            Role.LOYAL_SERVANT,
        ]
    elif players == 6:
        return [
            Role.MERLIN,
            Role.PERCIVAL,
            Role.ASSASSIN,
            Role.MORGANA,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
        ]
    elif players == 7:
        return [
            Role.MERLIN,
            Role.PERCIVAL,
            Role.ASSASSIN,
            Role.MORGANA,
            Role.OBERON,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
        ]
    elif players == 8:
        return [
            Role.MERLIN,
            Role.PERCIVAL,
            Role.ASSASSIN,
            Role.MORGANA,
            Role.OBERON,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
        ]
    elif players == 9:
        return [
            Role.MERLIN,
            Role.PERCIVAL,
            Role.ASSASSIN,
            Role.MORGANA,
            Role.OBERON,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
        ]
    elif players == 10:
        return [
            Role.MERLIN,
            Role.PERCIVAL,
            Role.ASSASSIN,
            Role.MORGANA,
            Role.OBERON,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
        ]
    else:
        raise ValueError(f"Unsupported player count: {players}")


def _setup_game(state: GameState) -> None:
    """Assign roles and prepare the first round."""

    total_players = state.ruleset.players if state.ruleset else 5
    seats = list(shuffle_seats(state.rng, total_players))
    role_order = _get_role_assignment(total_players)
    for seat, role in zip(seats, role_order):
        state.roles[seat] = role

    state.state = State.PROPOSAL_DRAFT


async def _handle_proposal_draft(state: GameState, services: PhaseServices) -> None:
    payload = await _request_proposal_payload(state, services, Phase.PROPOSAL_DRAFT)
    proposal = Proposal(
        leader=state.leader,
        members=canonical_team(payload.action.members),
        round_num=state.round_num,
    )
    state.current_proposal = proposal
    state.state = State.DISCUSSION


async def _handle_discussion(state: GameState, services: PhaseServices) -> None:
    ordered_seats = _seat_rotation(state.leader, state.ruleset.players)
    for seat in ordered_seats:
        payload = await _request_discussion_payload(state, services, seat)
        state.record_speech(seat, payload.speech)
    state.state = State.PROPOSAL_FINAL


async def _handle_proposal_final(state: GameState, services: PhaseServices) -> None:
    if services.proposal_summary is not None:
        summary_payload = await _call_phase_callable(services.proposal_summary, state, state.leader)
        summary_text = getattr(summary_payload, "speech", None)
        if isinstance(summary_payload, dict):
            summary_text = summary_payload.get("speech") or summary_payload.get("action", {}).get("speech")
        if summary_text:
            state.record_summary(state.leader, summary_text, round_num=state.round_num)

    payload = await _request_proposal_payload(state, services, Phase.PROPOSAL_FINAL)
    if state.current_proposal is None:
        state.current_proposal = Proposal(
            leader=state.leader,
            members=canonical_team(payload.action.members),
            round_num=state.round_num,
        )
    else:
        state.current_proposal.members = canonical_team(payload.action.members)
    state.record_proposal(state.current_proposal)
    state.state = State.VOTE


async def _handle_vote_async(state: GameState, services: PhaseServices) -> bool:
    if state.current_proposal is None:
        raise FSMError("Vote phase reached without a current proposal")

    seat_agents = getattr(services, "agents", {}) or {}

    # Create async tasks for all votes
    async def get_vote_for_seat(seat: int) -> Tuple[int, Vote, Optional[str], Any]:
        # Get the vote payload
        agent = seat_agents.get(seat)
        recorded_by_agent = False

        if agent and hasattr(agent, "vote_async"):
            try:
                payload = await agent.vote_async(state, seat)
            except ProviderError as e:
                # Default to APPROVE vote when SSL/API errors occur
                payload = create_default_vote_payload("APPROVE")
            recorded_by_agent = getattr(agent, "records_vote", False)
        else:
            payload = await _request_vote_payload(state, services, seat)

        # Don't record here - will be done sequentially after gather

        vote = Vote(seat=seat, value=payload.action.value)
        reflection = getattr(payload.action, 'reflection', None)
        # Return payload so we can record sequentially
        return seat, vote, reflection, payload if not recorded_by_agent else None

    # Run all votes concurrently but record them sequentially to avoid race conditions
    tasks = [get_vote_for_seat(seat) for seat in _seat_rotation(state.leader, state.ruleset.players)]
    results = await asyncio.gather(*tasks)

    # Process results and record votes sequentially
    votes: List[Vote] = []
    approves = 0
    for seat, vote, reflection, payload in results:
        votes.append(vote)
        if vote.value == VoteValue.APPROVE:
            approves += 1

        # Store reflection if present
        if reflection:
            state.record_reflection(seat, state.round_num, reflection)

        # Record vote here sequentially to avoid concurrent access issues
        recorder = getattr(services, "_recorder", None)
        if recorder and payload:
            payload_dict = payload.model_dump()
            vote_value = payload_dict.get("action", {}).get("value", "ABSTAIN")
            # Convert enum to string value if needed
            if hasattr(vote_value, 'value'):
                value_str = vote_value.value
            else:
                value_str = str(vote_value)

            # Get reasoning for AI players only
            reasoning = None
            if seat != 3:  # Not human player
                reasoning = getattr(payload, 'reasoning', None)

            recorder.record_vote(
                round_num=state.round_num,
                seat=seat,
                value=value_str,
                reasoning=reasoning,
            )

    state.record_vote_round(votes)

    # Process approval
    required_approves = state.ruleset.players // 2 + 1
    approved = approves >= required_approves
    state.current_proposal.approved = approved
    return approved


async def _handle_vote(state: GameState, services: PhaseServices) -> None:
    approved = await _handle_vote_async(state, services)

    if approved:
        state.failed_proposals = 0
        mission = Mission(
            members=state.current_proposal.members,
            round_num=state.round_num,
        )
        state.current_mission = mission
        state.state = State.MISSION
    else:
        state.failed_proposals += 1
        if state.failed_proposals >= state.ruleset.max_failed_proposals:
            auto_fail_mission = Mission(
                members=[],  # No team members for auto-fail missions
                round_num=state.round_num,
                fails=0,  # No actual fail votes, just auto-failed
                result=MissionOutcome.FAIL,
            )
            state.record_mission(auto_fail_mission)
            state.failed_proposals = 0
            state.current_proposal = None
            state.state = State.ROUND_END
        else:
            state.current_proposal = None
            state.advance_leader()
            state.state = State.PROPOSAL_DRAFT


async def _handle_mission_async(state: GameState, services: PhaseServices) -> None:
    if state.current_mission is None:
        raise FSMError("Mission phase reached without a current mission")

    # Create async tasks for all mission members
    async def get_mission_action_for_seat(seat: int) -> MissionOutcome:
        try:
            payload = await _request_mission_payload(state, services, seat)
            return payload.action.value
        except OpenRouterError as e:
            # Default to SUCCESS when SSL/API errors occur (good players try to succeed)
            return MissionOutcome.SUCCESS

    # Run all mission actions concurrently
    tasks = [get_mission_action_for_seat(seat) for seat in state.current_mission.members]
    results = await asyncio.gather(*tasks)

    # Count failures
    fails = sum(1 for result in results if result == MissionOutcome.FAIL)

    # Check if this round requires 2 fails (7+ players, round 4)
    fails_required = 2 if state.ruleset.requires_double_fail(state.round_num) else 1

    state.current_mission.fails = fails
    state.current_mission.result = (
        MissionOutcome.FAIL if fails >= fails_required else MissionOutcome.SUCCESS
    )
    state.record_mission(state.current_mission)
    state.current_mission = None
    state.state = State.ROUND_END


async def _handle_mission(state: GameState, services: PhaseServices) -> None:
    await _handle_mission_async(state, services)



def _handle_round_end(state: GameState) -> None:
    state.current_proposal = None

    if state.is_good_win():
        state.state = State.EVIL_CONCLAVE
        return

    if state.is_evil_win():
        state.winner = Outcome.EVIL_WIN
        state.state = State.END_GAME
        return

    state.round_num += 1
    state.advance_leader()
    state.current_mission = None
    state.state = State.PROPOSAL_DRAFT


async def _handle_evil_conclave(state: GameState, services: PhaseServices) -> None:
    """Evil conclave phase - all evil players including Oberon participate."""
    for seat in state.evil_seats():
        payload = await _request_evil_conclave_payload(state, services, seat)
        state.evil_conclave_messages.append(
            EvilConclaveMessage(round_num=state.round_num, seat=seat, speech=payload.speech)
        )
    state.state = State.ASSASSIN_GUESS


async def _handle_assassin_guess(state: GameState, services: PhaseServices) -> None:
    assassin_seat = state.assassin_seat()
    payload = await _request_assassin_guess_payload(state, services, assassin_seat)
    target = payload.action.target
    correct = target == state.merlin_seat()
    state.assassin_guess = AssassinGuess(seat=assassin_seat, target=target, correct=correct)
    state.winner = Outcome.EVIL_WIN if correct else Outcome.GOOD_WIN
    state.state = State.END_GAME


def _handle_end_game(state: GameState) -> None:
    if state.winner is None:
        # If we reach here without a winner, evil wins by default.
        state.winner = Outcome.EVIL_WIN


async def _call_phase_callable(
    func: Optional[PhaseCallable], *args: Any, **kwargs: Any
) -> Optional[BaseModel | dict]:
    if func is None:
        return None
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _request_proposal_payload(
    state: GameState,
    services: PhaseServices,
    stage: Phase,
) -> ProposalPayload:
    team_size = state.current_team_size()

    async def fetch() -> Optional[BaseModel | dict]:
        return await _call_phase_callable(services.proposal, state, stage)

    return await _execute_phase(
        phase=stage,
        request=fetch,
        fallback=lambda: create_default_proposal_payload(state.leader, team_size),
        post_validate=lambda payload: _validate_proposal_members(
            payload.action.members, team_size, state.leader, state.ruleset.players
        ),
    )


async def _request_discussion_payload(
    state: GameState, services: PhaseServices, seat: int
) -> DiscussionPayload:
    async def fetch() -> Optional[BaseModel | dict]:
        return await _call_phase_callable(services.discussion, state, seat)

    return await _execute_phase(
        phase=Phase.DISCUSSION,
        request=fetch,
        fallback=create_default_discussion_payload,
    )


async def _request_vote_payload(
    state: GameState, services: PhaseServices, seat: int
) -> VotePayload:
    async def fetch() -> Optional[BaseModel | dict]:
        return await _call_phase_callable(services.vote, state, seat)

    return await _execute_phase(
        phase=Phase.VOTE,
        request=fetch,
        fallback=create_default_vote_payload,
    )


async def _request_mission_payload(
    state: GameState, services: PhaseServices, seat: int
) -> MissionPayload:
    async def fetch() -> Optional[BaseModel | dict]:
        return await _call_phase_callable(services.mission, state, seat)

    return await _execute_phase(
        phase=Phase.MISSION,
        request=fetch,
        fallback=create_default_mission_payload,
    )


async def _request_evil_conclave_payload(
    state: GameState, services: PhaseServices, seat: int
) -> EvilConclavePayload:
    async def fetch() -> Optional[BaseModel | dict]:
        return await _call_phase_callable(services.evil_conclave, state, seat)

    return await _execute_phase(
        phase=Phase.EVIL_CONCLAVE,
        request=fetch,
        fallback=create_default_evil_conclave_payload,
    )


async def _request_assassin_guess_payload(
    state: GameState, services: PhaseServices, seat: int
) -> AssassinGuessPayload:
    async def fetch() -> Optional[BaseModel | dict]:
        return await _call_phase_callable(services.assassin_guess, state, seat)

    return await _execute_phase(
        phase=Phase.ASSASSIN_GUESS,
        request=fetch,
        fallback=create_default_assassin_guess_payload,
        post_validate=lambda payload: _validate_assassin_target(payload.action.target, state.ruleset.players),
    )


async def _execute_phase(
    *,
    phase: Phase,
    request: Callable[[], Awaitable[Optional[BaseModel | dict]]] | None,
    fallback: Callable[[], T | Awaitable[T]],
    post_validate: Optional[Callable[[T], None]] = None,
) -> T:
    attempts = 0

    if request is not None:
        while attempts < 1:
            attempts += 1
            try:
                raw = await request()
                if raw is None:
                    raise ValueError("Phase request returned no payload")
                payload = _normalize_payload(phase, raw)
                if post_validate:
                    post_validate(payload)
                return payload
            except KeyboardInterrupt:
                raise
            except Exception:
                continue

    payload_result = fallback()
    if inspect.isawaitable(payload_result):
        payload = await payload_result
    else:
        payload = payload_result
    if post_validate:
        post_validate(payload)
    return payload


def _normalize_payload(phase: Phase, raw: BaseModel | dict) -> T:
    if isinstance(raw, BaseModel):
        return raw  # type: ignore[return-value]
    if isinstance(raw, dict):
        validated = validate_payload(phase=phase.value, payload=raw)
        if validated is None:
            raise ValueError(f"Payload for phase {phase} did not validate")
        return validated  # type: ignore[return-value]
    raise TypeError(f"Unsupported payload type for phase {phase}: {type(raw)!r}")


def _validate_proposal_members(members: Sequence[int], team_size: int, leader: int, max_seats: int = TOTAL_SEATS) -> None:
    if len(members) != team_size:
        raise ValueError("Team size does not match required mission size")
    if len(set(members)) != len(members):
        raise ValueError("Team members must be unique")
    # Note: In standard Avalon, the leader does NOT have to include themselves
    for seat in members:
        _validate_seat(seat, max_seats)


def _validate_assassin_target(target: int, max_seats: int = TOTAL_SEATS) -> None:
    _validate_seat(target, max_seats)


def _validate_seat(seat: int, max_seats: int = TOTAL_SEATS) -> None:
    if not (1 <= seat <= max_seats):
        raise ValueError(f"Seat {seat} is out of range (1-{max_seats})")


def _seat_rotation(start: int, total_seats: int = TOTAL_SEATS) -> List[int]:
    """Return seating order starting from the given leader and wrapping around."""

    order = []
    current = start
    for _ in range(total_seats):
        order.append(current)
        current = (current % total_seats) + 1
    return order
