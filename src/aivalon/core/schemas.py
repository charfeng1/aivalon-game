"""Pydantic contracts for Avalon phase outputs."""

from enum import Enum
from typing import Any, List, Literal, Union

try:  # pragma: no cover - allow running without orjson
    import orjson  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for CI environments
    import json as _json

    class _OrjsonShim:
        @staticmethod
        def loads(data: Any) -> Any:  # type: ignore[override]
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            return _json.loads(data)

        @staticmethod
        def dumps(obj: Any) -> bytes:  # type: ignore[override]
            return _json.dumps(obj).encode("utf-8")

    orjson = _OrjsonShim()  # type: ignore
from pydantic import BaseModel, Field, ValidationError


class Phase(str, Enum):
    """Game phases."""
    SETUP = "SETUP"
    PROPOSAL_DRAFT = "PROPOSAL_DRAFT"
    DISCUSSION = "DISCUSSION"
    PROPOSAL_SUMMARY = "PROPOSAL_SUMMARY"
    PROPOSAL_FINAL = "PROPOSAL_FINAL"
    VOTE = "VOTE"
    MISSION = "MISSION"
    ROUND_END = "ROUND_END"
    EVIL_CONCLAVE = "EVIL_CONCLAVE"
    ASSASSIN_GUESS = "ASSASSIN_GUESS"
    END_GAME = "END_GAME"


class ActionType(str, Enum):
    """Action types."""
    PROPOSE = "PROPOSE"
    VOTE = "VOTE"
    MISSION = "MISSION"
    ASSASSIN_GUESS = "ASSASSIN_GUESS"


class VoteValue(str, Enum):
    """Vote values."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class MissionOutcome(str, Enum):
    """Mission outcomes."""
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"


class ProposeAction(BaseModel):
    """Action for proposing a team."""
    type: Literal[ActionType.PROPOSE] = ActionType.PROPOSE
    members: List[int] = Field(..., min_length=2, max_length=10)


class VoteAction(BaseModel):
    """Action for voting on a proposal."""
    type: Literal[ActionType.VOTE] = ActionType.VOTE
    value: VoteValue
    reflection: str = Field(..., description="Private strategic reflection for this vote and notes from observing this round (aim for ≤200 chars)")


class MissionAction(BaseModel):
    """Action for mission execution."""
    type: Literal[ActionType.MISSION] = ActionType.MISSION
    value: MissionOutcome


class AssassinGuessAction(BaseModel):
    """Action for assassin guessing Merlin."""
    type: Literal[ActionType.ASSASSIN_GUESS] = ActionType.ASSASSIN_GUESS
    target: int


class ProposalPayload(BaseModel):
    """Payload for PROPOSAL phases (draft/final)."""
    action: ProposeAction


class DiscussionPayload(BaseModel):
    """Payload for DISCUSSION phase."""

    thinking: str = Field(
        ...,

        description="Private analysis/planning not shown to other players",
    )
    speech: str = Field(...)


class VotePayload(BaseModel):
    """Payload for VOTE phase."""
    action: VoteAction


class MissionPayload(BaseModel):
    """Payload for MISSION phase."""
    action: MissionAction


class EvilConclavePayload(BaseModel):
    """Payload for EVIL_CONCLAVE phase."""

    thinking: str = Field(
        ...,
        description="Private evil coordination notes",
    )
    speech: str = Field(..., min_length=5, max_length=300)


class AssassinGuessPayload(BaseModel):
    """Payload for ASSASSIN_GUESS phase."""
    action: AssassinGuessAction


# Union of all phase payloads for validation
PhasePayload = Union[
    ProposalPayload,
    VotePayload,
    MissionPayload,
    EvilConclavePayload,
    AssassinGuessPayload
]


class SchemaValidationError(Exception):
    """Raised when a payload fails schema validation."""
    def __init__(self, phase: str, errors: list):
        self.phase = phase
        self.errors = errors
        super().__init__(f"Validation failed for {phase} phase: {errors}")


def validate_payload(*, phase: str, payload: dict) -> Union[BaseModel, None]:
    """Validate phase payloads and return structured data or raise SchemaValidationError."""
    try:
        if phase in [Phase.PROPOSAL_DRAFT, Phase.PROPOSAL_FINAL]:
            return ProposalPayload(**payload)
        elif phase in [Phase.DISCUSSION, Phase.PROPOSAL_SUMMARY]:
            return DiscussionPayload(**payload)
        elif phase == Phase.VOTE:
            return VotePayload(**payload)
        elif phase == Phase.MISSION:
            return MissionPayload(**payload)
        elif phase == Phase.EVIL_CONCLAVE:
            return EvilConclavePayload(**payload)
        elif phase == Phase.ASSASSIN_GUESS:
            return AssassinGuessPayload(**payload)
        else:
            return None
    except ValidationError as e:
        raise SchemaValidationError(phase, e.errors())


def create_default_proposal_payload(leader: int, team_size: int) -> ProposalPayload:
    """Create default proposal payload."""
    # Simple default: consecutive seats starting from leader
    members = []
    for i in range(team_size):
        seat = (leader + i - 1) % 5 + 1
        members.append(seat)
    normalized = sorted(dict.fromkeys(members))[:team_size]
    return ProposalPayload(
        action=ProposeAction(members=normalized)
    )


def create_default_vote_payload() -> VotePayload:
    """Create default vote payload."""
    return VotePayload(
        action=VoteAction(value=VoteValue.REJECT, reflection="No specific observations to note.")
    )


def create_default_mission_payload() -> MissionPayload:
    """Create default mission payload."""
    return MissionPayload(
        action=MissionAction(value=MissionOutcome.SUCCESS)
    )


def create_default_discussion_payload() -> DiscussionPayload:
    """Create default discussion payload."""
    return DiscussionPayload(
        thinking="Fallback planning due to validation failure",
        speech="Fallback discussion due to validation failure",
    )


def create_default_evil_conclave_payload() -> EvilConclavePayload:
    """Create default evil conclave payload."""
    return EvilConclavePayload(
        thinking="Fallback evil planning due to validation failure",
        speech="Fallback conclave message due to validation failure",
    )


def create_default_assassin_guess_payload() -> AssassinGuessPayload:
    """Create default assassin guess payload."""
    return AssassinGuessPayload(
        action=AssassinGuessAction(target=1)
    )


def serialize_for_logging(obj: BaseModel) -> str:
    """Serialize Pydantic models to JSON string for logging/transcripts."""
    return orjson.dumps(obj.model_dump()).decode('utf-8')


# Doctests
def _test_proposal_payload():
    """
    Test ProposalPayload creation and validation.
    
    >>> payload = ProposalPayload(
    ...     action=ProposeAction(members=[1, 2])
    ... )
    >>> isinstance(payload, ProposalPayload)
    True
    >>> payload.action.type
    <ActionType.PROPOSE: 'PROPOSE'>
    >>> payload.action.members
    [1, 2]
    """
    pass


def _test_discussion_payload():
    """
    Test DiscussionPayload creation and validation.
    
    >>> payload = DiscussionPayload(
    ...     thinking="Plan my talking points",
    ...     speech="This is a test speech"
    ... )
    >>> isinstance(payload, DiscussionPayload)
    True
    >>> payload.thinking
    'Plan my talking points'
    >>> payload.speech
    'This is a test speech'
    """
    pass


def _test_vote_payload():
    """
    Test VotePayload creation and validation.
    
    >>> payload = VotePayload(
    ...     action=VoteAction(value=VoteValue.APPROVE, reflection="Example reflection for testing")
    ... )
    >>> isinstance(payload, VotePayload)
    True
    >>> payload.action.value is VoteValue.APPROVE
    True
    """
    pass


def _test_mission_payload():
    """
    Test MissionPayload creation and validation.
    
    >>> payload = MissionPayload(
    ...     action=MissionAction(value=MissionOutcome.FAIL)
    ... )
    >>> isinstance(payload, MissionPayload)
    True
    >>> payload.action.value is MissionOutcome.FAIL
    True
    """
    pass


def _test_evil_conclave_payload():
    """
    Test EvilConclavePayload creation and validation.
    
    >>> payload = EvilConclavePayload(thinking="Coordinate with ally", speech="Secret evil plan")
    >>> isinstance(payload, EvilConclavePayload)
    True
    >>> payload.thinking
    'Coordinate with ally'
    >>> len(payload.speech) > 0
    True
    """
    pass


def _test_assassin_guess_payload():
    """
    Test AssassinGuessPayload creation and validation.
    
    >>> payload = AssassinGuessPayload(
    ...     action=AssassinGuessAction(target=3)
    ... )
    >>> isinstance(payload, AssassinGuessPayload)
    True
    >>> payload.action.target
    3
    """
    pass


def _test_validation():
    """
    Test payload validation function.
    
    >>> # Test valid proposal payload
    >>> proposal_data = {
    ...     "action": {"type": "PROPOSE", "members": [1, 2]}
    ... }
    >>> result = validate_payload(phase="PROPOSAL_DRAFT", payload=proposal_data)
    >>> isinstance(result, ProposalPayload)
    True
    
    >>> # Test invalid payload
    >>> invalid_data = {"invalid": "data"}
    >>> try:
    ...     result = validate_payload(phase="PROPOSAL_DRAFT", payload=invalid_data)
    ... except SchemaValidationError:
    ...     pass  # Expected
    ... else:
    ...     assert False, "Should have raised SchemaValidationError"
    """
    pass


def _test_defaults():
    """
    Test default payload creation functions.
    
    >>> proposal = create_default_proposal_payload(leader=1, team_size=2)
    >>> isinstance(proposal, ProposalPayload)
    True
    >>> len(proposal.action.members) == 2
    True
    
    >>> vote = create_default_vote_payload()
    >>> isinstance(vote, VotePayload)
    True
    >>> vote.action.value == VoteValue.REJECT
    True
    
    >>> mission = create_default_mission_payload()
    >>> isinstance(mission, MissionPayload)
    True
    >>> mission.action.value == MissionOutcome.SUCCESS
    True
    """
    pass


def _test_serialization():
    """
    Test serialization helper.
    
    >>> payload = DiscussionPayload(
    ...     speech="Test speech"
    ... )
    >>> json_str = serialize_for_logging(payload)
    >>> isinstance(json_str, str)
    True
    >>> '"speech":"Test speech"' in json_str
    True
    """
    pass


# Run doctests when script is executed directly
if __name__ == "__main__":
    import doctest
    doctest.testmod()
