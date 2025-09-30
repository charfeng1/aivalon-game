"""GLM (Zhipu AI) provider using OpenAI-compatible client."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypeVar

try:  # pragma: no cover - optional dependency for speed
    import orjson  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when orjson missing
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
from ..utils.dotenv_shim import load_dotenv
from ..utils.openai_shim import OpenAI
from pydantic import BaseModel

from ..utils.leak_filter import sanitize_speech
from .providers import (
    ModelProvider,
    ProviderRequest,
    ProviderResponse,
    ProviderError,
    ProviderFactory,
    JsonCapability
)
from ..core.schemas import SchemaValidationError, validate_payload

load_dotenv()

LOGGER = structlog.get_logger(__name__)

# GLM API base URL
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_TIMEOUT = 60.0

T = TypeVar("T", bound=BaseModel)

LeakFilter = type(sanitize_speech)


class GLMProvider(ModelProvider):
    """GLM (Zhipu AI) provider using OpenAI-compatible API."""

    @property
    def json_capability(self) -> JsonCapability:
        """GLM supports JSON object mode but not strict JSON schema."""
        return JsonCapability.JSON_OBJECT

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        leak_filter: Optional[LeakFilter] = None,
        **kwargs: Any
    ) -> None:
        self.api_key = api_key or os.getenv("GLM_API_KEY") or os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not self.api_key:
            raise ProviderError("GLM_API_KEY, ZHIPUAI_API_KEY, or ZHIPU_API_KEY is not set")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.leak_filter = leak_filter or sanitize_speech

        # Initialize OpenAI client with GLM endpoint
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            **kwargs
        )

    def close(self) -> None:
        """Close the underlying OpenAI client."""
        self._client.close()

    def __enter__(self) -> "GLMProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def call_phase(self, request: ProviderRequest[T]) -> ProviderResponse[T]:
        """Call the GLM API for a phase and validate the response."""

        logger = LOGGER.bind(phase=request.phase, model=request.model, seat=request.seat)

        # Try up to 2 times
        for attempt in range(2):
            try:
                payload_options = dict(request.metadata.get("options", {}))

                messages = list(request.messages)
                # On retry, add error message
                if attempt == 1:
                    messages.append({
                        "role": "user",
                        "content": "ERROR: Your previous response was invalid. You MUST use the provided tool/function call. Do not respond with plain text. Use the tool now."
                    })
                    logger.warning("glm.retrying_with_tool_reminder", attempt=attempt + 1)

                create_kwargs: Dict[str, Any] = {
                    "model": request.model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "seed": request.seed,
                    **payload_options,
                }
                if request.tools:
                    create_kwargs["tools"] = request.tools
                if request.tool_choice:
                    create_kwargs["tool_choice"] = request.tool_choice

                response = self._client.chat.completions.create(**create_kwargs)

            except Exception as exc:
                logger.error("glm.api_error", error=str(exc), attempt=attempt + 1)
                if attempt == 1:  # Last attempt
                    raise ProviderError(f"GLM API error: {exc}") from exc
                continue

            # Convert response to our internal format
            raw_response = response.model_dump()

            violations: List[str] = []
            try:
                payload_model, violations = self._parse_payload(request, raw_response)
                payload_model = self._maybe_filter_leaks(request, payload_model, logger)
                # Success! Break out of retry loop
                break
            except SchemaValidationError as exc:
                logger.warning(
                    "glm.schema_validation_failed",
                    errors=getattr(exc, "errors", exc.args),
                    attempt=attempt + 1,
                )
                if attempt == 1:  # Last attempt - use fallback
                    logger.error("glm.falling_back_to_default", phase=request.phase)
                    payload_model = self._create_fallback_payload(request)
                    violations = ["fallback_used"]
                    break
                continue
            except (ValueError, KeyError) as exc:
                logger.warning("glm.payload_parse_failed", error=str(exc), attempt=attempt + 1)
                if attempt == 1:  # Last attempt - use fallback
                    logger.error("glm.falling_back_to_default", phase=request.phase)
                    payload_model = self._create_fallback_payload(request)
                    violations = ["fallback_used"]
                    break
                continue

        logger.info(
            "glm.call_success",
            usage=raw_response.get("usage"),
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
        )

        if violations:
            logger.info("glm.payload_adjusted", violations=violations)

        return ProviderResponse(
            payload=payload_model,
            raw_response=raw_response,
            usage=raw_response.get("usage"),
            retries=0,  # OpenAI client handles retries internally
            reasoning=None,  # GLM doesn't provide reasoning tokens
            violations=violations or None,
            json_capability_used=JsonCapability.NONE,
        )

    def _parse_payload(self, request: ProviderRequest[T], data: Dict[str, Any]) -> tuple[T, List[str]]:
        """Parse and validate the API response payload."""

        choices = data.get("choices")
        if not choices:
            raise ValueError("GLM response missing 'choices'")

        message = choices[0].get("message")
        if not message:
            raise ValueError("GLM response missing 'message'")

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            raise ValueError("GLM response missing 'tool_calls'")

        selected_call = self._select_tool_call(request, tool_calls)
        function = selected_call.get("function", {})
        arguments = function.get("arguments")
        if arguments is None:
            raise ValueError("GLM tool call missing arguments")

        try:
            parsed_payload = orjson.loads(arguments)
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"GLM tool arguments were not valid JSON: {exc}") from exc

        violations: List[str] = []

        # Validate against our schemas
        try:
            validated = validate_payload(phase=request.expected_phase, payload=parsed_payload)
        except SchemaValidationError as exc:
            # First try length remediation
            try:
                validated, violations = self._attempt_length_remediation(
                    request=request,
                    payload_dict=parsed_payload,
                    error=exc,
                )
            except SchemaValidationError:
                raise exc

        if validated is None:
            raise SchemaValidationError(request.expected_phase, ["Validation returned None"])

        return validated, violations  # type: ignore[return-value]

    def _maybe_filter_leaks(
        self,
        request: ProviderRequest[T],
        payload_model: T,
        logger: Any,
    ) -> T:
        """Apply leak filtering if configured."""

        speech_field = request.speech_field
        if not speech_field or not self.leak_filter or not hasattr(payload_model, speech_field):
            return payload_model

        seat = request.seat or -1
        speech = getattr(payload_model, speech_field)
        if not isinstance(speech, str):
            return payload_model

        sanitized, needs_retry = self.leak_filter(seat, speech)
        if sanitized != speech:
            logger.info("glm.leak_sanitised", original=speech, sanitized=sanitized, seat=seat)
            payload_model = payload_model.model_copy(update={speech_field: sanitized})

        return payload_model

    def _attempt_length_remediation(
        self,
        *,
        request: ProviderRequest[T],
        payload_dict: Dict[str, Any],
        error: SchemaValidationError,
    ) -> tuple[T, List[str]]:
        """Attempt to fix string length validation errors."""

        violations: List[str] = []
        speech_field = request.speech_field or "speech"

        for detail in getattr(error, "errors", []):
            if detail.get("type") != "string_too_long":
                continue

            loc = detail.get("loc") or ()
            if isinstance(loc, (list, tuple)) and speech_field in loc:
                max_length = detail.get("ctx", {}).get("max_length") or 300
                speech_value = payload_dict.get(speech_field)
                if isinstance(speech_value, str):
                    truncated = speech_value[:max_length]
                    payload_dict[speech_field] = truncated
                    violations.append("speech_truncated")
                    break

        if not violations:
            raise error

        validated = validate_payload(phase=request.expected_phase, payload=payload_dict)
        if validated is None:
            raise error

        return validated, violations  # type: ignore[return-value]

    def _select_tool_call(
        self,
        request: ProviderRequest[Any],
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        expected_name = request.tool_name
        if expected_name:
            for call in tool_calls:
                function = call.get("function", {}) or {}
                if function.get("name") == expected_name:
                    return call
        if tool_calls:
            return tool_calls[0]
        raise ValueError("GLM response did not include any tool calls")

    def _create_fallback_payload(self, request: ProviderRequest[T]) -> T:
        """Create a safe fallback payload when the model fails to respond correctly."""
        from ..core.schemas import (
            Phase,
            ProposalPayload,
            DiscussionPayload,
            VotePayload,
            VoteValue,
            VoteAction,
            MissionPayload,
            MissionOutcome,
            MissionAction,
            AssassinGuessPayload,
            AssassinGuessAction,
            ProposeAction,
            EvilConclavePayload,
        )
        import random

        phase = request.expected_phase

        # Vote: Always APPROVE
        if phase == Phase.VOTE:
            return VotePayload(
                action=VoteAction(
                    value=VoteValue.APPROVE,
                    reflection="Fallback: Auto-approved due to tool call failure."
                )
            )  # type: ignore[return-value]

        # Mission: Always SUCCESS
        elif phase == Phase.MISSION:
            return MissionPayload(
                action=MissionAction(value=MissionOutcome.SUCCESS)
            )  # type: ignore[return-value]

        # Assassin: Random seat from available seats
        elif phase == Phase.ASSASSIN_GUESS:
            options = request.metadata.get("options", {})
            available_seats = options.get("availableSeats", [1, 2, 3, 4, 5])
            target = random.choice(available_seats) if available_seats else 1
            return AssassinGuessPayload(
                action=AssassinGuessAction(target=target)
            )  # type: ignore[return-value]

        # Proposal: Random team of correct size
        elif phase in {Phase.PROPOSAL_DRAFT, Phase.PROPOSAL_FINAL}:
            options = request.metadata.get("options", {})
            team_size = options.get("teamSize", 2)
            available_seats = options.get("availableSeats", [1, 2, 3, 4, 5])
            members = random.sample(available_seats, min(team_size, len(available_seats)))
            return ProposalPayload(
                action=ProposeAction(members=sorted(members))
            )  # type: ignore[return-value]

        # Discussion/Summary/Conclave: Empty speech
        elif phase == Phase.DISCUSSION or phase == Phase.PROPOSAL_SUMMARY:
            return DiscussionPayload(
                thinking="Fallback response due to tool call failure.",
                speech=""
            )  # type: ignore[return-value]

        elif phase == Phase.EVIL_CONCLAVE:
            return EvilConclavePayload(
                thinking="Fallback response due to tool call failure.",
                speech=""
            )  # type: ignore[return-value]

        # Default fallback
        else:
            return DiscussionPayload(
                thinking="Unknown phase fallback.",
                speech=""
            )  # type: ignore[return-value]


# Register GLM provider with the factory
ProviderFactory.register("glm", GLMProvider)