"""DashScope (Alibaba Cloud) provider using OpenAI-compatible client."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypeVar

try:  # pragma: no cover - optional dependency for speed
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

# Alibaba Cloud DashScope base URL
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TIMEOUT = 60.0

T = TypeVar("T", bound=BaseModel)

LeakFilter = type(sanitize_speech)


class DashScopeProvider(ModelProvider):
    """DashScope (Alibaba Cloud) provider using OpenAI-compatible API."""

    @property
    def json_capability(self) -> JsonCapability:
        """DashScope supports JSON object mode but not strict JSON schema."""
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
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ProviderError("DASHSCOPE_API_KEY is not set")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.leak_filter = leak_filter or sanitize_speech

        # Initialize OpenAI client with DashScope endpoint
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            **kwargs
        )

    def close(self) -> None:
        """Close the underlying OpenAI client."""
        self._client.close()

    def __enter__(self) -> "DashScopeProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def call_phase(self, request: ProviderRequest[T]) -> ProviderResponse[T]:
        """Call the DashScope API for a phase and validate the response."""

        logger = LOGGER.bind(phase=request.phase, model=request.model, seat=request.seat)

        try:
            payload_options = dict(request.metadata.get("options", {}))

            create_kwargs: Dict[str, Any] = {
                "model": request.model,
                "messages": request.messages,
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
            logger.error("dashscope.api_error", error=str(exc))
            raise ProviderError(f"DashScope API error: {exc}") from exc

        # Convert response to our internal format
        raw_response = response.model_dump()

        violations: List[str] = []
        try:
            payload_model, violations = self._parse_payload(request, raw_response)
            payload_model = self._maybe_filter_leaks(request, payload_model, logger)
        except SchemaValidationError as exc:
            logger.warning(
                "dashscope.schema_validation_failed",
                errors=getattr(exc, "errors", exc.args),
            )
            raise
        except (ValueError, KeyError) as exc:
            logger.warning("dashscope.payload_parse_failed", error=str(exc))
            raise ProviderError(f"Failed to parse DashScope payload: {exc}") from exc

        logger.info(
            "dashscope.call_success",
            usage=raw_response.get("usage"),
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
        )

        # Log raw response to markdown file for debugging (same as OpenRouter)
        self._log_raw_response_to_file(request, raw_response)

        if violations:
            logger.info("dashscope.payload_adjusted", violations=violations)

        return ProviderResponse(
            payload=payload_model,
            raw_response=raw_response,
            usage=raw_response.get("usage"),
            retries=0,  # OpenAI client handles retries internally
            reasoning=None,  # DashScope doesn't provide reasoning tokens
            violations=violations or None,
            json_capability_used=JsonCapability.NONE,
        )

    def _parse_payload(self, request: ProviderRequest[T], data: Dict[str, Any]) -> tuple[T, List[str]]:
        """Parse and validate the API response payload."""

        choices = data.get("choices")
        if not choices:
            raise ValueError("DashScope response missing 'choices'")

        message = choices[0].get("message")
        if not message:
            raise ValueError("DashScope response missing 'message'")

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            raise ValueError("DashScope response missing 'tool_calls'")

        selected_call = self._select_tool_call(request, tool_calls)
        function = selected_call.get("function", {})
        arguments = function.get("arguments")
        if arguments is None:
            raise ValueError("DashScope tool call missing arguments")

        try:
            parsed_payload = orjson.loads(arguments)
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"DashScope tool arguments were not valid JSON: {exc}") from exc

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
            logger.info("dashscope.leak_sanitised", original=speech, sanitized=sanitized, seat=seat)
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
        raise ValueError("DashScope response did not include any tool calls")

    def _log_raw_response_to_file(self, request: ProviderRequest[T], raw_response: Dict[str, Any]) -> None:
        """Log raw model responses to a markdown file for debugging."""
        import datetime
        from pathlib import Path
        import os

        # Create debug logs directory if it doesn't exist
        debug_dir = Path("debug_logs")
        debug_dir.mkdir(exist_ok=True)

        # Generate session-specific filename using seed or unique ID
        seed = getattr(request, 'seed', None)
        if seed is not None:
            session_id = f"seed_{seed}"
        else:
            # Use process ID for sessionless calls
            session_id = f"session_{os.getpid()}"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        log_file = debug_dir / f"responses_{session_id}.md"

        # Format the log entry with enhanced metadata
        choices = raw_response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        tool_calls = message.get("tool_calls", [])
        content = message.get("content", "")
        usage = raw_response.get("usage", {})

        # Extract round information if available in metadata
        metadata = getattr(request, 'metadata', {}) or {}
        context_meta = metadata.get('context', {})
        round_info = context_meta.get('round', 'N/A')
        leader_info = 'N/A'  # DashScope doesn't have leader info in metadata

        log_entry = f"""
# 🎯 {timestamp} | {request.phase} | Round {round_info} | Seat {request.seat} | {request.model}

## 📋 Request Context
| Field | Value |
|-------|--------|
| **Phase** | `{request.phase}` |
| **Round** | `{round_info}` |
| **Leader** | `{leader_info}` |
| **Seat** | `{request.seat}` |
| **Model** | `{request.model}` |
| **Tool Name** | `{getattr(request, 'tool_name', 'N/A')}` |
| **Temperature** | `{getattr(request, 'temperature', 'N/A')}` |
| **Seed** | `{getattr(request, 'seed', 'N/A')}` |

## 📊 Usage Statistics
| Metric | Count |
|--------|-------|
| **Prompt Tokens** | `{usage.get('prompt_tokens', 'N/A')}` |
| **Completion Tokens** | `{usage.get('completion_tokens', 'N/A')}` |
| **Total Tokens** | `{usage.get('total_tokens', 'N/A')}` |
| **Reasoning Tokens** | `{(usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 'N/A')}` |

## 🔧 Raw Response
```json
{orjson.dumps(raw_response, option=orjson.OPT_INDENT_2).decode('utf-8')}
```

## 📝 Response Analysis
- **Message Content**: {content or '*(empty)*'}
- **Tool Calls Count**: {len(tool_calls)}
- **Response ID**: `{raw_response.get('id', 'N/A')}`
- **Model Used**: `{raw_response.get('model', 'N/A')}`

"""

        if tool_calls:
            log_entry += "## 🛠️ Tool Calls\n\n"
            for i, call in enumerate(tool_calls):
                function = call.get("function", {})
                call_id = call.get("id", "N/A")
                args_raw = function.get("arguments", "null")

                # Try to format arguments as readable JSON
                try:
                    if args_raw and args_raw != "null":
                        args_parsed = orjson.loads(args_raw)
                        args_formatted = orjson.dumps(args_parsed, option=orjson.OPT_INDENT_2).decode('utf-8')
                    else:
                        args_formatted = args_raw
                except:
                    args_formatted = args_raw

                log_entry += f"""### 🔧 Tool Call #{i+1}
| Field | Value |
|-------|---------|
| **Call ID** | `{call_id}` |
| **Function Name** | `{function.get("name", "unknown")}` |
| **Type** | `{call.get("type", "unknown")}` |

**Arguments:**
```json
{args_formatted}
```

"""

        log_entry += "\n---\n"

        # Append to log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)


# Register DashScope provider with the factory
ProviderFactory.register("dashscope", DashScopeProvider)