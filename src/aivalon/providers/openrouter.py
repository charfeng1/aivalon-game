"""OpenRouter provider that speaks the OpenAI-compatible tool API."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, TypeVar

try:  # pragma: no cover - allow running without orjson
    import orjson  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for CI/test
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
from pydantic import BaseModel

from ..utils.structlog_shim import structlog
from ..utils.dotenv_shim import load_dotenv
from ..utils.openai_shim import OpenAI
from ..utils.leak_filter import sanitize_speech
from .providers import (
    JsonCapability,
    ModelProvider,
    ProviderError,
    ProviderFactory,
    ProviderRequest,
    ProviderResponse,
)
from ..core.schemas import SchemaValidationError, validate_payload

load_dotenv()

LOGGER = structlog.get_logger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 60.0

T = TypeVar("T", bound=BaseModel)

LeakFilter = Callable[[int, str], tuple[str, bool]]


class OpenRouterClient(ModelProvider):
    """Thin wrapper around OpenRouter chat completions with tool-call validation."""

    @property
    def json_capability(self) -> JsonCapability:
        """OpenRouter advertises JSON schema support, but tool calls drive parsing."""
        return JsonCapability.JSON_SCHEMA

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        leak_filter: Optional[LeakFilter] = None,
        app_name: Optional[str] = None,
        site_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.leak_filter = leak_filter or sanitize_speech
        self.app_name = app_name or os.getenv("OPENROUTER_APP_NAME")
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL")

        self._default_headers = self._build_default_headers()
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            default_headers=self._default_headers,
            **kwargs,
        )

    def close(self) -> None:
        """Close the underlying OpenAI-compatible client."""
        self._client.close()

    def __enter__(self) -> "OpenRouterClient":  # pragma: no cover - context helper
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context helper
        self.close()

    def call_phase(self, request: ProviderRequest[T]) -> ProviderResponse[T]:
        """Call the OpenRouter API for a phase and validate the response."""

        logger = LOGGER.bind(phase=request.phase, model=request.model, seat=request.seat)

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

        extra_headers = self._metadata_headers(request.metadata)
        if extra_headers:
            create_kwargs["extra_headers"] = extra_headers

        try:
            response = self._client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            logger.error("openrouter.api_error", error=str(exc))
            raise ProviderError(f"OpenRouter API error: {exc}") from exc

        raw_response = response.model_dump()

        # Log raw response to markdown file for debugging
        self._log_raw_response_to_file(request, raw_response)

        violations: List[str] = []
        try:
            payload_model, violations = self._parse_payload(request, raw_response)
            payload_model = self._maybe_filter_leaks(request, payload_model, logger)
        except SchemaValidationError as exc:
            logger.warning(
                "openrouter.schema_validation_failed",
                errors=getattr(exc, "errors", exc.args),
            )
            raise
        except (ValueError, KeyError) as exc:
            logger.warning("openrouter.payload_parse_failed", error=str(exc))
            raise ProviderError(f"Failed to parse OpenRouter payload: {exc}") from exc

        reasoning_info = self._extract_reasoning_info(raw_response)
        if violations:
            logger.info("openrouter.payload_adjusted", violations=violations)

        logger.info(
            "openrouter.call_success",
            usage=raw_response.get("usage"),
            reasoning_tokens=reasoning_info.get("reasoning_tokens"),
            has_reasoning=bool(reasoning_info.get("reasoning") or reasoning_info.get("reasoning_details")),
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
        )

        reasoning_payload = None
        if any(reasoning_info.values()):
            reasoning_payload = reasoning_info

        return ProviderResponse(
            payload=payload_model,
            raw_response=raw_response,
            usage=raw_response.get("usage"),
            retries=0,
            reasoning=reasoning_payload,
            violations=violations or None,
            json_capability_used=JsonCapability.NONE,
        )

    # Internal helpers ------------------------------------------------------------------

    def _build_default_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.app_name:
            headers["X-Title"] = self.app_name
        if self.site_url:
            headers["X-Referer"] = self.site_url
            headers.setdefault("HTTP-Referer", self.site_url)
        return headers

    def _metadata_headers(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if not metadata:
            return headers
        extra_headers = metadata.get("headers")
        if isinstance(extra_headers, dict):
            headers.update({str(k): str(v) for k, v in extra_headers.items()})
        return headers

    def _parse_payload(self, request: ProviderRequest[T], data: Dict[str, Any]) -> tuple[T, List[str]]:
        choices = data.get("choices")
        if not choices or not choices[0]:
            raise ValueError("OpenRouter response missing 'choices' or choices[0] is None")
        message = choices[0].get("message") if choices[0] else None
        if not message:
            raise ValueError("OpenRouter response missing 'message'")
        tool_calls = message.get("tool_calls") or []
        parsed_payload: Dict[str, Any]

        if tool_calls:
            selected_call = self._select_tool_call(request, tool_calls)
            function = selected_call.get("function", {})
            arguments = function.get("arguments")
            if arguments is None:
                raise ValueError("OpenRouter tool call missing arguments")
            try:
                parsed_payload = orjson.loads(arguments)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"OpenRouter tool arguments were not valid JSON: {exc}") from exc
        else:
            content = message.get("content")
            if content is None:
                raise ValueError("OpenRouter response missing 'tool_calls' and 'content'")

            content_text = self._extract_content_text(content)

            try:
                parsed_payload = orjson.loads(content_text)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"OpenRouter content was not valid JSON: {exc}") from exc

        violations: List[str] = []

        try:
            validated = validate_payload(phase=request.expected_phase, payload=parsed_payload)
        except SchemaValidationError as exc:
            validated, violations = self._attempt_length_remediation(
                request=request,
                payload_dict=parsed_payload,
                error=exc,
            )
        if validated is None:
            raise SchemaValidationError(request.expected_phase, ["Validation returned None"])
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
        raise ValueError("OpenRouter response did not include any tool calls")

    def _maybe_filter_leaks(
        self,
        request: ProviderRequest[T],
        payload_model: T,
        logger: Any,
    ) -> T:
        speech_field = request.speech_field
        if not speech_field or not self.leak_filter or not hasattr(payload_model, speech_field):
            return payload_model

        seat = request.seat or -1
        speech = getattr(payload_model, speech_field)
        if not isinstance(speech, str):
            return payload_model

        sanitized, needs_retry = self.leak_filter(seat, speech)
        if sanitized != speech:
            logger.info("openrouter.leak_sanitised", original=speech, sanitized=sanitized, seat=seat)
            payload_model = payload_model.model_copy(update={speech_field: sanitized})
        return payload_model

    def _attempt_length_remediation(
        self,
        *,
        request: ProviderRequest[T],
        payload_dict: Dict[str, Any],
        error: SchemaValidationError,
    ) -> tuple[T, List[str]]:
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

    @staticmethod
    def _extract_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        fragments.append(text)
                elif isinstance(item, str):
                    fragments.append(item)
            if fragments:
                return "".join(fragments)
        raise ValueError("Unsupported OpenRouter content format")

    def _extract_reasoning_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reasoning information from OpenRouter response."""
        choices = data.get("choices", [])
        if not choices or choices[0] is None:
            return {}

        message = choices[0].get("message") if choices[0] else {}
        if message is None:
            message = {}
        usage = data.get("usage") if data else {}
        if usage is None:
            usage = {}

        return {
            "reasoning": message.get("reasoning"),
            "reasoning_details": message.get("reasoning_details"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
        }

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
        message = {}
        if choices and choices[0]:
            message = choices[0].get("message", {}) or {}
        tool_calls = message.get("tool_calls", [])
        content = message.get("content", "")
        usage = raw_response.get("usage", {}) or {}

        # Extract round information if available in metadata
        metadata = getattr(request, 'metadata', {}) or {}
        round_info = metadata.get('round_num', 'N/A')
        leader_info = metadata.get('leader', 'N/A')

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
| **Reasoning Tokens** | `{usage.get('completion_tokens_details', {}).get('reasoning_tokens', 'N/A')}` |

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
|-------|-------|
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


def call_model(
    request: ProviderRequest[T],
    *,
    client: Optional[OpenRouterClient] = None,
    **client_kwargs: Any,
) -> ProviderResponse[T]:
    """Convenience wrapper to execute a phase call with an optional ephemeral client."""

    owns_client = False
    if client is None:
        client = OpenRouterClient(**client_kwargs)
        owns_client = True

    try:
        return client.call_phase(request)
    finally:
        if owns_client:
            client.close()


# Register OpenRouter provider with the factory
ProviderFactory.register("openrouter", OpenRouterClient)
