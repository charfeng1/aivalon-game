"""Phase context builders for OpenRouter requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

try:  # pragma: no cover - optional dependency for speed
    import orjson  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for test envs
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

from ..providers.providers import ProviderRequest
from .fsm import MAX_FAILED_PROPOSALS, TOTAL_SEATS
from .schemas import (
    AssassinGuessPayload,
    DiscussionPayload,
    EvilConclavePayload,
    MissionOutcome,
    MissionPayload,
    ProposalPayload,
    VotePayload,
    VoteValue,
)
from pydantic import BaseModel
from ..tools import (
    ALL_TOOLS, 
    PAYLOAD_TO_TOOL_NAME, 
    TOOLSET_VERSION,
    build_tools_for_round,
    get_static_tools_for_game
)

try:  # Optional import until Task 6 wires full role knowledge
    from . import roles
except Exception:  # pragma: no cover - best effort fallback
    roles = None  # type: ignore

DEFAULT_CONFIG_PATH = Path("config/config.local.json")
DEFAULT_MODEL = "deepseek-chat"
SPEECH_PHASES = {"DISCUSSION", "EVIL_CONCLAVE"}

PHASE_LABELS: Dict[str, str] = {
    "PROPOSAL_DRAFT": "组队提案（草案）",
    "PROPOSAL_SUMMARY": "终稿总结",
    "PROPOSAL_FINAL": "组队提案（定案）",
    "DISCUSSION": "公开讨论",
    "VOTE": "提案投票",
    "MISSION": "执行任务",
    "EVIL_CONCLAVE": "邪恶密谈",
    "ASSASSIN_GUESS": "刺客猜测",
    "ROUND_END": "回合结算",
    "SETUP": "开局准备",
    "END_GAME": "游戏结束",
}

ROLE_NAME_MAP = {
    "Merlin": "梅林",
    "Percival": "帕西瓦尔",
    "Assassin": "刺客",
    "Morgana": "莫甘娜",
    "Loyal Servant": "忠诚侍从",
}


def _seat_label(seat: Any) -> str:
    try:
        number = int(seat)
        return f"{number}号"
    except Exception:
        return f"{seat}号"


def _phase_label(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase)


def _round_label(value: Any) -> str:
    try:
        number = int(value)
        return f"第{number}回合"
    except Exception:
        return f"第{value}回合"


def _vote_value_label(value: Any) -> str:
    if ContextBuilder._is_approve(value):
        return "赞成"
    return "反对"

PayloadModel = TypeVar("PayloadModel", bound="BasePhaseModel")
PayloadModel = TypeVar("PayloadModel", bound=BaseModel)


@dataclass(slots=True)
class PhaseDefinition(Generic[PayloadModel]):
    """Metadata describing how to prompt and validate a phase."""

    name: str
    payload_model: type[PayloadModel]
    speech_field: Optional[str] = None
    tool_name: Optional[str] = None

    @property
    def schema_name(self) -> str:
        return self.payload_model.__name__

    @property
    def schema(self) -> Dict[str, Any]:
        # Pydantic produces deterministic JSON schema dictionaries.
        return self.payload_model.model_json_schema()

    @property
    def tool_choice(self) -> Optional[Dict[str, Any]]:
        if not self.tool_name:
            return None
        return {"type": "function", "function": {"name": self.tool_name, "strict": True}}


PHASE_DEFINITIONS: Dict[str, PhaseDefinition[Any]] = {
    "PROPOSAL_DRAFT": PhaseDefinition(
        "PROPOSAL_DRAFT",
        ProposalPayload,
        tool_name=PAYLOAD_TO_TOOL_NAME["ProposalPayload"],
    ),
    "PROPOSAL_SUMMARY": PhaseDefinition(
        "PROPOSAL_SUMMARY",
        DiscussionPayload,
        speech_field="speech",
        tool_name=PAYLOAD_TO_TOOL_NAME["DiscussionPayload"],
    ),
    "PROPOSAL_FINAL": PhaseDefinition(
        "PROPOSAL_FINAL",
        ProposalPayload,
        tool_name=PAYLOAD_TO_TOOL_NAME["ProposalPayload"],
    ),
    "DISCUSSION": PhaseDefinition(
        "DISCUSSION",
        DiscussionPayload,
        speech_field="speech",
        tool_name=PAYLOAD_TO_TOOL_NAME["DiscussionPayload"],
    ),
    "VOTE": PhaseDefinition(
        "VOTE",
        VotePayload,
        tool_name=PAYLOAD_TO_TOOL_NAME["VotePayload"],
    ),
    "MISSION": PhaseDefinition(
        "MISSION",
        MissionPayload,
        tool_name=PAYLOAD_TO_TOOL_NAME["MissionPayload"],
    ),
    "EVIL_CONCLAVE": PhaseDefinition(
        "EVIL_CONCLAVE",
        EvilConclavePayload,
        speech_field="speech",
        tool_name=PAYLOAD_TO_TOOL_NAME["EvilConclavePayload"],
    ),
    "ASSASSIN_GUESS": PhaseDefinition(
        "ASSASSIN_GUESS",
        AssassinGuessPayload,
        tool_name=PAYLOAD_TO_TOOL_NAME["AssassinGuessPayload"],
    ),
}


@dataclass(slots=True)
class ReasoningConfig:
    """Reasoning configuration for models."""
    
    enabled: bool = False
    per_model_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fallback_config: Dict[str, Any] = field(default_factory=lambda: {"max_tokens": 400})
    exclude_from_response: bool = False
    phases: Dict[str, bool] = field(default_factory=dict)

    def get_reasoning_config_for_model(self, model: str, phase: str = "") -> Optional[Dict[str, Any]]:
        """Get reasoning configuration for a specific model and phase."""
        if not self.enabled:
            return None
        
        # Check if this phase is enabled for reasoning
        if phase and not self.phases.get(phase, True):
            return None
        
        # Get model-specific config
        model_config = self.per_model_config.get(model, {})
        if not model_config.get("enabled", True):
            return None
        
        # Build reasoning config
        reasoning_config: Dict[str, Any] = {}

        # Determine cap
        if "max_tokens" in model_config:
            reasoning_config["max_tokens"] = model_config["max_tokens"]
        elif "effort" not in model_config:
            # Use fallback max_tokens (default 400)
            cap = self.fallback_config.get("max_tokens")
            if cap is not None:
                reasoning_config["max_tokens"] = cap

        # Add effort if specifically requested and max_tokens not set
        # Add exclude flag if configured
        if self.exclude_from_response:
            reasoning_config["exclude"] = True

        return reasoning_config if reasoning_config else None


@dataclass(slots=True)
class ContextConfig:
    """Configuration derived from config/config.local.json."""

    seat_models: Dict[int, str] = field(default_factory=dict)
    seat_providers: Dict[int, str] = field(default_factory=dict)
    default_model: str = DEFAULT_MODEL
    default_provider: str = "deepseek"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    headers: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)

    def model_for_seat(self, seat: int) -> str:
        return self.seat_models.get(seat, self.default_model)

    def provider_for_seat(self, seat: int) -> str:
        return self.seat_providers.get(seat, self.default_provider)


@dataclass(slots=True)
class PhaseContext(Generic[PayloadModel]):
    """Outcome of building a phase-specific context."""

    request: ProviderRequest[PayloadModel]
    prompt_payload: Dict[str, Any]
    is_speech_phase: bool


RoleCardProvider = Callable[[Any, int], str]


def load_context_config(path: Path = DEFAULT_CONFIG_PATH) -> ContextConfig:
    """Load context configuration from disk, falling back to defaults."""

    if not path.exists():
        return ContextConfig()

    data = orjson.loads(path.read_bytes())

    seat_models: Dict[int, str] = {}
    for key, value in data.get("seat_models", {}).items():
        try:
            seat_models[int(key)] = str(value)
        except (ValueError, TypeError):
            continue

    seat_providers: Dict[int, str] = {}
    for key, value in data.get("seat_providers", {}).items():
        try:
            seat_providers[int(key)] = str(value)
        except (ValueError, TypeError):
            continue

    request_cfg = data.get("request", {}) or {}
    temperature = request_cfg.get("temperature")
    top_p = request_cfg.get("top_p")
    # Any leftover keys become generic request options (e.g. frequency_penalty)
    options = {
        str(k): v
        for k, v in request_cfg.items()
        if k not in {"temperature", "top_p"}
    }

    headers = {
        str(k): str(v)
        for k, v in data.get("headers", {}).items()
        if isinstance(k, str)
    }

    default_model = str(data.get("default_model", DEFAULT_MODEL))
    default_provider = str(data.get("default_provider", "deepseek"))

    # Parse reasoning configuration
    reasoning_data = data.get("reasoning", {})
    reasoning_config = ReasoningConfig(
        enabled=reasoning_data.get("enabled", False),
        per_model_config=reasoning_data.get("per_model_config", {}),
        fallback_config=reasoning_data.get("fallback_config", {"max_tokens": 400}),
        exclude_from_response=reasoning_data.get("exclude_from_response", False),
        phases=reasoning_data.get("phases", {}),
    )

    return ContextConfig(
        seat_models=seat_models,
        seat_providers=seat_providers,
        default_model=default_model,
        default_provider=default_provider,
        temperature=temperature,
        top_p=top_p,
        headers=headers,
        options=options,
        reasoning=reasoning_config,
    )


@dataclass(slots=True)
class ContextBuilder:
    """Builds structured phase prompts for the FSM to feed into OpenRouter."""

    config: ContextConfig = field(default_factory=ContextConfig)
    role_card_provider: Optional[RoleCardProvider] = None
    violation_notes: Dict[Tuple[str, int], List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role_card_provider is None:
            self.role_card_provider = default_role_card_provider

    # Public API ------------------------------------------------------------------

    def build_phase_context(
        self,
        *,
        state: Any,
        phase: Union[str, Any],
        seat: int,
        seed: Optional[int] = None,
        memory: Optional[Iterable[str]] = None,
    ) -> PhaseContext[Any]:
        phase_name = _normalize_phase(phase)
        if phase_name not in PHASE_DEFINITIONS:
            raise ValueError(f"Unknown phase: {phase_name}")

        definition = PHASE_DEFINITIONS[phase_name]
        notes = self._consume_notes(phase_name, seat)
        prompt_payload = self._compile_prompt_payload(state, phase_name, seat, notes=notes, memory=memory)
        system_prompt = self._build_system_prompt(state, phase_name, seat)
        user_content = self._render_prompt_payload(prompt_payload, schema_name=definition.schema_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        metadata: Dict[str, Any] = {
            "context": {
                "round": getattr(state, "round_num", None),
                "phase": phase_name,
                "seat": seat,
            }
        }
        if self.config.headers:
            metadata["headers"] = dict(self.config.headers)
        if self.config.options:
            metadata.setdefault("options", {}).update(self.config.options)

        # Add reasoning configuration
        model = self.config.model_for_seat(seat)
        reasoning_config = self.config.reasoning.get_reasoning_config_for_model(model, phase_name)
        if reasoning_config:
            metadata.setdefault("options", {})["reasoning"] = reasoning_config
            metadata["context"]["reasoning_enabled"] = True

        if definition.tool_name:
            tooling_meta = metadata.setdefault("tooling", {})
            tooling_meta["version"] = TOOLSET_VERSION
            tooling_meta["tool"] = definition.tool_name

        # Build round-aware tools for proposal phases
        tools = ALL_TOOLS  # Default fallback
        if hasattr(state, "ruleset") and state.ruleset is not None:
            round_num = getattr(state, "round_num", 1) or 1
            if phase_name in {"PROPOSAL_DRAFT", "PROPOSAL_FINAL"}:
                # Use round-specific tools that enforce exact team size
                tools = build_tools_for_round(state.ruleset, round_num)
            else:
                # Use static tools for other phases
                static_tools = get_static_tools_for_game(state.ruleset)
                tools = list(static_tools.values())

        request = ProviderRequest(
            phase=phase_name,
            model=self.config.model_for_seat(seat),
            messages=messages,
            schema_name=definition.schema_name,
            schema=definition.schema,
            expected_phase=phase_name,
            seat=seat,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=seed if seed is not None else getattr(state, "seed", None),
            metadata=metadata,
            speech_field=definition.speech_field,
            tools=tools,
            tool_choice=definition.tool_choice,
            tool_name=definition.tool_name,
            tool_version=TOOLSET_VERSION,
        )

        return PhaseContext(
            request=request,
            prompt_payload=prompt_payload,
            is_speech_phase=definition.speech_field is not None,
        )

    # Internals -------------------------------------------------------------------

    def _compile_prompt_payload(
        self,
        state: Any,
        phase: str,
        seat: int,
        *,
        notes: Optional[List[str]] = None,
        memory: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        round_num = int(getattr(state, "round_num", 1) or 1)
        note_messages = self._resolve_note_messages(notes or [])
        
        # Get ruleset-specific information
        ruleset = getattr(state, "ruleset", None)
        if ruleset:
            ruleset_name = f"阿瓦隆-{ruleset.players}人"
            total_seats = ruleset.players
        else:
            # Fallback for legacy code
            ruleset_name = "阿瓦隆-5人"
            total_seats = TOTAL_SEATS

        payload: Dict[str, Any] = {
            "meta": {
                "seat": seat,
                "phase": phase,
                "ruleset": ruleset_name,
            },
            "state": {
                "round": round_num,
                "leader": getattr(state, "leader", None),
                "team": self._current_team_size(state),
                "fails": getattr(state, "failed_proposals", None),
            },
            "leaders": list(range(1, total_seats + 1)),
            "history": self._build_history_entries(state, round_num),
            "round": self._build_round_section(state, round_num),
            "speeches": self._build_speech_entries(state, phase),
            "summaries": self._build_summary_entries(state, phase),
            "reflection": self._build_reflection_entry(state, seat, round_num),
            "directive": self._phase_directive(state, phase, seat, note_messages=note_messages),
            "schema": PHASE_DEFINITIONS[phase].schema_name,
            "memory": list(memory or []),
        }

        return payload

    def _build_system_prompt(self, state: Any, phase: str, seat: int) -> str:
        role_card = self.role_card_provider(state, seat)
        team_size = self._current_team_size(state)
        seat_name = _seat_label(seat)
        phase_name = _phase_label(phase)
        failed_proposals = int(getattr(state, "failed_proposals", 0) or 0)
        
        # Get ruleset-specific information
        ruleset = getattr(state, "ruleset", None)
        if ruleset:
            player_count = ruleset.players
            game_name = f"{player_count}人《阿瓦隆》"
            team_sizes_str = "-".join(str(ruleset.team_size_for_round(r)) for r in range(1, 6))
            task_rule = f"五轮任务依次要求 {team_sizes_str} 人，请勿提名不符合该轮人数的队伍。"

            # Get role composition
            from .fsm import get_role_composition
            role_composition = get_role_composition(player_count)
            game_info = f"本局游戏共{player_count}人，角色配置：{role_composition}。"
        else:
            # Fallback for legacy code
            player_count = 5
            game_name = "5人《阿瓦隆》"
            task_rule = "五轮任务依次要求 2-3-2-3-3 人，请勿提名不符合该轮人数的队伍。"
            game_info = "本局游戏共5人，角色配置：梅林、帕西瓦尔、刺客、莫甘娜、忠诚侍从。"

        base_lines = [
            f"你是一个的{game_name}对局中的{seat_name}。",
            role_card.strip(),
            f"当前阶段：{phase_name}。本回合需要组队人数：{team_size}人。",
            "",
            "=== 阿瓦隆游戏规则 ===",
            game_info,
            task_rule,
            "注意：队伍是集合（顺序无关）。请使用规范顺序（例如 [1,3,4]）引用队伍。",
            "提案规则：领袖可以提名任意符合人数的队伍组合，无需包含自己。",
            "投票规则：所有玩家对提案投票，需要超过半数赞成才能通过。",
            "规则提醒：若连续三次提案未获批准，该轮任务会自动判定失败。",
            "任务执行关键规则：善阵营玩家必须且只能选择成功，无法使任务失败；邪恶阵营玩家可选择成功或失败。因此，任何任务失败都意味着队伍中必定包含至少一位邪恶玩家。",
            "每位恶阵营玩家在一次任务中只能投一张坏票，因此有几张坏票就代表几个坏人"
            "胜利条件：善阵营需要完成3次任务胜利；邪恶阵营需要失败3次任务，或刺客猜中梅林身份。",
            "特殊阶段：如果善阵营完成3次任务，进入刺客猜测阶段，刺客有一次机会猜测梅林身份翻盘。",
            "",
            "=== 游戏俚语 ===",
            "任务队伍=车队，任务失败=翻车，邪恶阵营玩家=坏人。理解并适当使用这些术语与其他玩家交流。",
            "",
            "使用提供的工具函数回应此阶段。不要返回自由格式文本，而要调用相应的工具函数。",
            "DSL说明：",
            "- @state 表示当前回合号、领袖、所需队伍人数等信息（team 即本轮任务需带的座位数）。",
            "- @mission R1 区块列出当前回合的提案、投票与任务状态。",
            "- 提案行格式为 P[编号]: L{实际领袖座位} [队伍] -> 状态，状态取值 P=待定、Y=通过、N=否决。",
            "- 投票行格式为 V: [1:Y,2:N,…] -> 赞成-反对，Y 表示赞成，N 表示反对。",
            "- @speeches P1-s1 显示最近提案的发言历史；s{座位}(YOU) 表示你的发言。",
            "- @speeches 标签对应某一次提案 (Proposal)，而不是回合。一次回合中可能有多个提案与对应发言记录。",
            "当发言时，记得将dsl语言转为自然语言"
            "避免使用过于通用的语言，避免circlejerking,多利用场上局势和玩家发言，过往投票，任务记录和逻辑推理进行分析。",
            "专注游戏内容，避免打招呼，自我介绍，守护阿瓦隆等无关内容"
        ]
        if failed_proposals >= max(0, MAX_FAILED_PROPOSALS - 1):
            base_lines.append(
                f"警告：当前已连续 {failed_proposals} 次提案被拒，再无共识本轮任务将自动失败。"
            )
        return "\n".join(line for line in base_lines if line)

    def _phase_directive(
        self,
        state: Any,
        phase: str,
        seat: int,
        *,
        note_messages: Optional[List[str]] = None,
    ) -> str:
        team_size = self._current_team_size(state)
        phase_def = PHASE_DEFINITIONS.get(phase)
        tool_name = phase_def.tool_name if phase_def else None
        tool_hint = f"`{tool_name}`" if tool_name else "对应的工具函数"
        if phase in {"PROPOSAL_DRAFT", "PROPOSAL_FINAL"}:
            directive = (
                f"请调用工具函数 {tool_hint} 提名恰好 {team_size} 位不同的座位。"
                " 目标是推动任务成功：综合既往投票与任务结果，优先选择可信成员，"
                "避免重复使用已失败组合中的嫌疑人，除非你能充分说明再次验证的理由。"
            )
        elif phase == "PROPOSAL_SUMMARY":
            directive = (
                f"请调用工具函数 {tool_hint}，先整理私人思路，再给出公开总结。"
                " 请保持结构清晰，避免在公开发言中泄露隐藏身份或队友信息。"
            )
        elif phase == "DISCUSSION":
            directive = f"使用工具函数 {tool_hint} 回应此阶段，不要返回自由文本。"
        elif phase == "VOTE":
            directive = (
                f"请调用工具函数 {tool_hint} 表达你对提案的投票立场。"
                " 结合身份与场上信息，简洁说明支持或反对的核心理由。"
            )
            current_team = tuple(getattr(getattr(state, "current_proposal", None), "members", ()))
            if current_team:
                round_history = getattr(state, "stance_history", {}).get(getattr(state, "round_num", 0), {})
                previous_vote = round_history.get(current_team, {}).get(seat)
                if previous_vote is not None:
                    prev_label = "赞成" if previous_vote == VoteValue.APPROVE else "反对"
                    directive += (
                        f" 你此前对该队伍投过{prev_label}票；若本次改变立场，请在反思内容中写一句理由。"
                    )
        elif phase == "MISSION":
            directive = (
                f"请调用工具函数 {tool_hint}，根据阵营选择 SUCCESS 或 FAIL 并提交任务结果。"
            )
        elif phase == "EVIL_CONCLAVE":
            directive = (
                f"请调用工具函数 {tool_hint}。善方已经完成3次任务获胜，现在是邪恶阵营最后机会！"
                "在私人笔记中分析谁是梅林，在密谈中与队友讨论刺杀目标，协调暗杀策略。"
                "记住：只有刺客猜中梅林才能翻盘获胜，否则邪恶失败。"
            )
        elif phase == "ASSASSIN_GUESS":
            directive = f"请调用工具函数 {tool_hint}，指出你认为的梅林座位并给出最终判断。"
        else:
            directive = "请调用对应的工具函数作答，遵循该阶段的规则要求。"

        if note_messages:
            directive = " ".join([directive.strip()] + [msg.strip() for msg in note_messages])
        return directive

    def _render_prompt_payload(self, payload: Dict[str, Any], *, schema_name: str) -> str:
        """Render prompt payload using the v0.2 DSL format."""

        lines: List[str] = []

        meta = payload.get("meta", {})
        state_info = payload.get("state", {})
        leaders = payload.get("leaders", [])
        history_entries = payload.get("history", [])
        round_section = payload.get("round", {})
        speeches_blocks = payload.get("speeches", [])
        reflection = payload.get("reflection")
        directive = payload.get("directive")
        round_tag = round_section.get("tag", f"M{state_info.get('round', '?')}")

        seat_value = meta.get("seat", "?")
        actor_seat = None
        try:
            actor_seat = int(seat_value)
        except (TypeError, ValueError):
            try:
                actor_seat = int(str(seat_value).strip())
            except Exception:
                actor_seat = None
        phase_value = meta.get("phase", "?")
        lines.append(
            "@meta seat={seat} phase={phase} ruleset={ruleset}".format(
                seat=seat_value,
                phase=phase_value,
                ruleset=meta.get("ruleset", "avalon-5p"),
            )
        )

        leader_value = state_info.get("leader")
        lines.append(
            "@state R={round} leader={leader} team={team} fails={fails}".format(
                round=state_info.get("round", "?"),
                leader=leader_value if leader_value is not None else "?",
                team=state_info.get("team") if state_info.get("team") is not None else "?",
                fails=state_info.get("fails", 0) or 0,
            )
        )

        if leaders:
            lines.append("@leaders " + " ".join(str(seat) for seat in leaders))

        if history_entries:
            lines.append("@history " + " | ".join(history_entries))

        required_team = state_info.get("team") if state_info.get("team") is not None else "?"
        lines.append(f"@mission {round_tag} team_size={required_team}")
        for entry in round_section.get("lines", []):
            lines.append(entry)

        if speeches_blocks:
            for proposal_id, speech_entries in speeches_blocks:
                lines.append(f"@speeches {proposal_id}")
                for seat_id, text in speech_entries:
                    label = f"s{seat_id}(YOU)" if actor_seat == seat_id else f"s{seat_id}"
                    lines.append(f"{label}: {text}")

        summary_blocks = payload.get("summaries", [])
        if summary_blocks:
            for summary_round, summary_entries in summary_blocks:
                lines.append(f"@summary R{summary_round}")
                for seat_id, text in summary_entries:
                    label = f"Seat {seat_id}(YOU)" if actor_seat == seat_id else f"Seat {seat_id}"
                    lines.append(f"{label}: {text}")

        if reflection:
            lines.append(f"@memory {reflection}")

        for memo in payload.get("memory", []) or []:
            lines.append(f"@memory {memo}")

        if directive:
            lines.append(f"@do {directive}")

        return "\n".join(lines)

    def _build_history_entries(self, state: Any, current_round: int) -> List[str]:
        entries: List[str] = []

        for proposal in getattr(state, "proposals", []) or []:
            if getattr(proposal, "round_num", 0) >= current_round:
                continue
            members = ",".join(str(member) for member in getattr(proposal, "members", []) or [])
            status = self._proposal_status_char(getattr(proposal, "approved", None))
            proposal_id = getattr(proposal, "proposal_id", "") or f"P{proposal.round_num}-s{getattr(proposal, 'leader', '?')}"
            vote_summary = self._vote_summary(state, proposal.round_num, proposal.members)
            entries.append(
                f"PR {proposal_id} [{members}] -> {status}{vote_summary}"
            )

        # Remove duplicate VR entries - votes are now only shown inline with proposals
        # for vote_entry in getattr(state, "votes", []) or []:
        #     ... (removed to eliminate duplication)

        for mission in getattr(state, "missions", []) or []:
            if getattr(mission, "round_num", 0) >= current_round:
                continue
            members = ",".join(str(member) for member in getattr(mission, "members", []) or [])
            fails = getattr(mission, "fails", 0) or 0
            result = self._mission_result_char(getattr(mission, "result", None))
            entries.append(f"MR R{mission.round_num} [{members}] F={fails} -> {result}")

        return entries

    def _build_round_section(self, state: Any, current_round: int) -> Dict[str, Any]:
        lines: List[str] = []

        round_proposals: List[Any] = [
            proposal
            for proposal in getattr(state, "proposals", []) or []
            if getattr(proposal, "round_num", None) == current_round
        ]

        seen_ids = {id(proposal) for proposal in round_proposals}

        current_proposal = getattr(state, "current_proposal", None)
        if (
            current_proposal is not None
            and getattr(current_proposal, "round_num", None) == current_round
            and id(current_proposal) not in seen_ids
        ):
            round_proposals.append(current_proposal)

        for attempt_idx, proposal in enumerate(round_proposals, start=1):
            members = ",".join(str(member) for member in getattr(proposal, "members", []) or [])
            status = self._proposal_status_char(getattr(proposal, "approved", None))
            proposal_id = getattr(proposal, "proposal_id", "") or f"P{proposal.round_num}-s{getattr(proposal, 'leader', '?')}"
            vote_summary = self._vote_summary(state, proposal.round_num, proposal.members)
            lines.append(
                f"Attempt{attempt_idx} [{proposal_id}]: L{getattr(proposal, 'leader', '?')} [{members}] -> {status}{vote_summary}"
            )

        summary_entries = getattr(state, "summaries", {}).get(current_round, {})
        for seat_id in sorted(summary_entries.keys()):
            text = summary_entries[seat_id]
            lines.append(f"Summary Seat {seat_id}: {text}")

        for vote_entry in getattr(state, "votes", []) or []:
            if len(vote_entry) >= 2:
                round_num, votes = vote_entry[0], vote_entry[1]
            else:
                continue
            if round_num != current_round:
                continue
            pairs = ",".join(self._format_vote_pair(vote) for vote in votes)
            approve = sum(1 for vote in votes if self._is_approve(getattr(vote, "value", None)))
            reject = len(votes) - approve
            lines.append(f"V: [{pairs}] -> {approve}-{reject}")

        for mission in getattr(state, "missions", []) or []:
            if getattr(mission, "round_num", None) != current_round:
                continue
            members = ",".join(str(member) for member in getattr(mission, "members", []) or [])
            fails = getattr(mission, "fails", 0) or 0
            result = self._mission_result_char(getattr(mission, "result", None))
            lines.append(f"M: [{members}] F={fails} -> {result}")

        return {"tag": f"M{current_round}", "lines": lines}

    def _build_speech_entries(self, state: Any, phase: str) -> List[Tuple[str, List[Tuple[int, str]]]]:
        if phase not in {"DISCUSSION", "VOTE"}:
            return []

        speech_map: Dict[str, Dict[int, str]] = getattr(state, "speeches", {}) or {}
        if not speech_map:
            return []

        # Get recent proposals (last 2 proposals)
        recent_proposals = sorted(speech_map.keys())[-2:]
        blocks: List[Tuple[str, List[Tuple[int, str]]]] = []
        for proposal_id in recent_proposals:
            proposal_entries = []
            proposal_speeches = speech_map.get(proposal_id, {}) or {}
            for seat_id, text in proposal_speeches.items():
                if not text:
                    continue
                proposal_entries.append((seat_id, text))
            if proposal_entries:
                blocks.append((proposal_id, proposal_entries))
        return blocks

    def _build_summary_entries(self, state: Any, phase: str) -> List[Tuple[int, List[Tuple[int, str]]]]:
        if phase not in {"PROPOSAL_FINAL", "VOTE", "MISSION", "PROPOSAL_SUMMARY"}:
            return []
        summary_map: Dict[int, Dict[int, str]] = getattr(state, "summaries", {}) or {}
        if not summary_map:
            return []
        current_round = int(getattr(state, "round_num", 1) or 1)
        blocks: List[Tuple[int, List[Tuple[int, str]]]] = []
        for round_num in sorted(summary_map.keys())[-2:]:
            if round_num > current_round:
                continue
            entries = [(seat_id, summary_map[round_num][seat_id]) for seat_id in sorted(summary_map[round_num].keys())]
            if entries:
                blocks.append((round_num, entries))
        return blocks

    def _build_reflection_entry(self, state: Any, seat: int, round_num: int) -> Optional[str]:
        """Build reflection entry from previous round for this seat."""
        if hasattr(state, 'get_previous_reflection'):
            reflection = state.get_previous_reflection(seat, round_num)
            if reflection:
                return f"上一回合 {_round_label(round_num - 1)} 反思：\"{reflection}\""
        return None

    @staticmethod
    def _proposal_status_char(status: Optional[bool]) -> str:
        if status is True:
            return "Y"
        if status is False:
            return "N"
        return "P"

    @staticmethod
    def _format_vote_pair(vote: Any) -> str:
        value = getattr(vote, "value", None)
        mark = "Y" if ContextBuilder._is_approve(value) else "N"
        return f"{getattr(vote, 'seat', '?')}:{mark}"

    @staticmethod
    def _mission_result_char(result: Any) -> str:
        if isinstance(result, MissionOutcome):
            return "S" if result == MissionOutcome.SUCCESS else "F"
        raw = getattr(result, "value", result)
        if isinstance(raw, str) and raw.upper().startswith("S"):
            return "S"
        return "F"

    def _vote_summary(self, state: Any, round_num: int, members: Tuple[int, ...]) -> str:
        history = getattr(state, "stance_history", {}) or {}
        team_history = history.get(round_num, {}).get(tuple(members))
        if not team_history:
            return ""
        sorted_votes = sorted(team_history.items())
        pairs = []
        approve = 0
        for seat, value in sorted_votes:
            mark = "Y" if self._is_approve(value) else "N"
            if self._is_approve(value):
                approve += 1
            pairs.append(f"{seat}:{mark}")
        reject = len(sorted_votes) - approve
        return f" (Votes: [{','.join(pairs)}] -> {approve}-{reject})"

    @staticmethod
    def _is_approve(value: Any) -> bool:
        if isinstance(value, VoteValue):
            return value == VoteValue.APPROVE
        raw = getattr(value, "value", value)
        if isinstance(raw, str):
            return raw.upper().startswith("APPROVE")
        return False

    @staticmethod
    def _current_team_size(state: Any) -> Optional[int]:
        if hasattr(state, "current_team_size"):
            try:
                return int(state.current_team_size())
            except Exception:  # pragma: no cover - fallback safety
                return None
        return None

    def register_violation(self, *, phase: str, seat: int, code: str) -> None:
        key = (phase, seat)
        self.violation_notes.setdefault(key, []).append(code)

    def _consume_notes(self, phase: str, seat: int) -> List[str]:
        key = (phase, seat)
        return self.violation_notes.pop(key, [])

    @staticmethod
    def _resolve_note_messages(notes: Iterable[str]) -> List[str]:
        messages: List[str] = []
        for note in notes:
            if note == "speech_truncated":
                messages.append(
                    "提醒：你上一条回复超过300字符被截断，请确保本次发言控制在300字符以内。"
                )
        return messages


def default_role_card_provider(state: Any, seat: int) -> str:
    """Fallback role card if Task 6 has not supplied detailed knowledge injections."""

    role_name: Optional[str] = None
    role_map = getattr(state, "roles", {}) or {}
    role = role_map.get(seat)
    if role is not None:
        raw_name = getattr(role, "value", str(role))
        role_name = ROLE_NAME_MAP.get(raw_name, raw_name)

    if roles and hasattr(roles, "get_role_card"):
        try:
            return roles.get_role_card(seat=seat, role=role_map.get(seat), all_seats=role_map)
        except NotImplementedError:
            pass
        except TypeError:
            # Roles module may expect additional parameters; fall back to simple message.
            pass

    if role_name:
        base = f"你的隐藏身份是{role_name}。"
    else:
        base = "暂时无法得知你的隐藏身份。"
    return base + " 请通过提案、投票、任务和讨论来影响局势，谨慎透露真实身份。"


def _normalize_phase(phase: Union[str, Any]) -> str:
    if hasattr(phase, "value"):
        return str(phase.value)
    return str(phase)


DEFAULT_CONTEXT_BUILDER = ContextBuilder(config=load_context_config())


def build_phase_context(
    *,
    state: Any,
    phase: Union[str, Any],
    seat: int,
    seed: Optional[int] = None,
    builder: Optional[ContextBuilder] = None,
) -> PhaseContext[Any]:
    """Convenience wrapper using the default builder."""

    active_builder = builder or DEFAULT_CONTEXT_BUILDER
    return active_builder.build_phase_context(state=state, phase=phase, seat=seat, seed=seed)


def is_speech_phase(phase: Union[str, Any]) -> bool:
    """Return True when the phase requires leak filtering of speech output."""

    return _normalize_phase(phase) in SPEECH_PHASES
def _round_label(value: Any) -> str:
    try:
        number = int(value)
        return f"第{number}回合"
    except Exception:
        return f"第{value}回合"
