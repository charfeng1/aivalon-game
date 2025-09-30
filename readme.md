# Aivalon

**阿瓦隆 - Agentic Workflow Experimentation on Avalon**

Aivalon is an AI-powered multi-agent platform for the classic social deduction game **Avalon**. Watch AI agents negotiate, deceive, deduce, and strategize in real-time through an intuitive web interface. Mix and match LLMs from different providers, inject human players, and observe emergent social intelligence in action.

## Highlights

- **Interactive Web Interface** – Real-time game visualization with pixel-art aesthetics, live AI agent speeches, voting animations, and mission results
- **Multi-Provider AI Support** – Seamlessly mix models from OpenRouter, DashScope (Qwen), DeepSeek, GLM, and Kimi in a single game with per-seat provider routing
- **Human-AI Hybrid Games** – Jump into any seat as a human player and compete against AI agents
- **Bilingual Support** – Full English and Chinese UI with i18n support
- **Finite State Machine Architecture** – Robust game logic handling all Avalon phases: evil conclave, proposals, discussions, votes, missions, and assassin endgame
- **Custom DSL for AI Context** – Structured game state representation optimized for LLM understanding (see `contest/dsl.md`)
- **Provider Abstraction Layer** – Easy integration of new AI providers through unified interface
- **Retry & Fallback Mechanisms** – Automatic error handling ensures games never crash from AI failures
- **Rich Transcripts** – JSON/Markdown game logs with full prompt/response history for research and debugging

## Project Layout

### Backend (Python/FastAPI)
```
src/aivalon/
├── core/
│   ├── context.py       # Prompt compilation & DSL rendering
│   ├── fsm.py           # GameState, mission logic, PhaseServices
│   ├── roles.py         # Localized role cards & private knowledge
│   ├── rulesets.py      # Game rule definitions and team sizes
│   ├── schemas.py       # Pydantic payload definitions + defaults
│   └── transcript.py    # Transcript generation and formatting
├── providers/
│   ├── providers.py     # Provider abstraction and factory
│   ├── openrouter.py    # OpenRouter LLM provider
│   ├── dashscope_provider.py  # Alibaba DashScope (Qwen) provider
│   ├── deepseek_provider.py   # DeepSeek LLM provider
│   ├── glm_provider.py        # GLM (Zhipu AI) provider
│   ├── kimi_provider.py       # Kimi (Moonshot AI) provider
│   └── multi_provider.py      # Multi-provider routing system
├── services/
│   ├── cli.py           # Typer CLI for command-line games
│   ├── web_api.py       # FastAPI REST API for web interface
│   └── web_session.py   # Game session management for web
├── config/
│   └── model_registry.py # Centralized model/provider registry
├── utils/
│   └── *.py             # Utility modules (logging, RNG, filters)
├── agents.py            # Seat agent implementations
├── human_player.py      # Human player interface
└── tools.py             # Tool schema definitions for LLMs
```

### Frontend (React/TypeScript)
```
frontend/
├── src/
│   ├── App.tsx                      # Root component, game state management
│   ├── main.tsx                     # React entry point
│   ├── types.ts                     # TypeScript interfaces
│   ├── pixel-styles.scss            # Main stylesheet (medieval pixel-art theme)
│   ├── components/
│   │   ├── NegotiationTable.tsx     # Game table with seats, avatars, speeches
│   │   ├── SeatConfigScreen.tsx     # Seat configuration modal
│   │   └── LanguageSwitcher.tsx     # Language toggle component
│   └── i18n/
│       ├── index.ts                 # i18next configuration
│       └── locales/
│           ├── en-US.json           # English translations
│           └── zh-CN.json           # Chinese translations
├── index.html                       # Entry HTML
├── package.json                     # Dependencies
├── vite.config.ts                   # Vite configuration
└── README.md                        # Frontend-specific documentation
```

### Supporting Directories
- `config/` – Seat-to-model mapping, context configuration
- `contest/` – Project documentation for contest submission (architecture, DSL guide)
- `documents/` – Technical documentation and guides
- `runs/` – Generated game transcripts (JSON/Markdown)
- `debug_logs/` – Detailed API request/response logs
- `tests/` – Comprehensive test suite

## Getting Started

### Prerequisites

**Backend:**
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for dependency management

**Frontend:**
- Node.js 18+
- npm or yarn

**API Keys:**
- At least one AI provider API key (OpenRouter, DeepSeek, DashScope, GLM, or Kimi)

### Installation

**1. Install Backend Dependencies:**
```bash
uv sync
```

**2. Install Frontend Dependencies:**
```bash
cd frontend
npm install
cd ..
```

### Configuration

**1. Create `.env` file** (in project root) with your API keys:
```env
OPENROUTER_API_KEY=sk-or-v1-...
DASHSCOPE_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
GLM_API_KEY=...
KIMI_API_KEY=...
```

**Supported Providers:**
- **OpenRouter** – Access to 200+ models (GPT, Claude, Gemini, Llama, etc.)
- **DashScope** – Alibaba's Qwen model family
- **DeepSeek** – DeepSeek-Chat and DeepSeek-Reasoner
- **GLM** – Zhipu AI's ChatGLM models (glm-4-plus, glm-4-flash)
- **Kimi** – Moonshot AI's Kimi models

**2. (Optional) Configure default seats** in `config/config.local.json`:
```json
{
  "seat_models": {
    "1": "google/gemini-2.5-flash-lite-preview-09-2025",
    "2": "deepseek-chat",
    "3": "qwen-max",
    "4": "glm-4-flash",
    "5": "kimi-k2-0905-preview"
  },
  "seat_providers": {
    "1": "openrouter",
    "2": "deepseek",
    "3": "dashscope",
    "4": "glm",
    "5": "kimi"
  }
}
```

> **Note:** You can also configure seats directly in the web interface!

### Running the Web Interface (Recommended)

**1. Start the Backend API:**
```bash
uvicorn aivalon.services.web_api:app --reload --port 8000
```

**2. Start the Frontend Dev Server:**
```bash
cd frontend
npm run dev
```

**3. Open your browser:**
Navigate to `http://localhost:5173` (or the URL shown by Vite)

**4. Configure and start a game:**
- Click "Configure Seats" to assign AI models to each seat
- Optionally select a seat for yourself to play as a human
- Click "Start Session" and watch the AI agents play!

### Running CLI Simulations (Alternative)

**AI-only game:**
```bash
uv run aivalon
```

**Human player at seat 3:**
```bash
uv run aivalon --human --seat 3
```

**Quick test game:**
```bash
uv run aivalon --max-rounds 2 --seed 12345
```

Generated transcripts land in `runs/YYYYMMDD-HHMMSS_seed.*`

### Testing & Tooling
Install dev extras and run the suite:
```bash
uv sync --dev
uv run pytest
```

## Architecture Overview

### Core Systems

**1. Finite State Machine (FSM)**
- `fsm.py` defines the complete game flow: EVIL_CONCLAVE → PROPOSAL_DRAFT → DISCUSSION → VOTE → MISSION → ASSASSIN_GUESS
- Handles state transitions, vote counting, mission validation, and win condition detection
- Guarantees game never enters invalid states

**2. Custom DSL for Game Context**
- Structured text format optimized for LLM understanding (see `contest/dsl.md`)
- Tags like `@state`, `@mission`, `@speeches` provide compact game state representation
- Saves ~40% tokens compared to natural language or JSON
- Example: `@state R=2 leader=4 team=3 fails=1`

**3. Multi-Provider System**
- `MultiProvider` routes requests to different AI backends per seat
- Unified `ProviderRequest`/`ProviderResponse` interface abstracts API differences
- Supports OpenRouter, DeepSeek, DashScope, GLM, Kimi (easily extensible)
- Enables mixed-model games (e.g., GPT-4 vs DeepSeek vs Qwen)

**4. Tool Calling / Structured Outputs**
- All AI decisions use OpenAI-compatible function calling
- Pydantic schemas (`ProposalPayload`, `VotePayload`, etc.) validate responses
- Retry mechanism: 1st attempt → error feedback → 2nd attempt → fallback to safe defaults
- Prevents game crashes from malformed AI outputs

**5. Web Interface (React + FastAPI)**
- Real-time game visualization via HTTP polling (1s interval)
- FastAPI backend exposes REST endpoints (`/api/session/create`, `/api/session/{id}`)
- Frontend displays live AI speeches, voting animations, mission results
- Seat configuration UI lets users mix models and inject human players

**6. Context Builder**
- Assembles prompts from game state, role knowledge, and speech history
- Filters information based on role (Merlin sees evil seats, Percival sees Merlin/Morgana, etc.)
- Injects game rules, DSL syntax guide, and phase-specific instructions
- Maintains speech clearing logic to prevent context pollution across rounds

**7. Retry & Fallback Mechanisms**
- If AI fails to return valid tool call: retry with error message
- If still fails: use safe defaults (approve votes, success missions, random assassin target)
- Ensures game always progresses even if AI models misbehave

**8. Role System & Information Asymmetry**
- Each seat has `role_knowledge` dict showing visible information
- Good players see nothing, Evil sees each other, Merlin sees Evil (but not Oberon), etc.
- Frontend respects `perspectiveSeat` to hide/reveal information appropriately

## Features & Capabilities

### Current Features
- ✅ **Full Web Interface** – Interactive game visualization with real-time updates
- ✅ **5-Player Games** – Complete implementation with all classic Avalon roles
- ✅ **Multi-Provider Support** – Mix and match models from 5+ providers
- ✅ **Human-AI Hybrid** – Play alongside AI agents
- ✅ **Bilingual UI** – English and Chinese support
- ✅ **DiceBear Avatars** – Unique pixel-art avatars for each seat
- ✅ **Retry & Fallback** – Robust error handling for AI failures
- ✅ **Rich Transcripts** – Full game logs in JSON and Markdown
- ✅ **Model Display** – See which AI model controls each seat

### Roadmap
- **Extended player counts** – 7-player and 10-player variants with additional roles (Oberon, Mordred)
- **More providers** – Anthropic Claude direct API, Azure OpenAI, local models via Ollama
- **WebSocket support** – Replace HTTP polling with WebSocket for lower latency
- **Replay system** – Step-by-step replay of completed games
- **Agent memory** – Persistent cross-game learning and strategy evolution
- **Advanced analytics** – Win rate tracking, model performance comparison, strategy analysis
- **Custom rulesets** – Support for Avalon variants and house rules

## Documentation

### Project Documentation
- **[Frontend README](frontend/README.md)** – Frontend architecture, components, and styling guide
- **[Codemap](documents/codemap.md)** – Comprehensive backend architecture overview
- **[Provider Integration Guide](documents/provider_integration.md)** – How to add new AI providers
- **[Game Rulesets](documents/game_rulesets.md)** – Game rule definitions and variants
- **[Tool Schema Refactoring](documents/tool_schema_refactoring.md)** – LLM tool calling implementation
- **[Frontend/Backend Protocol](documents/frontend_backend_protocol.md)** – Web API specification

### Contest Documentation
- **[Project Overview](contest/contest.md)** – Technical stack and architectural decisions (Chinese)
- **[DSL Guide](contest/dsl.md)** – Game state DSL specification and rationale (Chinese)

## Example Usage

### Web Interface
1. Configure seats in the UI (or use defaults from `config/config.local.json`)
2. Optionally select a seat to play as human
3. Click "Start Session"
4. Watch AI agents negotiate, vote, and complete missions in real-time
5. Download transcript when game completes

### CLI Examples

**Mixed-provider game with 5 different models:**
```bash
# Configure seats in config/config.local.json first
uv run aivalon --seed 12345
```

**Human player competing against AI:**
```bash
uv run aivalon --human --seat 3
```

**Quick test game:**
```bash
uv run aivalon --max-rounds 2
```

## Contributing

### Backend (Python)
- Keep code ASCII by default and explain non-trivial logic with concise comments
- Run the test suite before submitting: `uv run pytest`
- New prompt/agent behaviors should include transcript samples or unit tests
- When adding new providers, follow the [Provider Integration Guide](documents/provider_integration.md)
- Update documentation when making architectural changes

### Frontend (React/TypeScript)
- Follow existing component structure and naming conventions
- Test UI changes in both English and Chinese
- Ensure responsive design works on different screen sizes
- Run frontend tests: `cd frontend && npm run test`
- See [Frontend README](frontend/README.md) for detailed guidelines

### Documentation
- Update relevant `.md` files when changing functionality
- Keep examples up-to-date with current API
- Document new features in README.md highlights

## Tech Stack Summary

**Backend:**
- Python 3.13 + FastAPI + Uvicorn
- Pydantic for data validation
- OpenAI SDK for LLM communication
- Structlog for logging

**Frontend:**
- React 18 + TypeScript + Vite
- SCSS for styling (medieval pixel-art theme)
- i18next for internationalization
- Vitest for testing

**AI Integration:**
- Multi-provider abstraction layer
- Tool calling / function calling for structured outputs
- Retry + fallback mechanisms for reliability

## Support & Contact
- **Issues**: Use GitHub issue tracker for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Logs**: For AI/model issues, attach `runs/*_contexts.md` and `debug_logs/*` excerpts

## License
See LICENSE file for details.
