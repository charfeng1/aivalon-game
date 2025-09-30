# Aivalon Frontend

A Vite + React application providing a web interface for the Avalon simulation engine. Players can launch AI-only matches or join as a human player, monitor game state in real-time, and submit actions through an interactive UI.

## Quick Start

```bash
npm install
npm run dev
```

The dev server runs at `http://localhost:5173` and proxies API calls to `http://localhost:8000`. Override the backend URL by setting `VITE_API_BASE` in a `.env` file.

Ensure the backend is running first:
```bash
uvicorn aivalon.web_api:app --reload
```

---

## Architecture Overview

### Main Components

#### **App.tsx** - Application Shell & Game Controller
The root component that orchestrates the entire game experience.

**Responsibilities:**
- **Session Management**: Create/stop/restart game sessions
- **State Polling**: Fetches game state every 1 second via `/api/sessions/{id}`
- **Phase Logic**: Tracks current phase and round transitions
- **Speech Management**: Controls when speeches appear/clear based on phase changes
- **Action Forms**: Renders modal forms for human player actions (proposals, votes, missions)
- **History Display**: Shows completed missions and proposals in sidebar

**Key State:**
- `sessionId`: Current game session identifier
- `status`: Latest game state from backend
- `pending`: Human action request (if awaiting player input)
- `displaySpeeches`: Filtered speeches shown on screen (cleared on phase transitions)
- `humanMode`: Whether a human player is in the game
- `humanSeat`: Which seat number the human occupies

**Speech Clearing Logic:**
- Clears when: round changes, exiting VOTE phase, entering PROPOSAL_DRAFT, entering EVIL_CONCLAVE
- Uses `clearedSpeechesRef` to prevent speeches from re-appearing after being cleared
- Only restores speeches when new ones arrive from backend

**Mission Display Logic:**
- Only shows missions from current or previous round (`mission.round >= currentRound - 1`)
- Older missions are hidden to avoid stale data

---

#### **NegotiationTable.tsx** - Central Game Display
The main visual component showing players, speeches, proposals, and mission results.

**Layout:**
- **Left Column**: Seats 1-5 (or all seats if ≤5 players)
- **Center**: Current proposal and latest mission result
- **Right Column**: Seats 6-10 (only for 6+ player games)

**Features:**
- **Player Cards**: Avatar (from DiceBear API), seat number, role name (if visible)
- **Speech Bubbles**: Display player speeches with left/right orientation
- **Thinking Animation**: Shows "思考中..." when AI is waiting for API response
- **Proposal Box**: Current leader and team members
- **Mission Result Box**: Shows success/fail with team members (auto-fades after 3 seconds)

**Props:**
- `players`: Array of player info (seat, role, role_name)
- `speeches`: Current speeches to display
- `leader`: Current leader seat number
- `perspectiveSeat`: Seat from which to show role visibility
- `thinkingSeat`: Which seat is currently waiting for API response
- `currentProposal`: Current proposal data (leader, members, approved status)
- `latestMission`: Most recent mission result (or null)

**Avatar System:**
- Uses DiceBear API: `https://api.dicebear.com/7.x/pixel-art/svg?seed={seat}`
- Deterministic based on seat number
- No API key required

---

#### **SeatConfigScreen.tsx** - Game Setup Modal
Pre-game configuration screen for setting up players and AI models.

**Features:**
- **Player Count Selector**: 5-10 players (buttons to increment/decrement)
- **Seat Configuration**: For each seat:
  - Toggle between "You" (human) and "AI"
  - Select AI provider (DeepSeek, OpenRouter, DashScope, GLM, Kimi)
  - Choose model from provider's available models
- **Validation**: Ensures at least one seat is configured
- **Start Game**: Sends configuration to `/api/sessions` endpoint

**Props:**
- `providers`: List of available AI providers and models (fetched from `/api/config/models`)
- `onStart`: Callback with seat configuration
- `onCancel`: Close modal callback

---

#### **LanguageSwitcher.tsx** - Internationalization Toggle
Simple button to switch between Chinese (中文) and English (EN).

**Implementation:**
- Uses `react-i18next` for translations
- Persists language choice to localStorage
- Toggles between `zh-CN` and `en-US`

---

### Styling

#### **pixel-styles.scss** - Main Stylesheet
Global styles with a dark pixel-art theme.

**Key Variables:**
- `--pixel-yellow`: `#ffb300` (primary accent color)
- `--pixel-dark`: `#0f172a` (background)
- `--pixel-radius`: `4px` (border radius)

**Major Sections:**
- `.app-shell`: Main layout container
- `.negotiation-table-container`: Player grid layout
- `.negotiation-player-card`: Player card styling (border, background, transitions)
- `.speech-bubble`: Speech bubble styling (with left/right orientations)
- `.proposal-card` / `.mission-result-card`: Central info displays
- `.history-bubbles`: Mission/proposal history sidebar
- `.action-modal`: Human action form modal

**Theme Colors:**
- Human seats: Green glow (`#48bb78`)
- AI seats: Blue tint (`#60a5fa`)
- Leader: Yellow border (`#ffb300`)
- Mission Success: Green
- Mission Fail: Red

---

#### **SeatConfigScreen.css** - Configuration Modal Styles
Dark theme styling to match the main game UI.

**Key Elements:**
- `.seat-config-modal`: Main modal container (yellow border, dark background)
- `.player-count-controls`: Increment/decrement buttons
- `.seat-card`: Individual seat configuration (green for human, blue for AI)
- `.model-selector`: Dropdown styling for provider and model selection
- `.config-actions`: Start/Cancel button styling

---

### Internationalization (i18n)

#### **locales/en-US.json** - English Translations
All English text used in the UI.

**Key Translation Groups:**
- `app.*`: App title, subtitle, buttons
- `button.*`: Button labels (run, stop, restart, download)
- `metadata.strip.*`: Session info (round, phase, score)
- `player.grid.*`: Player card labels
- `history.*`: History panel text
- `discussion.*`: Discussion panel text
- `action.*`: Action form labels and instructions
- `config.*`: Configuration screen text
- `proposal.*`: Proposal display text
- `mission.*`: Mission result text

---

#### **locales/zh-CN.json** - Chinese Translations
Chinese (Simplified) translations for all UI text.

**Structure:** Mirrors `en-US.json` structure exactly.

---

### Types

#### **types.ts** - TypeScript Type Definitions
Shared type definitions for the frontend.

**Key Types:**
- `SessionStatus`: Complete game state returned from backend
- `SessionState`: Game state (round, phase, scores, players, proposals, missions, speeches)
- `PlayerSummary`: Player info (seat, role, role_name)
- `SpeechEntry`: Speech data (round, seat, speech, kind)
- `PendingRequest`: Human action request (seat, phase, options)
- `ProviderConfig`: AI provider info (name, display_name, models)
- `SeatConfig`: Seat configuration (isHuman, provider, model)

---

## Key Behaviors

### Speech Display Logic
**Location:** `App.tsx` lines 542-605

Speeches are cleared during phase transitions to avoid visual clutter:
1. **Clear on round change**: New round = new discussion
2. **Clear after vote**: Vote → (any phase) transition clears speeches
3. **Clear on new proposal cycle**: Entering PROPOSAL_DRAFT clears old discussion
4. **Clear on evil conclave**: Entering EVIL_CONCLAVE clears previous speeches

**Anti-pattern prevention:**
- Uses `clearedSpeechesRef` to prevent speeches from reappearing after being cleared
- Only restores speeches when `currentRoundSpeeches.length > displaySpeeches.length` (new speeches arrived)

### Mission Display Logic
**Location:** `App.tsx` lines 708-718

Only shows missions from the current or previous round to avoid stale mission results persisting:
```typescript
const lastMission = sessionState.missions[sessionState.missions.length - 1];
if (lastMission.round >= currentRound - 1) {
  return lastMission;
}
return null;
```

### Avatar Loading Delay
**Location:** Backend `web_session.py` line 337-338

Backend waits 1 second before starting game loop to allow DiceBear avatars to load from CDN.

### Thinking Indicator
**Location:** `App.tsx` lines 598-607, `NegotiationTable.tsx` lines 85-90

Shows "思考中..." animation when:
- `thinkingSeat` is set by backend (AI agent waiting for API response)
- Player has no speech bubble yet
- Uses animated dots: ".", "..", "...", "."

---

## API Endpoints

### Backend Communication
All API calls go to `http://localhost:8000` (configurable via `VITE_API_BASE`).

**Endpoints:**
- `GET /api/config/models` - Fetch available AI providers and models
- `POST /api/sessions` - Create new game session with seat configuration
- `GET /api/sessions/{id}` - Fetch current game state (polled every 1 second)
- `POST /api/sessions/{id}/actions` - Submit human player action
- `DELETE /api/sessions/{id}` - Stop/delete session
- `GET /api/sessions/{id}/transcripts/{format}` - Download game transcript

---

## Common Modifications

### Adding a New Phase Display
1. **Update phase label mapping** in `App.tsx` (lines 515-526)
2. **Add translation keys** to `en-US.json` and `zh-CN.json`
3. **Add action form** in `App.tsx` (lines 730-840) if phase requires human input
4. **Update speech clearing logic** if needed (lines 573-589)

### Changing UI Colors
1. **Edit CSS variables** in `pixel-styles.scss` (top of file)
2. **Update seat card colors** in `.negotiation-player-card` classes
3. **Update modal borders** in `.seat-config-modal` and `.action-modal`

### Adding a New AI Provider
1. **Backend:** Register provider in `src/aivalon/config/model_registry.py`
2. **Frontend:** Will automatically appear in `SeatConfigScreen` dropdown (fetched from `/api/config/models`)

### Customizing Speech Behavior
1. **Clearing logic:** `App.tsx` lines 573-589
2. **Display filtering:** `App.tsx` lines 536-540 (`currentRoundSpeeches`)
3. **Bubble styling:** `pixel-styles.scss` `.speech-bubble` class

### Adjusting Mission Result Display
1. **Display duration:** `NegotiationTable.tsx` line 168 (currently 3 seconds)
2. **Auto-fail messages:** `NegotiationTable.tsx` lines 263-264
3. **Fade animation:** `pixel-styles.scss` `.mission-result-text` class

---

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── NegotiationTable.tsx    # Main game display (players, proposals, missions)
│   │   ├── SeatConfigScreen.tsx    # Pre-game setup modal
│   │   └── LanguageSwitcher.tsx    # i18n toggle
│   ├── i18n/
│   │   ├── locales/
│   │   │   ├── en-US.json          # English translations
│   │   │   └── zh-CN.json          # Chinese translations
│   │   └── index.ts                # i18next configuration
│   ├── styles/
│   │   ├── pixel-styles.scss       # Main stylesheet (dark pixel-art theme)
│   │   └── SeatConfigScreen.css    # Config modal styles
│   ├── App.tsx                     # Root component (session management, state polling)
│   ├── types.ts                    # TypeScript definitions
│   └── main.tsx                    # Entry point
├── public/                         # Static assets
├── index.html                      # HTML template
├── package.json                    # Dependencies
├── vite.config.ts                  # Vite configuration
└── README.md                       # This file
```

---

## Troubleshooting

### Speeches Not Clearing
- Check `clearedSpeechesRef` logic in `App.tsx` (lines 594-604)
- Verify phase transitions are triggering correctly
- Backend may still be sending old speeches (check network tab)

### Mission Result Persisting
- Check mission round filtering logic in `App.tsx` (lines 708-718)
- Verify `sessionState.round` is incrementing correctly

### Avatars Not Loading
- DiceBear CDN may be slow/down
- Check browser console for CORS errors
- Backend startup delay (1 second) should handle this

### Configuration Not Saving
- Check provider/model data in `SeatConfigScreen.tsx` (lines 45-70)
- Verify `/api/config/models` returns expected format
- Check browser console for validation errors

---

## Development Tips

1. **Hot Reload:** Vite automatically reloads on file changes
2. **Backend Logs:** Run `uvicorn` with `--reload` for backend hot reload
3. **Network Inspection:** Use browser DevTools to monitor API calls
4. **State Debugging:** Add `console.log(sessionState)` in `App.tsx` useEffect
5. **CSS Changes:** Instant reload without full page refresh

---

## Contributing

When making changes:
1. Update TypeScript types in `types.ts` if adding new data structures
2. Add translation keys to **both** `en-US.json` and `zh-CN.json`
3. Test with both AI-only and human player modes
4. Test with different player counts (5, 7, 10)
5. Verify speech clearing on phase transitions
6. Check mobile responsiveness (media queries in `pixel-styles.scss`)