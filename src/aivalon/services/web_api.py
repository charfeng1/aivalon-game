"""FastAPI application exposing the Avalon simulation for the frontend."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydantic import ConfigDict

from .web_session import SESSION_MANAGER, GameSession, PendingRequest


class SessionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    """Payload for creating a new game session."""

    human_seat: Optional[int] = Field(None, alias="humanSeat", description="Seat reserved for the human player")
    seat_models: Optional[Dict[str, str]] = Field(None, alias="seatModels", description="Mapping of seat number to model ID")
    seat_providers: Optional[Dict[str, str]] = Field(None, alias="seatProviders", description="Mapping of seat number to provider name")
    seed: Optional[int] = Field(None, description="Deterministic RNG seed")
    max_rounds: Optional[int] = Field(None, alias="maxRounds", description="Optional round cap for quick demos")


class ActionSubmit(BaseModel):
    """Payload carrying the human response for a pending request."""

    request_id: str = Field(..., alias="requestId")
    payload: Dict[str, Any]


app = FastAPI(title="Aivalon Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/config/models")
async def get_available_models() -> Dict[str, Any]:
    """Return available AI providers and their models from centralized registry."""
    from aivalon.config.model_registry import get_all_providers

    providers = get_all_providers()
    return {"providers": providers}


def _serialize_pending(pending: Optional[PendingRequest]) -> Optional[Dict[str, Any]]:
    if pending is None:
        return None
    data = asdict(pending)
    data["createdAt"] = pending.created_at
    data["requestId"] = pending.request_id
    data["stateSnapshot"] = data.get("state_snapshot")
    # Drop dataclass-specific keys for cleaner payloads
    data.pop("created_at", None)
    data.pop("request_id", None)
    data.pop("state_snapshot", None)
    return data


def _serialize_transcript_links(
    session: GameSession,
    metadata: Optional[Dict[str, Dict[str, str]]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not metadata:
        return None
    links: Dict[str, Dict[str, Any]] = {}
    for key, entry in metadata.items():
        filename = entry.get("filename")
        links[key] = {
            "url": f"/api/sessions/{session.session_id}/transcripts/{key}",
            "filename": filename,
            "path": entry.get("path"),
        }
    return links


async def _serialize_session(session: GameSession) -> Dict[str, Any]:
    status = await session.status_async()
    transcript = _serialize_transcript_links(session, status.get("transcript"))
    return {
        "sessionId": status.get("sessionId"),
        "completed": status.get("completed"),
        "error": status.get("error"),
        "state": status.get("state"),
        "pending": _serialize_pending(await session.current_pending_async()),
        "thinkingSeat": status.get("thinkingSeat"),
        "transcript": transcript,
    }


@app.post("/api/sessions")
async def create_session(payload: SessionCreate) -> Dict[str, Any]:
    """Start a new simulation session."""

    session = await SESSION_MANAGER.create_session(
        human_seat=payload.human_seat,
        seat_models=payload.seat_models,
        seat_providers=payload.seat_providers,
        seed=payload.seed,
        max_rounds=payload.max_rounds,
    )
    return {"sessionId": session.session_id}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    """Fetch the latest state for a running session."""

    try:
        session = await SESSION_MANAGER.get(session_id)
    except KeyError as exc:  # pragma: no cover - fastapi transforms into 404
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return await _serialize_session(session)


@app.post("/api/sessions/{session_id}/actions")
async def submit_action(session_id: str, payload: ActionSubmit) -> Dict[str, Any]:
    """Submit a human decision for the currently pending phase."""

    try:
        session = await SESSION_MANAGER.get(session_id)
    except KeyError as exc:  # pragma: no cover - fastapi transforms into 404
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        await session.submit_action_async(payload.request_id, payload.payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"status": "ok"}


@app.delete("/api/sessions/{session_id}")
async def stop_session(session_id: str) -> Dict[str, Any]:
    """Terminate a running simulation session."""

    try:
        session = await SESSION_MANAGER.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    await SESSION_MANAGER.stop(session_id)
    return {"status": "stopped", "sessionId": session.session_id}


@app.get("/api/sessions/{session_id}/transcripts/{kind}")
async def download_transcript(session_id: str, kind: str) -> FileResponse:
    """Stream the requested transcript artifact for a completed session."""

    try:
        session = await SESSION_MANAGER.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    metadata = await session.transcript_metadata_async()
    if not metadata or kind not in metadata:
        raise HTTPException(status_code=404, detail="Transcript not available")

    entry = metadata[kind]
    path_value = entry.get("path")
    if not path_value:
        raise HTTPException(status_code=404, detail="Transcript file missing")

    file_path = Path(path_value)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")

    filename = entry.get("filename") or file_path.name
    return FileResponse(str(file_path), filename=filename)
