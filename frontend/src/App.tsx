import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import type {
  PlayerSummary,
  SessionState,
  SessionStatus,
  SessionTranscript,
  SpeechEntry,
  TranscriptLink,
  ProviderConfig,
  SeatConfig,
} from "./types";
import LanguageSwitcher from "./components/LanguageSwitcher";
import NegotiationTable from "./components/NegotiationTable";
import SeatConfigScreen from "./components/SeatConfigScreen";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

type ActionState = {
  selectedMembers?: number[];
  thinking?: string;
  speech?: string;
  vote?: "APPROVE" | "REJECT";
  reflection?: string;
  missionValue?: "SUCCESS" | "FAIL";
  assassinTarget?: number;
};

function sortMembers(members: number[]): number[] {
  return [...members].sort((a, b) => a - b);
}

function getHumanInstructions(phase: string, t: Function): string {
  switch (phase) {
    case "DISCUSSION":
    case "PROPOSAL_SUMMARY":
    case "EVIL_CONCLAVE":
      return t("action.instruction.discussion");
    case "VOTE":
      return t("action.instruction.vote");
    case "MISSION":
      return t("action.instruction.mission");
    case "ASSASSIN_GUESS":
      return t("action.instruction.assassin");
    case "PROPOSAL_DRAFT":
    case "PROPOSAL_FINAL":
      return t("action.instruction.proposal");
    default:
      return "";
  }
}


function getMissionStatusClass(result?: string | null): "success" | "failure" | "pending" {
  if (!result) {
    return "pending";
  }
  if (result === "SUCCESS") {
    return "success";
  }
  if (result === "FAIL") {
    return "failure";
  }
  return "pending";
}

function getProposalStatusClass(approved?: boolean | null): "success" | "failure" | "pending" {
  if (approved === true) {
    return "success";
  }
  if (approved === false) {
    return "failure";
  }
  return "pending";
}

function renderHistory(state: SessionState | null, t: Function): JSX.Element | null {
  if (!state) {
    return null;
  }

  // Helper function to get votes for a specific proposal (round + leader)
  const getVotesForProposal = (round: number, leader: number) => {
    const voteSummary = state.votes?.find(v => v.round === round && v.leader === leader);
    return voteSummary ? voteSummary.votes : [];
  };

  return (
    <div className="history-bubbles">
      <section className="history-column missions-column">
        <h3>{t("history.panels.missions")}</h3>
        {(state.missions?.length ?? 0) === 0 ? (
          <p className="muted">{t("history.panels.no.missions")}</p>
        ) : (
          <ul className="history-bubble-list">
            {state.missions?.map((mission) => {
              const status = getMissionStatusClass(mission.result);
              const resultKey = mission.result === "SUCCESS"
                ? "history.mission.result.success"
                : "history.mission.result.fail";
              const resultText = mission.result ? t(resultKey) : t("history.proposal.status.pending");
              const isAutoFail = mission.members.length === 0 && mission.result === "FAIL";

              return (
                <li key={`mission-${mission.round}`} className="history-bubble-item">
                  <div className={`history-token mission ${status}`} aria-hidden="true">
                    M{mission.round}
                  </div>
                  <div className="history-bubble bubble-right">
                    <div className="bubble-body">
                      <div className="bubble-text">
                        {isAutoFail ? (
                          language === "zh-CN"
                            ? `第${mission.round}轮任务因连续提案失败而自动失败`
                            : `Round ${mission.round} mission auto-failed due to consecutive proposal rejections`
                        ) : (
                          t("history.mission.natural", {
                            round: mission.round,
                            members: mission.members.join(", "),
                            result: resultText,
                            fails: mission.fails
                          })
                        )}
                      </div>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>
      <section className="history-column proposals-column">
        <h3>{t("history.panels.proposals")}</h3>
        {(state.proposals?.length ?? 0) === 0 ? (
          <p className="muted">{t("history.panels.no.proposals")}</p>
        ) : (
          <ul className="history-bubble-list">
            {state.proposals?.map((proposal) => {
              const status = getProposalStatusClass(proposal.approved);
              const votes = getVotesForProposal(proposal.round, proposal.leader);

              const statusKey = typeof proposal.approved === "boolean"
                ? (proposal.approved
                    ? "history.proposal.status.approved"
                    : "history.proposal.status.rejected")
                : "history.proposal.status.pending";
              const statusText = t(statusKey);
              const membersText = proposal.members.join(", ");

              return (
                <li key={`proposal-${proposal.round}-${proposal.leader}`} className="history-bubble-item proposal-item">
                  <div className="history-bubble bubble-left">
                    <div className="bubble-body">
                      <div className="bubble-text">
                        {t("history.proposal.natural", {
                          round: proposal.round,
                          leader: proposal.leader,
                          members: membersText,
                          status: statusText
                        })}
                      </div>
                      <div className="bubble-meta votes-meta">
                        {t("history.panels.proposal.votes")}:{" "}
                        {votes.length > 0 ? (
                          votes.map((v, idx) => (
                            <span key={v.seat}>
                              {idx > 0 && " · "}
                              {v.seat}:
                              <span className={`vote-icon ${v.value === "APPROVE" ? "approve" : "reject"}`}>
                                {v.value === "APPROVE" ? "✓" : "✗"}
                              </span>
                            </span>
                          ))
                        ) : (
                          t("history.panels.proposal.noVotes")
                        )}
                      </div>
                    </div>
                  </div>
                  <div className={`history-token proposal ${status}`} aria-hidden="true">
                    P{proposal.round}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
}


function App(): JSX.Element {
  const { t, i18n } = useTranslation();
  const language = i18n.language;

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<SessionStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionState, setActionState] = useState<ActionState>({});
  const [submitting, setSubmitting] = useState(false);
  const [humanMode, setHumanMode] = useState(false);
  const [humanSeat, setHumanSeat] = useState<number | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [showConfigScreen, setShowConfigScreen] = useState(false);
  const [providers, setProviders] = useState<ProviderConfig[]>([]);

  const pending = status?.pending ?? null;
  const serverThinkingSeat = status?.thinkingSeat ?? null; // Get the thinking seat from server

  // Fetch available providers and models on mount
  useEffect(() => {
    const fetchProviders = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/config/models`);
        if (!response.ok) {
          throw new Error(`Failed to fetch providers: ${response.status}`);
        }
        const data = await response.json();
        setProviders(data.providers || []);
      } catch (err) {
        console.error("Failed to fetch providers:", err);
        setError(t("error.banner") + " Could not load providers");
      }
    };
    fetchProviders();
  }, [t]);

  useEffect(() => {
    if (!pending) {
      setActionState({});
      return;
    }
    const defaults: ActionState = {};
    if (pending.phase === "PROPOSAL_DRAFT" || pending.phase === "PROPOSAL_FINAL") {
      defaults.selectedMembers = pending.options.currentMembers.length
        ? sortMembers(pending.options.currentMembers)
        : sortMembers(pending.options.availableSeats.slice(0, pending.options.teamSize));
    } else if (pending.phase === "VOTE") {
      defaults.vote = "APPROVE";
      defaults.reflection = "";
    } else if (pending.phase === "MISSION") {
      defaults.missionValue = pending.options.canFailMission ? "FAIL" : "SUCCESS";
    } else if (pending.phase === "ASSASSIN_GUESS") {
      defaults.assassinTarget = pending.options.availableSeats[0];
    }
    defaults.thinking = "";
    defaults.speech = "";
    setActionState(defaults);
  }, [pending?.requestId]);

  const fetchStatus = useCallback(async () => {
    if (!sessionId) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch session: ${response.status}`);
      }
      const data: SessionStatus = await response.json();
      setStatus(data);
      setError(null);
    } catch (err) {
      console.error(err);
      setError(t("error.banner"));
    }
  }, [sessionId, t]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, [sessionId, fetchStatus]);

  const startSession = useCallback(
    async (seatConfig: Record<number, SeatConfig>) => {
      setLoading(true);
      setStatus(null);
      setSessionId(null);
      setShowConfigScreen(false);

      // Extract human seat and build seat_models/seat_providers
      let humanSeatNum: number | null = null;
      const seatModels: Record<string, string> = {};
      const seatProviders: Record<string, string> = {};

      // Get all seat numbers from config
      const seatNumbers = Object.keys(seatConfig).map(Number).sort((a, b) => a - b);

      for (const seat of seatNumbers) {
        const config = seatConfig[seat];
        if (config.isHuman) {
          humanSeatNum = seat;
        } else if (config.model && config.provider) {
          seatModels[seat.toString()] = config.model;
          seatProviders[seat.toString()] = config.provider;
        }
      }

      setHumanMode(humanSeatNum !== null);
      setHumanSeat(humanSeatNum);

      try {
        const response = await fetch(`${API_BASE}/api/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            humanSeat: humanSeatNum,
            seatModels,
            seatProviders,
          }),
        });
        if (!response.ok) {
          throw new Error(`Failed to create session: ${response.status}`);
        }
        const data = await response.json();
        setSessionId(data.sessionId);
        setError(null);
      } catch (err) {
        console.error(err);
        setError(t("error.banner"));
      } finally {
        setLoading(false);
      }
    },
    [t]
  );

  const stopSession = useCallback(async () => {
    if (!sessionId) {
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(`Failed to stop session: ${response.status}`);
      }
      setStatus(null);
      setSessionId(null);
      setActionState({});
      setError(null);
    } catch (err) {
      console.error(err);
      setError(t("error.banner"));
    } finally {
      setLoading(false);
    }
  }, [sessionId, t]);

  const restartSession = useCallback(async () => {
    if (!sessionId) {
      setShowConfigScreen(true);
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(`Failed to stop session: ${response.status}`);
      }
      setShowConfigScreen(true);
    } catch (err) {
      console.error(err);
      setError(t("error.banner"));
    } finally {
      setLoading(false);
    }
  }, [sessionId, t]);

  const toggleMember = (seat: number) => {
    if (!pending) {
      return;
    }
    setActionState((prev) => {
      const current = prev.selectedMembers ?? [];
      let next: number[];
      if (current.includes(seat)) {
        next = current.filter((value) => value !== seat);
      } else {
        next = sortMembers([...current, seat]);
      }
      if (next.length > pending.options.teamSize) {
        next = next.slice(0, pending.options.teamSize);
      }
      return { ...prev, selectedMembers: next };
    });
  };

  const submitAction = async (event: FormEvent) => {
    event.preventDefault();
    if (!sessionId || !pending) {
      return;
    }

    let payload: Record<string, unknown> = {};
    if (pending.phase === "PROPOSAL_DRAFT" || pending.phase === "PROPOSAL_FINAL") {
      const members = actionState.selectedMembers ?? [];
      if (members.length !== pending.options.teamSize) {
        setError(t("action.form.select.members") + pending.options.teamSize + t("action.form.members"));
        return;
      }
      payload = {
        action: {
          type: "PROPOSE",
          members,
        },
      };
    } else if (pending.phase === "DISCUSSION" || pending.phase === "PROPOSAL_SUMMARY") {
      payload = {
        thinking: actionState.thinking ?? "",
        speech: actionState.speech ?? "",
      };
    } else if (pending.phase === "VOTE") {
      if (!actionState.vote) {
        setError(t("error.selectVote"));
        return;
      }
      payload = {
        action: {
          type: "VOTE",
          value: actionState.vote,
          reflection: actionState.reflection ?? "",
        },
      };
    } else if (pending.phase === "MISSION") {
      const value = actionState.missionValue ?? "SUCCESS";
      payload = {
        action: {
          type: "MISSION",
          value,
        },
      };
    } else if (pending.phase === "EVIL_CONCLAVE") {
      payload = {
        thinking: actionState.thinking ?? "",
        speech: actionState.speech ?? "",
      };
    } else if (pending.phase === "ASSASSIN_GUESS") {
      payload = {
        action: {
          type: "ASSASSIN_GUESS",
          target: actionState.assassinTarget ?? 1,
        },
      };
    }

    try {
      setSubmitting(true);
      const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/actions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requestId: pending.requestId, payload }),
      });
      if (!response.ok) {
        throw new Error(`Failed to submit action: ${response.status}`);
      }
      setError(null);
      setSubmitting(false);
      await fetchStatus();
    } catch (err) {
      console.error(err);
      setSubmitting(false);
      setError(t("error.banner")); // Using generic error message
    }
  };

  const sessionState = status?.state ?? null;
  const isComplete = status?.completed ?? false;
  const transcriptLinks: SessionTranscript | null = status?.transcript ?? null;

  const downloadTranscript = useCallback(
    async (format: keyof SessionTranscript = "json") => {
      if (!sessionId) {
        return;
      }
      const target: TranscriptLink | undefined = transcriptLinks?.[format];
      if (!target?.url) {
        return;
      }
      try {
        setDownloading(true);
        const requestUrl = target.url.startsWith("http") ? target.url : `${API_BASE}${target.url}`;
        const response = await fetch(requestUrl);
        if (!response.ok) {
          throw new Error(`Failed to download transcript: ${response.status}`);
        }
        const blob = await response.blob();
        const objectUrl = window.URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = objectUrl;
        const extension = format === "markdown" ? ".md" : ".json";
        anchor.download = target.filename ?? `${sessionId}-${format}${extension}`;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        window.URL.revokeObjectURL(objectUrl);
      } catch (err) {
        console.error(err);
        setError(t("error.banner"));
      } finally {
        setDownloading(false);
      }
    },
    [sessionId, transcriptLinks, t]
  );

  const phaseLabel = useMemo(() => {
    if (!sessionState) {
      return "—";
    }
    const mapping: Record<string, string> = {
      PROPOSAL_DRAFT: t("action.form.proposalDraft"),
      PROPOSAL_FINAL: t("action.form.proposalFinal"),
      PROPOSAL_SUMMARY: t("action.form.proposalSummary"),
      DISCUSSION: t("action.form.discussion"),
      VOTE: t("action.form.vote"),
      MISSION: t("action.form.mission"),
      EVIL_CONCLAVE: t("action.form.evilConclave"),
      ASSASSIN_GUESS: t("action.form.assassinGuess"),
    };
    return mapping[sessionState.phase] ?? sessionState.phase;
  }, [sessionState, t]);

  const currentRoundSpeeches = useMemo(() => {
    if (!sessionState) {
      return [] as SpeechEntry[];
    }
    const round = sessionState.round;
    return (sessionState.speeches ?? []).filter((entry) => entry.round === round);
  }, [sessionState?.speeches, sessionState?.round]);

  const [displaySpeeches, setDisplaySpeeches] = useState<SpeechEntry[]>([]);
  const lastRoundRef = useRef<number | null>(null);
  const lastPhaseRef = useRef<string | null>(null);
  const clearedSpeechesRef = useRef<boolean>(false);
  const [showRoundNotification, setShowRoundNotification] = useState(false);

  useEffect(() => {
    if (!sessionState) {
      lastRoundRef.current = null;
      lastPhaseRef.current = null;
      clearedSpeechesRef.current = false;
      setDisplaySpeeches([]);
      setShowRoundNotification(false);
      return;
    }
    const round = sessionState.round;
    const phase = sessionState.phase;

    // Show round change notification
    if (lastRoundRef.current !== round && lastRoundRef.current !== null) {
      setShowRoundNotification(true);
      setTimeout(() => setShowRoundNotification(false), 2000);
    }

    // Clear speeches when round changes
    if (lastRoundRef.current !== round) {
      lastRoundRef.current = round;
      clearedSpeechesRef.current = false;
      setDisplaySpeeches([]);
    }

    // Determine if we should clear speeches based on phase transitions
    let shouldClearSpeeches = false;

    // Clear speeches when transitioning FROM vote phase (after any vote result)
    if (lastPhaseRef.current === "VOTE" && phase !== "VOTE") {
      shouldClearSpeeches = true;
    }

    // Clear speeches when entering EVIL_CONCLAVE phase
    if (phase === "EVIL_CONCLAVE" && lastPhaseRef.current !== "EVIL_CONCLAVE") {
      shouldClearSpeeches = true;
    }

    // Clear speeches when entering PROPOSAL_DRAFT (new proposal cycle)
    if (phase === "PROPOSAL_DRAFT" && lastPhaseRef.current !== "PROPOSAL_DRAFT") {
      shouldClearSpeeches = true;
    }

    lastPhaseRef.current = phase;

    // Update speeches
    if (shouldClearSpeeches) {
      setDisplaySpeeches([]);
      clearedSpeechesRef.current = true;
    } else if (!clearedSpeechesRef.current) {
      // Only restore speeches if we haven't manually cleared them
      setDisplaySpeeches(currentRoundSpeeches);
    } else if (currentRoundSpeeches.length > displaySpeeches.length) {
      // New speeches arrived after clearing - allow them
      setDisplaySpeeches(currentRoundSpeeches);
      clearedSpeechesRef.current = false;
    }
  }, [sessionState?.round, sessionState?.phase, currentRoundSpeeches, displaySpeeches.length]);

  const [thinkingTick, setThinkingTick] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => {
      setThinkingTick((prev) => (prev + 1) % 4);
    }, 600);
    return () => window.clearInterval(id);
  }, []);

  const pendingSeat = pending?.seat ?? null;
  const dotCycle = [".", "..", "...", "."];
  const baseThinkingText = `思考中${dotCycle[thinkingTick]}`;
  const thinkingPhases = useMemo(() => new Set(["DISCUSSION", "PROPOSAL_SUMMARY", "EVIL_CONCLAVE"]), []);
  const shouldHighlightThinking = Boolean(pending && thinkingPhases.has(pending.phase));
  
  // Use the server's thinking seat info for the animation (when an AI agent is waiting for API response)
  // Otherwise, fallback to the pending seat (for human player actions)
  const thinkingSeat = serverThinkingSeat || (shouldHighlightThinking ? pendingSeat : null);

  return (
    <div className="app-shell">
      <div style={{ position: 'fixed', top: '1rem', right: '1rem', zIndex: 1000 }}>
        <LanguageSwitcher />
      </div>
      <header className="app-header">
        <h1>{t("app.title")}</h1>
        <p className="subtitle">{t("subtitle")}</p>
        <div className="header-actions">
          {!sessionId && (
            <button
              type="button"
              onClick={() => setShowConfigScreen(true)}
              disabled={loading || providers.length === 0}
            >
              {t("button.configureGame")}
            </button>
          )}
        </div>
        {error && <p className="error-banner">{t("error.banner")} {error}</p>}
        {sessionId && (
          <div className="metadata-strip">
            <span>{t("metadata.strip.session")}{sessionId}</span>
            <span>{t("metadata.strip.round")}{sessionState?.round ?? "—"}</span>
            <span>{t("metadata.strip.phase")}{phaseLabel}</span>
            <span>
              {t("metadata.strip.score")}{sessionState?.scores?.good ?? 0}{t("metadata.strip.score.seperator")}{sessionState?.scores?.evil ?? 0}
            </span>
            {isComplete && <span className="badge">{t("metadata.strip.completed")}</span>}
          </div>
        )}
        {sessionId && (
          <div className="session-controls">
            {!isComplete && (
              <button type="button" className="ghost" onClick={stopSession} disabled={loading}>
                {t("button.stop")}
              </button>
            )}
            <button type="button" className="ghost" onClick={restartSession} disabled={loading}>
              {t("button.restart")}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => downloadTranscript("json")}
              disabled={!isComplete || !transcriptLinks?.json || downloading}
            >
              {downloading ? t("button.downloading") : t("button.downloadTranscript")}
            </button>
          </div>
        )}
      </header>
      <main className="app-main">
        {!sessionId && <p className="muted">{t("app.main.start.session")}</p>}
        {sessionId && !sessionState?.players?.length && (
          <p className="muted" style={{ textAlign: 'center', padding: '2rem' }}>
            {language === "zh-CN" ? "正在初始化游戏..." : "Initializing game..."}
          </p>
        )}
        {sessionState && sessionState.players?.length > 0 && (
          <>
            {isComplete && sessionState.winner && (
              <div className={`game-result-banner ${sessionState.winner.toLowerCase()}`}>
                <div className="result-content">
                  <h2>
                    {sessionState.winner === "GOOD"
                      ? (language === "zh-CN" ? "善良阵营获胜！" : "Good Team Wins!")
                      : (language === "zh-CN" ? "邪恶阵营获胜！" : "Evil Team Wins!")}
                  </h2>
                  <p className="final-score">
                    {t("metadata.strip.score")}{sessionState.scores?.good ?? 0}{t("metadata.strip.score.seperator")}{sessionState.scores?.evil ?? 0}
                  </p>
                </div>
              </div>
            )}

            {sessionState.privateCard && (
              <aside className="private-card">
                <h3>{t("your.role")}</h3>
                <p>{sessionState.privateCard}</p>
              </aside>
            )}
            <div className="negotiation-wrapper">
              <NegotiationTable
                players={sessionState.players}
                speeches={displaySpeeches}
                leader={sessionState.leader}
                perspectiveSeat={pending ? pending.seat : humanSeat}
                hasHumanPerspective={humanMode}
                thinkingSeat={thinkingSeat}
                thinkingText={baseThinkingText}
                currentProposal={sessionState.currentProposal}
                latestMission={(() => {
                  // Only show mission from current or previous round
                  if (!sessionState.missions || sessionState.missions.length === 0) return null;
                  const lastMission = sessionState.missions[sessionState.missions.length - 1];
                  const currentRound = sessionState.round;
                  // Show mission if it's from current round or previous round (during transition)
                  if (lastMission.round >= currentRound - 1) {
                    return lastMission;
                  }
                  return null;
                })()}
                t={t}
              />
              {showRoundNotification && (
                <div className="round-notification">
                  <div className="round-notification-content">
                    {language === "zh-CN"
                      ? `第${sessionState.round}轮 · ${phaseLabel} · 需要${sessionState.teamSize}人团队`
                      : `Round ${sessionState.round} · ${phaseLabel} · Team of ${sessionState.teamSize} needed`
                    }
                  </div>
                </div>
              )}
            </div>
            {renderHistory(sessionState, t)}
          </>
        )}

        {showConfigScreen && providers.length > 0 && (
          <SeatConfigScreen
            providers={providers}
            onStart={startSession}
            onCancel={() => setShowConfigScreen(false)}
            t={t}
          />
        )}

        {pending && humanMode && (
          <div className="modal-backdrop">
            <section className="action-modal">
              <div className="action-header">
                <h3>{phaseLabel}</h3>
              </div>
              <form onSubmit={submitAction} className="action-form">
              {(pending.phase === "PROPOSAL_DRAFT" || pending.phase === "PROPOSAL_FINAL") && (
                <div className="form-group compact">
                  <label className="compact-label">{t("action.form.select.members")}{pending.options.teamSize}:</label>
                  <div className="seat-selector">
                    {pending.options.availableSeats.map((seat) => {
                      const selected = (actionState.selectedMembers ?? []).includes(seat);
                      return (
                        <button
                          type="button"
                          key={seat}
                          className={selected ? "selected" : ""}
                          onClick={() => toggleMember(seat)}
                        >
                          {seat}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {(pending.phase === "DISCUSSION" || pending.phase === "PROPOSAL_SUMMARY" || pending.phase === "EVIL_CONCLAVE") && (
                <div className="form-group compact">
                  <textarea
                    placeholder={t("action.form.speech.placeholder")}
                    value={actionState.speech ?? ""}
                    onChange={(event) => setActionState((prev) => ({ ...prev, speech: event.target.value }))}
                    maxLength={300}
                    rows={3}
                  />
                </div>
              )}

              {pending.phase === "VOTE" && (
                <div className="form-group compact">
                  <label className="compact-label">{t("action.form.team")}: [{(pending.options.currentMembers ?? []).join(", ")}]</label>
                  <div className="vote-buttons">
                    <button
                      type="button"
                      className={`vote-btn ${actionState.vote === "APPROVE" ? "selected approve" : ""}`}
                      onClick={() => setActionState((prev) => ({ ...prev, vote: "APPROVE" }))}
                    >
                      {t("action.form.vote.approve")}
                    </button>
                    <button
                      type="button"
                      className={`vote-btn ${actionState.vote === "REJECT" ? "selected reject" : ""}`}
                      onClick={() => setActionState((prev) => ({ ...prev, vote: "REJECT" }))}
                    >
                      {t("action.form.vote.reject")}
                    </button>
                  </div>
                </div>
              )}

              {pending.phase === "MISSION" && (
                <div className="form-group compact">
                  <div className="vote-buttons">
                    <button
                      type="button"
                      className={`vote-btn ${actionState.missionValue === "SUCCESS" ? "selected approve" : ""}`}
                      onClick={() => setActionState((prev) => ({ ...prev, missionValue: "SUCCESS" }))}
                    >
                      {t("action.form.mission.success")}
                    </button>
                    <button
                      type="button"
                      disabled={!pending.options.canFailMission}
                      className={`vote-btn ${actionState.missionValue === "FAIL" ? "selected reject" : ""}`}
                      onClick={() =>
                        pending.options.canFailMission &&
                        setActionState((prev) => ({ ...prev, missionValue: "FAIL" }))
                      }
                    >
                      {t("action.form.mission.fail")}
                    </button>
                  </div>
                </div>
              )}

              {pending.phase === "ASSASSIN_GUESS" && (
                <div className="form-group compact">
                  <label className="compact-label">{t("action.form.assassin.target")}</label>
                  <select
                    value={actionState.assassinTarget ?? pending.options.availableSeats[0]}
                    onChange={(event) =>
                      setActionState((prev) => ({ ...prev, assassinTarget: Number(event.target.value) }))
                    }
                  >
                    {pending.options.availableSeats.map((seat) => (
                      <option key={seat} value={seat}>
                        {t("config.seat")} {seat}
                      </option>
                    ))}
                  </select>
                </div>
              )}

                <div className="action-footer">
                  <button type="submit" disabled={submitting} className="submit-btn">
                    {submitting ? t("action.form.submitting") : t("action.form.submit")}
                  </button>
                </div>
              </form>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
