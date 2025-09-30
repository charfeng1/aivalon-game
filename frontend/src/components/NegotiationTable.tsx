import React, { useState, useEffect } from "react";
import type { PlayerSummary, SpeechEntry } from "../types";

interface NegotiationTableProps {
  players: PlayerSummary[];
  speeches: SpeechEntry[];
  leader: number;
  perspectiveSeat: number | null;
  hasHumanPerspective: boolean;
  thinkingSeat?: number | null;
  thinkingText?: string | null;
  currentProposal?: { leader: number | null; members: number[]; approved: boolean | null };
  latestMission?: { round: number; members: number[]; fails: number; result: string | null } | null;
  t: (key: string, options?: any) => string;
}

// Generate deterministic avatar URL using DiceBear
function getAvatarUrl(seat: number, sessionId?: string): string {
  // Use seat number and optional sessionId for variety
  const seed = sessionId ? `${sessionId}-${seat}` : `seat-${seat}`;
  return `https://api.dicebear.com/7.x/pixel-art/svg?seed=${encodeURIComponent(seed)}`;
}

// Get the most recent speech for a seat
function getLatestSpeech(seat: number, speeches: SpeechEntry[]): SpeechEntry | null {
  for (let index = speeches.length - 1; index >= 0; index -= 1) {
    const entry = speeches[index];
    if (entry.seat === seat && entry.speech && entry.speech.trim().length > 0) {
      return entry;
    }
  }
  return null;
}

// Display full speech text
function truncateSpeech(raw: string): string {
  return raw.trim();
}

// Strip provider prefix from model name (e.g., "google/gemini-2.5-pro" -> "gemini-2.5-pro")
function getDisplayModelName(modelName: string | null | undefined): string | null {
  if (!modelName) return null;
  const lastSlashIndex = modelName.lastIndexOf('/');
  return lastSlashIndex >= 0 ? modelName.substring(lastSlashIndex + 1) : modelName;
}

function PlayerCard({
  player,
  isLeader,
  isSelf,
  hasKnownRole,
  speechEntry,
  showThinking,
  isThinkingActive,
  thinkingText,
  fallbackThinkingText,
  bubbleDirection,
  t,
}: {
  player: PlayerSummary;
  isLeader: boolean;
  isSelf: boolean;
  hasKnownRole: boolean;
  speechEntry: SpeechEntry | null;
  showThinking: boolean;
  isThinkingActive: boolean;
  thinkingText?: string | null;
  fallbackThinkingText: string;
  bubbleDirection: "left" | "right";
  t: (key: string, options?: any) => string;
}) {
  const avatarUrl = getAvatarUrl(player.seat);

  const cardClasses = [
    "negotiation-player-card",
    isLeader ? "leader" : "",
    isSelf ? "self" : "",
    hasKnownRole ? "known-role" : "",
  ].filter(Boolean).join(" ");

  const roleName = (() => {
    const baseUnknown = t("seat.unknown");
    if (player.role_name) {
      return player.role_name;
    }
    if (player.role) {
      return player.role;
    }
    return baseUnknown;
  })();

  const displayThinking = showThinking && (!speechEntry || !speechEntry.speech);
  const bubbleText = speechEntry
    ? speechEntry.speech
    : displayThinking
      ? (isThinkingActive ? thinkingText ?? fallbackThinkingText : fallbackThinkingText)
      : null;
  const bubbleClasses = [
    "speech-bubble",
    `bubble-${bubbleDirection}`,
  ];

  if (speechEntry?.kind === "SUMMARY") {
    bubbleClasses.push("summary");
  }
  if (displayThinking) {
    bubbleClasses.push("thinking");
  }

  return (
    <div className={`negotiation-seat seat-${player.seat}`}>
      <div className={cardClasses}>
        {/* Avatar */}
        <div className="negotiation-avatar" title={`Seat ${player.seat}`}>
          <img src={avatarUrl} alt={`Seat ${player.seat}`} className="avatar-image" />
          <div className="avatar-seat-number">{player.seat}</div>
        </div>

        {/* Player Info */}
        <div className="negotiation-player-info">
          <div className="seat-label">
            {t("player.grid.seat")}{player.seat}
          </div>
          <div className="role-label">
            {roleName}
          </div>
          {player.model && (
            <div className="model-label" title={player.provider ? `${player.provider}: ${player.model}` : player.model}>
              {getDisplayModelName(player.model)}
            </div>
          )}
        </div>
      </div>

      {/* Speech Bubble */}
      {bubbleText && (
        <div className={bubbleClasses.join(" ")}>
          <div className="bubble-content" title={bubbleText}>
            {truncateSpeech(bubbleText)}
          </div>
          <div className={`bubble-arrow arrow-${bubbleDirection}`}></div>
        </div>
      )}
    </div>
  );
}

export default function NegotiationTable({
  players,
  speeches,
  leader,
  perspectiveSeat,
  hasHumanPerspective,
  thinkingSeat = null,
  thinkingText = null,
  currentProposal,
  latestMission = null,
  t,
}: NegotiationTableProps): JSX.Element {
  // Separate players into left (1-5) and right (6-10) columns
  const leftColumnPlayers = players?.filter(p => p.seat <= 5) ?? [];
  const rightColumnPlayers = players?.filter(p => p.seat > 5) ?? [];
  const showRightColumn = rightColumnPlayers.length > 0;

  // Track mission result visibility for fade-out
  const [showMissionText, setShowMissionText] = useState(true);
  const [missionKey, setMissionKey] = useState<string | null>(null);

  // Reset and start fade timer when mission changes
  useEffect(() => {
    if (latestMission && latestMission.result) {
      const newKey = `${latestMission.round}-${latestMission.result}`;
      if (newKey !== missionKey) {
        setMissionKey(newKey);
        setShowMissionText(true);

        // Fade out after 3 seconds
        const timer = setTimeout(() => {
          setShowMissionText(false);
        }, 3000);

        return () => clearTimeout(timer);
      }
    } else {
      // Reset when no mission
      setMissionKey(null);
      setShowMissionText(true);
    }
  }, [latestMission?.round, latestMission?.result, missionKey]);

  const containerClass = [
    "negotiation-table-container",
    showRightColumn ? "two-columns" : "single-column",
  ].join(" ");

  // Check if there's an active proposal to display
  const hasProposal = currentProposal && currentProposal.members && currentProposal.members.length > 0;

  return (
    <div className={containerClass}>
      {/* Left Column - Seats 1-5 */}
      <div className="negotiation-column left-column">
        {leftColumnPlayers.map((player) => {
          const speechEntry = getLatestSpeech(player.seat, speeches);
          const isLeader = player.seat === leader;
          const isSelf = perspectiveSeat === player.seat;
          const hasKnownRole = Boolean(!hasHumanPerspective && player.role_name && !isSelf);
          const hasSpeech = Boolean(speechEntry && speechEntry.speech && speechEntry.speech.trim().length > 0);
          const isThinkingActive = thinkingSeat === player.seat;
          const showThinking = isThinkingActive && !hasSpeech;

          return (
            <PlayerCard
              key={player.seat}
              player={player}
              isLeader={isLeader}
              isSelf={isSelf}
              hasKnownRole={hasKnownRole}
              speechEntry={speechEntry}
              showThinking={showThinking}
              isThinkingActive={isThinkingActive}
              thinkingText={thinkingText}
              fallbackThinkingText={thinkingText ?? "思考中..."}
              bubbleDirection="right"
              t={t}
            />
          );
        })}
      </div>

      {/* Central Info Display - Proposal and Mission */}
      <div className="central-info-display">
        {/* Current Proposal Box */}
        <div className="proposal-card">
          <h3 className="proposal-title">{t("proposal.current") || "当前提案"}</h3>
          {hasProposal ? (
            <div className="proposal-content">
              <div className="proposal-label">{t("proposal.leader") || "队长"}</div>
              <div className="proposal-leader">座位 {currentProposal.leader}</div>
              <div className="proposal-label">{t("proposal.members") || "队员"}</div>
              <div className="proposal-members">
                {currentProposal.members.sort((a, b) => a - b).map((seat) => (
                  <span key={seat} className="proposal-member-badge">
                    {seat}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <div className="proposal-placeholder">
              {t("proposal.none") || "暂无提案"}
            </div>
          )}
        </div>

        {/* Mission Result Box */}
        <div className={`mission-result-card ${latestMission?.result ? latestMission.result.toLowerCase() : 'pending'}`}>
          <h3 className="mission-result-title">{t("mission.result.title") || "任务结果"}</h3>
          {latestMission && latestMission.result ? (
            <div className="mission-result-content">
              {latestMission.members.length > 0 ? (
                <div className="mission-result-team">
                  <div className="proposal-label">{t("proposal.members") || "队员"}</div>
                  <div className="proposal-members">
                    {latestMission.members.sort((a, b) => a - b).map((seat) => (
                      <span key={seat} className="proposal-member-badge">
                        {seat}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
              <div className={`mission-result-text ${showMissionText ? 'visible' : 'fade-out'}`}>
                <div className="mission-result-status">
                  {latestMission.members.length === 0 && latestMission.result === "FAIL"
                    ? t("mission.result.autofail") || "连续提案失败"
                    : latestMission.result === "SUCCESS"
                      ? t("mission.result.success") || "任务成功"
                      : t("mission.result.fail") || "任务失败"}
                </div>
                {latestMission.members.length > 0 && (
                  <div className="mission-result-details">
                    {t("mission.result.fails", { count: latestMission.fails }) || `${latestMission.fails} 张失败牌`}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="mission-placeholder">
              {t("mission.none") || "暂无任务"}
            </div>
          )}
        </div>
      </div>

      {/* Right Column - Seats 6-10 (only if needed) */}
      {showRightColumn && (
        <div className="negotiation-column right-column">
          {rightColumnPlayers.map((player) => {
            const speechEntry = getLatestSpeech(player.seat, speeches);
            const isLeader = player.seat === leader;
            const isSelf = perspectiveSeat === player.seat;
            const hasKnownRole = Boolean(!hasHumanPerspective && player.role_name && !isSelf);
            const hasSpeech = Boolean(speechEntry && speechEntry.speech && speechEntry.speech.trim().length > 0);
            const isThinkingActive = thinkingSeat === player.seat;
            const showThinking = isThinkingActive && !hasSpeech;

            return (
              <PlayerCard
                key={player.seat}
                player={player}
                isLeader={isLeader}
                isSelf={isSelf}
                hasKnownRole={hasKnownRole}
                speechEntry={speechEntry}
                showThinking={showThinking}
                isThinkingActive={isThinkingActive}
                thinkingText={thinkingText}
                fallbackThinkingText={thinkingText ?? "思考中..."}
                bubbleDirection="left"
                t={t}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
