export interface PlayerSummary {
  seat: number;
  role: string | null;
  role_name?: string | null;
  model?: string | null;
  provider?: string | null;
}

export interface ProposalSummary {
  round: number;
  leader: number;
  members: number[];
  approved: boolean | null;
}

export interface MissionSummary {
  round: number;
  members: number[];
  fails: number;
  result: string | null;
}

export interface VoteSummary {
  round: number;
  leader?: number | null;
  votes: Array<{ seat: number; value: string }>;
}

export type SpeechKind = "DISCUSSION" | "SUMMARY" | "EVIL_CONCLAVE";

export interface SpeechEntry {
  round: number;
  seat: number;
  speech: string;
  isYou?: boolean;
  kind?: SpeechKind;
}

export interface TranscriptLink {
  url: string;
  filename?: string;
  path?: string;
}

export interface SessionTranscript {
  json?: TranscriptLink;
  markdown?: TranscriptLink;
  context?: TranscriptLink;
}

export interface SessionState {
  round: number;
  phase: string;
  leader: number;
  teamSize: number;
  scores: { good: number; evil: number };
  failedProposals: number;
  currentProposal: { leader: number | null; members: number[]; approved: boolean | null };
  proposals: ProposalSummary[];
  missions: MissionSummary[];
  votes: VoteSummary[];
  speeches: SpeechEntry[];
  players: PlayerSummary[];
  winner: string | null;
  privateCard?: string | null;
}

export interface PendingOptions {
  phase: string;
  teamSize: number;
  availableSeats: number[];
  currentMembers: number[];
  missionMembers: number[];
  onMission: boolean;
  canFailMission: boolean;
}

export interface PendingRequest {
  requestId: string;
  seat: number;
  phase: string;
  instructions: string;
  options: PendingOptions;
  stateSnapshot: SessionState;
  createdAt: number;
}

export interface SessionStatus {
  sessionId: string;
  completed: boolean;
  error?: string | null;
  state: SessionState | null;
  pending: PendingRequest | null;
  thinkingSeat?: number | null;
  transcript?: SessionTranscript | null;
}

export interface ProviderConfig {
  name: string;
  display_name: string;
  models: string[];
}

export interface SeatConfig {
  isHuman: boolean;
  model?: string;
  provider?: string;
}

export interface SessionCreateRequest {
  humanSeat: number;
  seatModels: Record<string, string>;
  seatProviders: Record<string, string>;
  seed?: number | null;
  maxRounds?: number | null;
}
