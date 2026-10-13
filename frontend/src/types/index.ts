// ── Gesture / Recognition ──────────────────────────────────────────────────

export interface GestureVocab {
  id: string;
  label: string;
  description: string;
  handShape: string;
  language: "ASL" | "ISL";
  category: "alphabet" | "numeral" | "phrase";
}

export type RecognitionStatus =
  | "idle"
  | "detecting"
  | "recognized"
  | "confirmed"
  | "no_hand"
  | "low_confidence"
  | "model_unavailable";

export interface RecognitionResult {
  gesture: string;
  confidence: number;
  status: RecognitionStatus;
  timestamp: number;
}

export interface StabilizedResult {
  gesture: string;
  confidence: number;
  confirmed: boolean;
  timestamp: number;
}

// ── Conversation / Message ─────────────────────────────────────────────────

export type MessageSource = "sign_recognition" | "speech_recognition" | "manual";

export interface Message {
  id: string;
  content: string;
  source: MessageSource;
  language: string;
  confidence?: number;
  timestamp: string;
}

export interface Conversation {
  id: string;
  createdAt: string;
  updatedAt: string;
  messages: Message[];
  sessionId: string;
}

// ── Analytics ─────────────────────────────────────────────────────────────

export interface RecognitionEvent {
  id: string;
  gesture: string;
  confidence: number;
  accepted: boolean;
  latencyMs: number;
  timestamp: string;
}

export interface AnalyticsSummary {
  totalAttempts: number;
  accepted: number;
  rejected: number;
  avgLatencyMs: number;
  gestureDistribution: Record<string, number>;
  practiceCompleted: number;
}

// ── API responses ──────────────────────────────────────────────────────────

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  timestamp: string;
}

export interface ModelStatusResponse {
  loaded: boolean;
  modelType: string | null;
  classLabels: string[];
  trainedAt: string | null;
  message: string;
}

export interface ApiError {
  detail: string;
  code?: string;
}

// ── Settings ───────────────────────────────────────────────────────────────

export interface AppSettings {
  theme: "light" | "dark" | "system";
  language: "en" | "ta";
  voiceName: string;
  speechRate: number;
  confidenceThreshold: number;
  saveHistoryEnabled: boolean;
  reducedMotion: boolean;
}


// ── Phase 3: Hand landmarks ─────────────────────────────────────────────────

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export interface HandLandmarks {
  handedness: string | null;
  landmarks: LandmarkPoint[];
}

export interface LandmarkResponse {
  hands_detected: number;
  hands: HandLandmarks[];
  status: "hand_detected" | "no_hand" | "error";
  message: string;
  latency_ms: number;
}

// ── LSTM word-sign recognition (additional method) ──────────────────────────

export interface SequenceResponse {
  status: "collecting" | "no_hand" | "low_confidence" | "recognized" | "model_unavailable" | "error";
  word: string | null;
  /** Probability of the top word; 0 when the model did not run. */
  confidence: number;
  /** Set on the single frame where a word is confirmed (append it once). */
  confirmed_word: string | null;
  probabilities: Record<string, number> | null;
  frames_buffered: number;
  sequence_length: number;
  hands_in_window: number;
  window_seconds: number | null;
  message: string;
  session_id: string;
  latency_ms: number;
  method: "lstm";
}
