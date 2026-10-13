import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  AppSettings,
  Conversation,
  RecognitionResult,
  StabilizedResult,
} from "../types";

// ── Settings store ─────────────────────────────────────────────────────────

interface SettingsState {
  settings: AppSettings;
  updateSettings: (patch: Partial<AppSettings>) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      settings: {
        theme: "light",
        language: "en",
        voiceName: "",
        speechRate: 1.0,
        confidenceThreshold: 0.7,
        saveHistoryEnabled: true,
        reducedMotion: false,
      },
      updateSettings: (patch) =>
        set((s) => ({ settings: { ...s.settings, ...patch } })),
    }),
    { name: "signbridge-settings" }
  )
);

// ── Conversation store ─────────────────────────────────────────────────────

interface ConversationState {
  conversations: Conversation[];
  activeSessionMessages: Conversation["messages"];
  setConversations: (c: Conversation[]) => void;
  addMessage: (m: Conversation["messages"][number]) => void;
  clearSession: () => void;
}

export const useConversationStore = create<ConversationState>()((set) => ({
  conversations: [],
  activeSessionMessages: [],
  setConversations: (conversations) => set({ conversations }),
  addMessage: (m) =>
    set((s) => ({ activeSessionMessages: [...s.activeSessionMessages, m] })),
  clearSession: () => set({ activeSessionMessages: [] }),
}));

// ── Recognition store ──────────────────────────────────────────────────────

interface RecognitionState {
  lastResult: RecognitionResult | null;
  stabilized: StabilizedResult | null;
  currentMessage: string;
  isRecognizing: boolean;
  setLastResult: (r: RecognitionResult | null) => void;
  setStabilized: (r: StabilizedResult | null) => void;
  appendToMessage: (char: string) => void;
  backspaceMessage: () => void;
  clearMessage: () => void;
  setIsRecognizing: (v: boolean) => void;
}

export const useRecognitionStore = create<RecognitionState>()((set) => ({
  lastResult: null,
  stabilized: null,
  currentMessage: "",
  isRecognizing: false,
  setLastResult: (lastResult) => set({ lastResult }),
  setStabilized: (stabilized) => set({ stabilized }),
  appendToMessage: (char) =>
    set((s) => ({ currentMessage: s.currentMessage + char })),
  backspaceMessage: () =>
    set((s) => ({ currentMessage: s.currentMessage.slice(0, -1) })),
  clearMessage: () => set({ currentMessage: "" }),
  setIsRecognizing: (isRecognizing) => set({ isRecognizing }),
}));

// ── Analytics store (session-level counters) ───────────────────────────────

interface AnalyticsState {
  attempts: number;
  accepted: number;
  rejected: number;
  latencies: number[];
  incrementAttempt: (accepted: boolean, latencyMs: number) => void;
  resetSession: () => void;
}

export const useAnalyticsStore = create<AnalyticsState>()((set) => ({
  attempts: 0,
  accepted: 0,
  rejected: 0,
  latencies: [],
  incrementAttempt: (accepted, latencyMs) =>
    set((s) => ({
      attempts: s.attempts + 1,
      accepted: accepted ? s.accepted + 1 : s.accepted,
      rejected: accepted ? s.rejected : s.rejected + 1,
      latencies: [...s.latencies, latencyMs],
    })),
  resetSession: () =>
    set({ attempts: 0, accepted: 0, rejected: 0, latencies: [] }),
}));
