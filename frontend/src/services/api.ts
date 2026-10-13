import type {
  HealthResponse,
  ModelStatusResponse,
  GestureVocab,
  RecognitionResult,
  LandmarkResponse,
  SequenceResponse,
  Conversation,
  AnalyticsSummary,
} from "../types";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// ── Generic fetch helper ───────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Health ─────────────────────────────────────────────────────────────────

export const getHealth = () =>
  apiFetch<HealthResponse>("/api/health");

/**
 * Lightweight backend reachability probe for diagnostics.
 * Returns true when /api/health responds OK within the timeout, false otherwise.
 * Never throws.
 */
export async function checkBackend(timeoutMs = 3000): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${BASE_URL}/api/health`, { signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

// ── Model ──────────────────────────────────────────────────────────────────

export const getModelStatus = () =>
  apiFetch<ModelStatusResponse>("/api/model/status");

// ── Gestures ───────────────────────────────────────────────────────────────

export const getGestures = () =>
  apiFetch<GestureVocab[]>("/api/gestures");

// ── Recognition ───────────────────────────────────────────────────────────

export async function recognizeFrame(
  frameBlob: Blob
): Promise<RecognitionResult> {
  const form = new FormData();
  form.append("frame", frameBlob, "frame.jpg");
  const res = await fetch(`${BASE_URL}/api/recognize`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<RecognitionResult>;
}

// ── Conversations ─────────────────────────────────────────────────────────

export const getConversations = () =>
  apiFetch<Conversation[]>("/api/conversations");

export const saveConversation = (payload: Omit<Conversation, "id" | "createdAt" | "updatedAt">) =>
  apiFetch<Conversation>("/api/conversations", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const deleteConversation = (id: string) =>
  apiFetch<void>(`/api/conversations/${id}`, { method: "DELETE" });

// ── Analytics ─────────────────────────────────────────────────────────────

export const getAnalytics = () =>
  apiFetch<AnalyticsSummary>("/api/analytics");


export async function recognizeLandmarks(
  frameBlob: Blob
): Promise<LandmarkResponse> {
  const form = new FormData();
  form.append("frame", frameBlob, "frame.jpg");
  const res = await fetch(`${BASE_URL}/api/recognize/landmarks`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<LandmarkResponse>;
}

// ── LSTM word-sign recognition (additional method) ──────────────────────────

/** Feed one frame into the backend's rolling 30-frame LSTM window for `sessionId`. */
export async function recognizeSequence(
  frameBlob: Blob,
  sessionId: string
): Promise<SequenceResponse> {
  const form = new FormData();
  form.append("frame", frameBlob, "frame.jpg");
  form.append("session_id", sessionId);
  const res = await fetch(`${BASE_URL}/api/recognize/sequence`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<SequenceResponse>;
}

/** Drop the backend's buffered frames for `sessionId` (best effort). */
export async function resetSequence(sessionId: string): Promise<void> {
  const form = new FormData();
  form.append("session_id", sessionId);
  await fetch(`${BASE_URL}/api/recognize/sequence/reset`, { method: "POST", body: form });
}
