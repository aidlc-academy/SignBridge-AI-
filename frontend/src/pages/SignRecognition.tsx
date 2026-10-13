import { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera, CameraOff, Trash2, Delete, Volume2, Copy, Save,
  AlertCircle, CheckCircle, Loader, WifiOff, Hand,
} from "lucide-react";
import { Card, CardHeader, CardTitle, StatusDot } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { PageHeader, StatusBanner, ComingSoonBadge } from "../components/ui/Primitives";
import { useRecognitionStore, useSettingsStore } from "../store";
import { useTTS } from "../hooks/useSpeech";
import { useCamera } from "../hooks/useCamera";
import { recognizeLandmarks, recognizeSequence, resetSequence, checkBackend } from "../services/api";
import type { LandmarkResponse } from "../types";

// MediaPipe hand connections (pairs of landmark indices)
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],           // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],           // Index
  [0, 9], [9, 10], [10, 11], [11, 12],      // Middle
  [0, 13], [13, 14], [14, 15], [15, 16],    // Ring
  [0, 17], [17, 18], [18, 19], [19, 20],    // Pinky
  [5, 9], [9, 13], [13, 17],                // Palm
];

// ── Camera + Landmark overlay ────────────────────────────────────────────

function CameraFeed({
  videoRef,
  canvasRef,
  status,
  error,
  onStart,
  onStop,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  status: "idle" | "requesting" | "active" | "denied" | "error";
  error: string | null;
  onStart: () => void;
  onStop: () => void;
}) {
  return (
    <div className="relative w-full aspect-video bg-slate-900 rounded-2xl overflow-hidden">
      {/* Grid background */}
      <div className="absolute inset-0" aria-hidden="true">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(99,102,241,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(99,102,241,0.04)_1px,transparent_1px)] bg-[size:40px_40px]" />
      </div>

      {/* Live video (always rendered, hidden when not active) */}
      <video
        ref={videoRef}
        className={`absolute inset-0 w-full h-full object-cover [transform:scaleX(-1)] ${
          status === "active" ? "block" : "hidden"
        }`}
        playsInline
        muted
        aria-label="Webcam feed"
      />

      {/* Canvas landmark overlay */}
      <canvas
        ref={canvasRef}
        className={`absolute inset-0 w-full h-full [transform:scaleX(-1)] pointer-events-none ${
          status === "active" ? "block" : "hidden"
        }`}
        aria-hidden="true"
      />

      {/* Idle state */}
      {status === "idle" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center">
            <Camera size={28} className="text-slate-500" />
          </div>
          <p className="text-slate-400 text-sm text-center max-w-xs">
            Camera is off. Click <strong className="text-slate-200">Start Camera</strong> to begin.
          </p>
          <Button onClick={onStart} icon={<Camera size={16} />}>Start Camera</Button>
        </div>
      )}

      {/* Requesting state */}
      {status === "requesting" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
          <Loader size={28} className="text-brand-primary animate-spin" />
          <p className="text-slate-300 text-sm">Requesting camera permission…</p>
        </div>
      )}

      {/* Denied state */}
      {status === "denied" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6">
          <div className="w-12 h-12 rounded-xl bg-red-900/40 flex items-center justify-center">
            <CameraOff size={24} className="text-red-400" />
          </div>
          <p className="text-red-300 text-sm text-center">{error}</p>
        </div>
      )}

      {/* Error state */}
      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6">
          <WifiOff size={28} className="text-slate-400" />
          <p className="text-slate-400 text-sm text-center">{error}</p>
          <Button variant="secondary" size="sm" onClick={onStart}>Retry</Button>
        </div>
      )}

      {/* Active overlays */}
      {status === "active" && (
        <>
          <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 backdrop-blur-sm rounded-full px-3 py-1.5">
            <StatusDot status="active" />
            <span className="text-white text-xs font-medium">Camera active</span>
          </div>
          <button
            onClick={onStop}
            className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm rounded-full p-2 text-white hover:bg-red-900/70 transition-colors focus-visible:ring-2 focus-visible:ring-white outline-none"
            aria-label="Stop camera"
          >
            <CameraOff size={16} />
          </button>
        </>
      )}
    </div>
  );
}

// ── Recognition status indicator ──────────────────────────────────────────

function RecognitionStatus({ status }: {
  status: "idle" | "detecting" | "recognized" | "confirmed" | "no_hand" | "low_confidence" | "model_unavailable";
}) {
  const map = {
    idle:             { dot: "idle"    as const, label: "Idle",               color: "text-content-secondary" },
    detecting:        { dot: "active"  as const, label: "Detecting hand…",     color: "text-brand-primary" },
    recognized:       { dot: "active"  as const, label: "Gesture recognized",  color: "text-brand-accent" },
    confirmed:        { dot: "active"  as const, label: "Gesture confirmed",   color: "text-status-success" },
    no_hand:          { dot: "idle"    as const, label: "No hand detected",    color: "text-content-secondary" },
    low_confidence:   { dot: "warning" as const, label: "Low confidence",      color: "text-status-warning" },
    model_unavailable:{ dot: "error"   as const, label: "Model not available", color: "text-status-error" },
  };
  const s = map[status];
  return (
    <div className="flex items-center gap-2">
      <StatusDot status={s.dot} />
      <span className={`text-sm font-medium ${s.color}`}>{s.label}</span>
    </div>
  );
}

// ── Confidence bar ─────────────────────────────────────────────────────────

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "bg-status-success" : pct >= 60 ? "bg-status-warning" : "bg-status-error";
  return (
    <div>
      <div className="flex items-center justify-between text-xs text-content-secondary mb-1">
        <span>Confidence</span>
        <span className="font-medium">{pct}%</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className={`h-full rounded-full transition-all duration-300 ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ── Hand landmark drawing ──────────────────────────────────────────────────

function drawLandmarks(
  canvas: HTMLCanvasElement,
  landmarkData: LandmarkResponse,
  videoWidth: number,
  videoHeight: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Size the canvas backing store to its DISPLAYED box (same box the video
  // fills). Using the displayed size — not the native video size — keeps the
  // overlay geometry identical to what the user sees. Account for devicePixel
  // ratio so lines stay crisp on HiDPI screens.
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const dispW = Math.max(1, Math.round(rect.width));
  const dispH = Math.max(1, Math.round(rect.height));
  if (canvas.width !== dispW * dpr || canvas.height !== dispH * dpr) {
    canvas.width = dispW * dpr;
    canvas.height = dispH * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, dispW, dispH);

  // The video is rendered with `object-cover`, which scales the native frame
  // to COVER the box and crops the overflow. Landmark coords are normalized to
  // the captured frame, so we must apply the same cover transform to place
  // them correctly when the frame aspect ratio != the box aspect ratio.
  const frameAspect = videoWidth / videoHeight;
  const boxAspect = dispW / dispH;
  let scale: number, offsetX: number, offsetY: number;
  if (frameAspect > boxAspect) {
    // Frame is wider than box → cover by height, crop sides.
    scale = dispH / videoHeight;
    offsetX = (dispW - videoWidth * scale) / 2;
    offsetY = 0;
  } else {
    // Frame is taller than box → cover by width, crop top/bottom.
    scale = dispW / videoWidth;
    offsetX = 0;
    offsetY = (dispH - videoHeight * scale) / 2;
  }
  const mapX = (nx: number) => offsetX + nx * videoWidth * scale;
  const mapY = (ny: number) => offsetY + ny * videoHeight * scale;

  for (const hand of landmarkData.hands) {
    const pts = hand.landmarks;

    // Draw connections
    ctx.strokeStyle = "rgba(99, 102, 241, 0.85)";
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      const p1 = pts[a];
      const p2 = pts[b];
      if (!p1 || !p2) continue;
      // Note: both the video and this canvas are CSS-mirrored ([scaleX(-1)]),
      // and the backend processes the UN-mirrored captured frame, so the two
      // mirror transforms cancel and landmark x maps directly.
      ctx.beginPath();
      ctx.moveTo(mapX(p1.x), mapY(p1.y));
      ctx.lineTo(mapX(p2.x), mapY(p2.y));
      ctx.stroke();
    }

    // Draw landmark dots
    for (let i = 0; i < pts.length; i++) {
      const pt = pts[i];
      if (!pt) continue;
      const x = mapX(pt.x);
      const y = mapY(pt.y);
      // Fingertips (4,8,12,16,20) are larger
      const isTip = [4, 8, 12, 16, 20].includes(i);
      const radius = isTip ? 6 : 4;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = isTip ? "rgba(251, 191, 36, 0.95)" : "rgba(255, 255, 255, 0.9)";
      ctx.fill();
      ctx.strokeStyle = isTip ? "rgba(217, 119, 6, 1)" : "rgba(99, 102, 241, 0.8)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function SignRecognition() {
  const { currentMessage, appendToMessage, backspaceMessage, clearMessage } = useRecognitionStore();
  const { settings } = useSettingsStore();
  const { speak, isSpeaking, supported: ttsSupported } = useTTS();

  // Real camera state from hook
  const { videoRef, status: cameraStatus, error: cameraError, dimensions, startCamera, stopCamera, captureFrame } = useCamera();

  // Canvas ref for landmark overlay
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Recognition loop state
  const [recognitionStatus, setRecognitionStatus] = useState<
    "idle" | "detecting" | "recognized" | "confirmed" | "no_hand" | "low_confidence" | "model_unavailable"
  >("idle");
  const [handsDetected, setHandsDetected] = useState(0);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const isRequestingRef = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // LSTM word recognition (additional method): its own in-flight guard so a slow
  // sequence request never delays the landmark overlay, plus a per-camera-session id.
  const [lstmConfidence, setLstmConfidence] = useState(0);
  const isSequenceRequestingRef = useRef(false);
  const sequenceSessionRef = useRef("");

  // ── Dev-only diagnostics ──────────────────────────────────────────────
  // Guarded by import.meta.env.DEV so it is tree-shaken out of production.
  const [diag, setDiag] = useState({
    framesSent: 0,
    framesOk: 0,
    framesFailed: 0,
    lastFrameBytes: 0,
    backendReachable: null as boolean | null,
    lastError: null as string | null,
  });

  // Run recognition loop when camera is active
  useEffect(() => {
    if (cameraStatus !== "active") {
      // Stop loop
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      // Clear canvas
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      setRecognitionStatus("idle");
      setHandsDetected(0);
      setLatencyMs(null);
      setLstmConfidence(0);
      if (sequenceSessionRef.current) {
        // Camera stopped: drop the backend's buffered frames for this session.
        resetSequence(sequenceSessionRef.current).catch(() => {});
        sequenceSessionRef.current = "";
      }
      if (import.meta.env.DEV) {
        setDiag((d) => ({
          ...d,
          framesSent: 0,
          framesOk: 0,
          framesFailed: 0,
          lastFrameBytes: 0,
          lastError: null,
        }));
      }
      return;
    }

    // Start inference loop at ~8fps
    sequenceSessionRef.current =
      (typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : Math.random().toString(36).slice(2)) as string;
    intervalRef.current = setInterval(async () => {
      if (isRequestingRef.current) return; // prevent overlapping requests

      const blob = captureFrame();
      if (!blob) return;

      // LSTM word recognition runs alongside (not instead of) the landmark path.
      if (!isSequenceRequestingRef.current && sequenceSessionRef.current) {
        isSequenceRequestingRef.current = true;
        recognizeSequence(blob, sequenceSessionRef.current)
          .then((seq) => {
            if (intervalRef.current === null) return; // camera stopped while in flight
            setLstmConfidence(seq.status === "recognized" ? seq.confidence : 0);
            if (seq.confirmed_word) appendToMessage(seq.confirmed_word + " ");
            if (import.meta.env.DEV && (seq.status === "model_unavailable" || seq.status === "error")) {
              setDiag((d) => ({ ...d, lastError: `LSTM: ${seq.message}` }));
            }
          })
          .catch(() => {
            if (intervalRef.current !== null) setLstmConfidence(0);
          })
          .finally(() => {
            isSequenceRequestingRef.current = false;
          });
      }

      isRequestingRef.current = true;
      setRecognitionStatus("detecting");
      if (import.meta.env.DEV) {
        setDiag((d) => ({ ...d, framesSent: d.framesSent + 1, lastFrameBytes: blob.size }));
      }

      try {
        const result = await recognizeLandmarks(blob);
        setHandsDetected(result.hands_detected);
        setLatencyMs(result.latency_ms);
        if (import.meta.env.DEV) {
          setDiag((d) => ({ ...d, framesOk: d.framesOk + 1, backendReachable: true, lastError: null }));
        }

        if (result.status === "error") {
          // Backend-side vision error (e.g. MediaPipe init failure) — surface it.
          setRecognitionStatus("model_unavailable");
          if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          }
          if (import.meta.env.DEV) {
            setDiag((d) => ({ ...d, lastError: result.message }));
          }
          return;
        }

        if (result.hands_detected > 0 && canvasRef.current && videoRef.current) {
          const vw = videoRef.current.videoWidth || 640;
          const vh = videoRef.current.videoHeight || 480;
          drawLandmarks(canvasRef.current, result, vw, vh);
          setRecognitionStatus("recognized");
        } else {
          if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          }
          setRecognitionStatus("no_hand");
        }
      } catch (err) {
        // Distinguish a backend/network failure from a genuine no-hand result.
        // A thrown error means the request itself failed (backend down, CORS,
        // proxy misconfigured) — NOT that there is simply no hand in frame.
        const message = err instanceof Error ? err.message : "Request failed";
        setRecognitionStatus("model_unavailable");
        setHandsDetected(0);
        if (canvasRef.current) {
          const ctx = canvasRef.current.getContext("2d");
          ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
        if (import.meta.env.DEV) {
          setDiag((d) => ({
            ...d,
            framesFailed: d.framesFailed + 1,
            backendReachable: false,
            lastError: message,
          }));
        }
      } finally {
        isRequestingRef.current = false;
      }
    }, 125); // ~8fps

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [cameraStatus, captureFrame, videoRef, appendToMessage]);

  // Dev-only: probe backend reachability once on mount so the diagnostics
  // panel reflects connectivity before the camera is even started.
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    let cancelled = false;
    checkBackend().then((ok) => {
      if (!cancelled) setDiag((d) => ({ ...d, backendReachable: ok }));
    });
    return () => { cancelled = true; };
  }, []);

  const demoLetters = ["A", "B", "C", "D", "I", "L", "O", "V", "Y", "1"];

  const handleSpeak = useCallback(() => {
    if (!currentMessage.trim()) return;
    speak(currentMessage, settings.voiceName, settings.speechRate);
  }, [currentMessage, speak, settings]);

  const handleCopy = useCallback(() => {
    if (!currentMessage.trim()) return;
    navigator.clipboard.writeText(currentMessage).catch(() => {});
  }, [currentMessage]);

  // Derive display status for RecognitionStatus component
  const displayStatus = cameraStatus === "idle" || cameraStatus === "denied"
    ? "idle"
    : recognitionStatus;

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-5">
      <PageHeader
        title="Sign to Text"
        description="Live gesture recognition via webcam."
        icon={<Hand size={20} />}
        badge={<ComingSoonBadge phase={3} />}
      />

      {(cameraStatus === "denied" || cameraStatus === "error") && (
        <StatusBanner type="error">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <span>{cameraError}</span>
        </StatusBanner>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-5">
        {/* ── Left: Camera + status ── */}
        <div className="space-y-4">
          <CameraFeed
            videoRef={videoRef}
            canvasRef={canvasRef}
            status={cameraStatus}
            error={cameraError}
            onStart={startCamera}
            onStop={stopCamera}
          />

          {/* Controls row */}
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              icon={<Camera size={16} />}
              onClick={startCamera}
              disabled={cameraStatus !== "idle" && cameraStatus !== "denied" && cameraStatus !== "error"}
            >
              Start Camera
            </Button>
            <Button
              icon={<CameraOff size={16} />}
              variant="secondary"
              onClick={stopCamera}
              disabled={cameraStatus !== "active"}
            >
              Stop Camera
            </Button>
          </div>

          {/* Recognition status */}
          <Card>
            <CardHeader>
              <CardTitle>Recognition Status</CardTitle>
            </CardHeader>
            <div className="space-y-4">
              <RecognitionStatus status={displayStatus} />
              <ConfidenceBar value={lstmConfidence} />
              <div className="flex items-center justify-between text-sm">
                <span className="text-content-secondary">Hands detected</span>
                <span className="font-mono font-bold text-content-primary text-lg">
                  {cameraStatus === "active" ? handsDetected : "—"}
                </span>
              </div>
              {latencyMs !== null && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-content-secondary">Latency</span>
                  <span className="font-mono text-content-primary">{latencyMs}ms</span>
                </div>
              )}
            </div>
          </Card>

          {/* Dev-only diagnostics (tree-shaken from production builds) */}
          {import.meta.env.DEV && (
            <Card className="bg-slate-900 text-slate-100 border-slate-700">
              <CardHeader>
                <CardTitle className="text-slate-100">Diagnostics (dev only)</CardTitle>
              </CardHeader>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs font-mono">
                <dt className="text-slate-400">Camera status</dt>
                <dd className="text-right">{cameraStatus}</dd>

                <dt className="text-slate-400">Video dimensions</dt>
                <dd className="text-right">
                  {dimensions ? `${dimensions.width}×${dimensions.height}` : "—"}
                </dd>

                <dt className="text-slate-400">Backend reachable</dt>
                <dd className="text-right">
                  {diag.backendReachable === null
                    ? "checking…"
                    : diag.backendReachable
                    ? <span className="text-green-400">yes</span>
                    : <span className="text-red-400">no</span>}
                </dd>

                <dt className="text-slate-400">Frames sent</dt>
                <dd className="text-right">{diag.framesSent}</dd>

                <dt className="text-slate-400">Frames ok / failed</dt>
                <dd className="text-right">
                  <span className="text-green-400">{diag.framesOk}</span>
                  {" / "}
                  <span className={diag.framesFailed > 0 ? "text-red-400" : ""}>{diag.framesFailed}</span>
                </dd>

                <dt className="text-slate-400">Last frame size</dt>
                <dd className="text-right">
                  {diag.lastFrameBytes > 0 ? `${(diag.lastFrameBytes / 1024).toFixed(1)} KB` : "—"}
                </dd>

                <dt className="text-slate-400">Hands detected</dt>
                <dd className="text-right">{handsDetected}</dd>

                <dt className="text-slate-400">Latency</dt>
                <dd className="text-right">{latencyMs !== null ? `${latencyMs} ms` : "—"}</dd>
              </dl>
              {diag.lastError && (
                <p className="mt-3 text-xs text-red-300 break-words border-t border-slate-700 pt-2">
                  <span className="text-slate-400">Last error:</span> {diag.lastError}
                </p>
              )}
            </Card>
          )}

          {/* Manual input for demo */}
          <Card>
            <CardHeader>
              <CardTitle>
                Manual Input{" "}
                <span className="text-xs font-normal text-content-secondary">(demo only)</span>
              </CardTitle>
            </CardHeader>
            <p className="text-xs text-content-secondary mb-3">
              Tap a sign to add it to the output panel.
            </p>
            <div className="flex flex-wrap gap-2">
              {demoLetters.map((l) => (
                <button
                  key={l}
                  onClick={() => appendToMessage(l)}
                  className="w-10 h-10 rounded-xl bg-indigo-50 text-brand-primary font-bold text-sm hover:bg-brand-primary hover:text-white transition-colors focus-visible:ring-2 focus-visible:ring-brand-primary outline-none"
                  aria-label={`Add sign ${l}`}
                >
                  {l}
                </button>
              ))}
            </div>
          </Card>
        </div>

        {/* ── Right: Output panel ── */}
        <div className="space-y-4">
          <Card className="flex-1">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Recognized Message</CardTitle>
                {currentMessage && <CheckCircle size={16} className="text-status-success" />}
              </div>
            </CardHeader>

            <div
              className="min-h-[100px] bg-slate-50 rounded-xl p-4 mb-4 font-mono text-lg text-content-primary break-words leading-relaxed"
              aria-live="polite"
              aria-label="Current recognized message"
            >
              {currentMessage || (
                <span className="text-content-secondary text-sm font-sans not-italic">
                  Recognized gestures will appear here…
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2 mb-4">
              <Button
                variant="ghost"
                size="sm"
                icon={<Delete size={14} />}
                onClick={backspaceMessage}
                disabled={!currentMessage}
                aria-label="Backspace"
              >
                Backspace
              </Button>
              <Button
                variant="ghost"
                size="sm"
                icon={<Trash2 size={14} />}
                onClick={clearMessage}
                disabled={!currentMessage}
                aria-label="Clear message"
              >
                Clear
              </Button>
              <Button
                variant="ghost"
                size="sm"
                icon={<Copy size={14} />}
                onClick={handleCopy}
                disabled={!currentMessage}
                aria-label="Copy to clipboard"
              >
                Copy
              </Button>
              {ttsSupported ? (
                <Button
                  variant="secondary"
                  size="sm"
                  icon={<Volume2 size={14} />}
                  onClick={handleSpeak}
                  disabled={!currentMessage || isSpeaking}
                  aria-label="Speak message"
                >
                  {isSpeaking ? "Speaking…" : "Speak"}
                </Button>
              ) : (
                <Button variant="ghost" size="sm" disabled title="TTS not supported in this browser">
                  No TTS
                </Button>
              )}
            </div>

            <Button
              variant="primary"
              icon={<Save size={14} />}
              disabled={!currentMessage}
              className="w-full"
            >
              Save Conversation
            </Button>
          </Card>

          {/* How recognition works */}
          <Card className="bg-slate-50">
            <CardHeader><CardTitle>Recognition Pipeline</CardTitle></CardHeader>
            <ol className="space-y-2 text-xs text-content-secondary list-none">
              {[
                "Frame captured from webcam",
                "MediaPipe extracts 21 hand landmarks",
                "Landmarks normalized (wrist-relative)",
                "RandomForest predicts gesture class",
                "Sliding window stabilizes prediction",
                "Confirmed gesture added to message",
              ].map((step, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="flex-shrink-0 w-4 h-4 rounded-full bg-indigo-100 text-brand-primary flex items-center justify-center font-bold text-[10px]">
                    {i + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
          </Card>
        </div>
      </div>
    </div>
  );
}
