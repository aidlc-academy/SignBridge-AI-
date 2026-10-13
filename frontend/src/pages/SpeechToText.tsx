import { useCallback } from "react";
import { Mic, MicOff, Copy, Trash2, Volume2, AlertCircle } from "lucide-react";
import { Card, CardHeader, CardTitle, StatusDot } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { PageHeader, StatusBanner } from "../components/ui/Primitives";
import { useSTT } from "../hooks/useSpeech";
import { useTTS } from "../hooks/useSpeech";

export default function SpeechToText() {
  const {
    transcript, interimTranscript, status, error,
    supported, startListening, stopListening, resetTranscript, setTranscript,
  } = useSTT();

  const { speak, isSpeaking, supported: ttsSupported } = useTTS();

  const handleCopy = useCallback(() => {
    if (!transcript.trim()) return;
    navigator.clipboard.writeText(transcript).catch(() => {});
  }, [transcript]);

  const handleSpeak = useCallback(() => {
    if (!transcript.trim()) return;
    speak(transcript);
  }, [transcript, speak]);

  const isListening = status === "listening";
  const hasError    = status === "error";

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-5">
      <PageHeader
        title="Speech to Text"
        description="Capture spoken responses using your browser's speech recognition API."
        icon={<Mic size={20} />}
        iconColor="bg-teal-50 text-brand-accent"
      />

      {!supported && (
        <StatusBanner type="warning">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <span>
            Browser speech recognition is not available in this browser.
            It works in Chrome and Edge. You can type your message manually below.
          </span>
        </StatusBanner>
      )}

      {hasError && error && (
        <StatusBanner type="error">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </StatusBanner>
      )}

      {/* Microphone control */}
      <Card className="flex flex-col items-center gap-5 py-8">
        <button
          onClick={isListening ? stopListening : startListening}
          disabled={!supported}
          className={`
            w-20 h-20 rounded-full flex items-center justify-center
            transition-all duration-200 outline-none
            focus-visible:ring-4 focus-visible:ring-offset-2
            ${isListening
              ? "bg-status-error text-white focus-visible:ring-red-200 shadow-lg shadow-red-100 scale-105"
              : "bg-brand-primary text-white focus-visible:ring-indigo-200 hover:bg-brand-secondary"
            }
            disabled:opacity-40 disabled:cursor-not-allowed
          `}
          aria-label={isListening ? "Stop listening" : "Start listening"}
          aria-pressed={isListening}
        >
          {isListening ? <MicOff size={32} /> : <Mic size={32} />}
        </button>

        <div className="flex items-center gap-2">
          <StatusDot status={isListening ? "active" : hasError ? "error" : "idle"} />
          <span className="text-sm font-medium text-content-secondary">
            {isListening ? "Listening… speak now" : hasError ? "Error" : "Ready"}
          </span>
        </div>

        {/* Interim transcript hint */}
        {interimTranscript && (
          <p className="text-sm text-content-secondary italic text-center max-w-sm" aria-live="polite">
            "{interimTranscript}"
          </p>
        )}
      </Card>

      {/* Transcript area */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Transcript</CardTitle>
            {isListening && (
              <span className="text-xs text-status-error font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-status-error animate-pulse-slow" />
                Recording
              </span>
            )}
          </div>
        </CardHeader>

        {supported ? (
          <textarea
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder="Transcript appears here. You can also type directly."
            className="w-full min-h-[140px] resize-y text-sm text-content-primary bg-slate-50 border border-surface-border rounded-xl p-3 focus:ring-2 focus:ring-brand-primary focus:outline-none"
            aria-label="Speech transcript"
            aria-live="polite"
          />
        ) : (
          <textarea
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder="Type your message here (speech recognition unavailable)."
            className="w-full min-h-[140px] resize-y text-sm text-content-primary bg-slate-50 border border-surface-border rounded-xl p-3 focus:ring-2 focus:ring-brand-primary focus:outline-none"
            aria-label="Manual text entry"
          />
        )}

        <div className="flex items-center gap-2 mt-3 flex-wrap">
          <Button variant="ghost" size="sm" icon={<Copy size={14} />} onClick={handleCopy} disabled={!transcript}>
            Copy
          </Button>
          <Button variant="ghost" size="sm" icon={<Trash2 size={14} />} onClick={resetTranscript} disabled={!transcript}>
            Clear
          </Button>
          {ttsSupported && (
            <Button variant="secondary" size="sm" icon={<Volume2 size={14} />} onClick={handleSpeak} disabled={!transcript || isSpeaking}>
              {isSpeaking ? "Speaking…" : "Speak"}
            </Button>
          )}
        </div>
      </Card>

      {!supported && (
        <p className="text-xs text-content-secondary text-center">
          For speech recognition, use Google Chrome or Microsoft Edge.
          The SpeechRecognition API may route audio through a cloud service depending on your browser.
        </p>
      )}
    </div>
  );
}
