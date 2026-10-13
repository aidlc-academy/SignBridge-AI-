import { useState, useCallback, useRef, useEffect } from "react";

// ── Text-to-Speech ─────────────────────────────────────────────────────────

export interface UseTTSReturn {
  isSpeaking: boolean;
  voices: SpeechSynthesisVoice[];
  speak: (text: string, voiceName?: string, rate?: number) => void;
  stop: () => void;
  supported: boolean;
}

export function useTTS(): UseTTSReturn {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (!supported) return;
    const load = () => setVoices(window.speechSynthesis.getVoices());
    load();
    window.speechSynthesis.onvoiceschanged = load;
    return () => { window.speechSynthesis.onvoiceschanged = null; };
  }, [supported]);

  const speak = useCallback(
    (text: string, voiceName = "", rate = 1.0) => {
      if (!supported || !text.trim()) return;
      window.speechSynthesis.cancel(); // prevent overlap
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = Math.max(0.1, Math.min(rate, 10));
      if (voiceName) {
        const match = voices.find((v) => v.name === voiceName);
        if (match) utterance.voice = match;
      }
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utterance);
    },
    [supported, voices]
  );

  const stop = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, [supported]);

  return { isSpeaking, voices, speak, stop, supported };
}

// ── Speech-to-Text ─────────────────────────────────────────────────────────

type STTStatus = "idle" | "listening" | "error" | "unsupported";

export interface UseSTTReturn {
  transcript: string;
  interimTranscript: string;
  status: STTStatus;
  error: string | null;
  supported: boolean;
  startListening: () => void;
  stopListening: () => void;
  resetTranscript: () => void;
  setTranscript: (v: string) => void;
}

export function useSTT(): UseSTTReturn {
  // Feature detect: SpeechRecognition is available in Chrome/Edge; absent in Firefox
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = typeof window !== "undefined" ? (window as any) : null;
  const SpeechRecognitionAPI = w
    ? (w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null)
    : null;

  const supported = !!SpeechRecognitionAPI;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [status, setStatus] = useState<STTStatus>(supported ? "idle" : "unsupported");
  const [error, setError] = useState<string | null>(null);

  const startListening = useCallback(() => {
    if (!supported || !SpeechRecognitionAPI) return;
    if (status === "listening") return;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const recognition = new SpeechRecognitionAPI() as any;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event: any) => {
      let interim = "";
      let final = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          final += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }
      if (final) setTranscript((prev) => prev + final);
      setInterimTranscript(interim);
    };

    recognition.onerror = (event: any) => {
      const msg =
        event.error === "not-allowed"
          ? "Microphone permission was denied. Please allow microphone access and try again."
          : `Speech recognition error: ${event.error}`;
      setError(msg);
      setStatus("error");
    };

    recognition.onend = () => {
      setStatus("idle");
      setInterimTranscript("");
    };

    recognitionRef.current = recognition;
    recognition.start();
    setStatus("listening");
    setError(null);
  }, [supported, status, SpeechRecognitionAPI]);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setStatus("idle");
    setInterimTranscript("");
  }, []);

  const resetTranscript = useCallback(() => {
    setTranscript("");
    setInterimTranscript("");
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
    };
  }, []);

  return {
    transcript,
    interimTranscript,
    status,
    error,
    supported,
    startListening,
    stopListening,
    resetTranscript,
    setTranscript,
  };
}
