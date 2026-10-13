import { useState } from "react";
import { AlertTriangle, Volume2, Copy, Check } from "lucide-react";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { PageHeader, StatusBanner } from "../components/ui/Primitives";
import { useTTS } from "../hooks/useSpeech";

interface Phrase {
  en: string;
  ta: string;
}

const PHRASES: Phrase[] = [
  { en: "I need help",             ta: "எனக்கு உதவி தேவை" },
  { en: "I need water",            ta: "எனக்கு தண்ணீர் தேவை" },
  { en: "Please call a doctor",    ta: "தயவுசெய்து மருத்துவரை அழைக்கவும்" },
  { en: "I am in pain",            ta: "என்னால் வலி இருக்கிறது" },
  { en: "I need assistance",       ta: "எனக்கு உதவி தேவை" },
  { en: "Please wait",             ta: "தயவுசெய்து காத்திருங்கள்" },
  { en: "Thank you",               ta: "நன்றி" },
  { en: "Call emergency services", ta: "அவசர சேவைகளை அழைக்கவும்" },
];

function PhraseCard({ phrase }: { phrase: Phrase }) {
  const { speak, supported: ttsSupported } = useTTS();
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(phrase.en).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }).catch(() => {});
  };

  return (
    <Card className="flex flex-col gap-3 hover:shadow-md transition-shadow">
      <div>
        <p className="font-semibold text-content-primary text-base">{phrase.en}</p>
        <p className="text-sm text-content-secondary mt-0.5">{phrase.ta}</p>
      </div>
      <div className="flex items-center gap-2 mt-auto">
        {ttsSupported && (
          <Button
            size="sm"
            variant="secondary"
            icon={<Volume2 size={14} />}
            onClick={() => speak(phrase.en)}
            aria-label={`Speak: ${phrase.en}`}
          >
            Speak
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          icon={copied ? <Check size={14} /> : <Copy size={14} />}
          onClick={handleCopy}
          aria-label={`Copy: ${phrase.en}`}
        >
          {copied ? "Copied!" : "Copy"}
        </Button>
      </div>
    </Card>
  );
}

export default function Emergency() {
  return (
    <div className="p-6 max-w-4xl mx-auto space-y-5">
      <PageHeader
        title="Emergency Phrases"
        description="Quick-access communication cards for critical situations."
        icon={<AlertTriangle size={20} />}
        iconColor="bg-red-50 text-status-error"
      />

      <StatusBanner type="error">
        <AlertTriangle size={16} className="flex-shrink-0 mt-0.5" />
        <span>
          <strong>Important:</strong> This application is not a replacement for emergency services (112 / 911) or qualified sign-language interpreters.
          Use these cards as a communication aid only.
        </span>
      </StatusBanner>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {PHRASES.map((phrase) => (
          <PhraseCard key={phrase.en} phrase={phrase} />
        ))}
      </div>

      <p className="text-xs text-content-secondary text-center">
        Tamil translations are static text for display only.
        Tamil TTS depends on browser and OS voice availability.
        Speak out loud to verify Tamil voice support before relying on it.
      </p>
    </div>
  );
}
