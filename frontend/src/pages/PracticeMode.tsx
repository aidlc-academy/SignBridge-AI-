import { useState } from "react";
import { BookOpen, ChevronLeft, ChevronRight, AlertCircle } from "lucide-react";
import { Card, CardHeader, CardTitle, Badge } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { PageHeader, StatusBanner, ComingSoonBadge } from "../components/ui/Primitives";

interface GestureRef {
  label: string;
  description: string;
  hint: string;
}

const GESTURES: GestureRef[] = [
  { label: "A", description: "Fist with thumb resting on the side of the index finger.",       hint: "Make a fist, bring thumb to side." },
  { label: "B", description: "Flat hand, four fingers up, thumb folded across palm.",           hint: "Hold hand flat, thumb in." },
  { label: "C", description: "Curved hand forming a C shape, like holding a cup.",             hint: "Curve fingers and thumb toward each other." },
  { label: "D", description: "Index finger pointing up, other fingers curl to touch thumb.",   hint: "Tap index up, form O with rest." },
  { label: "I", description: "Pinky finger extended upward, others curled into a fist.",       hint: "Only pinky extended." },
  { label: "L", description: "Index up and thumb out, forming an L shape.",                    hint: "L-shape: index up, thumb out." },
  { label: "O", description: "All fingers curve down to touch the thumb, forming an O.",       hint: "All fingertips meet the thumb tip." },
  { label: "V", description: "Index and middle fingers extended in a V / peace sign.",          hint: "Two fingers up, others curled." },
  { label: "Y", description: "Thumb and pinky extended, three middle fingers curled.",          hint: "Shaka / hang-loose shape." },
  { label: "1", description: "Index finger pointing straight up, others curled.",              hint: "Pointer finger only, like number 1." },
];

export default function PracticeMode() {
  const [current, setCurrent] = useState(0);
  const gesture = GESTURES[current];

  const prev = () => setCurrent((c) => Math.max(0, c - 1));
  const next = () => setCurrent((c) => Math.min(GESTURES.length - 1, c + 1));

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-5">
      <PageHeader
        title="Practice Mode"
        description="Learn each supported sign with reference descriptions and live feedback."
        icon={<BookOpen size={20} />}
        iconColor="bg-amber-50 text-status-warning"
        badge={<ComingSoonBadge phase={9} />}
      />

      <StatusBanner type="info">
        <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
        <span>
          Live webcam feedback connects in Phase 9. For now, review the gesture library and reference descriptions.
        </span>
      </StatusBanner>

      {/* Progress */}
      <div>
        <div className="flex items-center justify-between text-xs text-content-secondary mb-2">
          <span>Gesture {current + 1} of {GESTURES.length}</span>
          <span>{Math.round(((current + 1) / GESTURES.length) * 100)}% complete</span>
        </div>
        <div className="h-2 bg-slate-100 rounded-full overflow-hidden" role="progressbar" aria-valuenow={current + 1} aria-valuemin={1} aria-valuemax={GESTURES.length}>
          <div
            className="h-full bg-status-warning rounded-full transition-all duration-300"
            style={{ width: `${((current + 1) / GESTURES.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Gesture card */}
      <Card>
        <div className="flex flex-col sm:flex-row items-center gap-6 py-4">
          {/* Sign display */}
          <div className="flex-shrink-0 w-32 h-32 rounded-3xl gradient-brand flex items-center justify-center shadow-lg">
            <span className="text-6xl font-black text-white">{gesture.label}</span>
          </div>

          <div className="flex-1 text-center sm:text-left">
            <div className="flex items-center gap-2 justify-center sm:justify-start mb-2">
              <h2 className="text-2xl font-bold text-content-primary">Sign "{gesture.label}"</h2>
              <Badge variant="info">ASL</Badge>
            </div>
            <p className="text-sm text-content-primary mb-2 leading-relaxed">{gesture.description}</p>
            <div className="inline-flex items-center gap-2 bg-amber-50 text-status-warning text-xs font-medium px-3 py-1.5 rounded-xl">
              <span>💡</span>
              {gesture.hint}
            </div>
          </div>
        </div>
      </Card>

      {/* Nav controls */}
      <div className="flex items-center justify-between">
        <Button variant="secondary" icon={<ChevronLeft size={16} />} onClick={prev} disabled={current === 0}>
          Previous
        </Button>
        <div className="flex gap-1.5">
          {GESTURES.map((_, i) => (
            <button
              key={i}
              onClick={() => setCurrent(i)}
              className={`w-2 h-2 rounded-full transition-all outline-none focus-visible:ring-2 focus-visible:ring-brand-primary ${
                i === current ? "bg-brand-primary scale-125" : "bg-slate-200 hover:bg-slate-300"
              }`}
              aria-label={`Go to gesture ${i + 1}`}
              aria-current={i === current}
            />
          ))}
        </div>
        <Button icon={<ChevronRight size={16} />} onClick={next} disabled={current === GESTURES.length - 1}>
          Next
        </Button>
      </div>

      {/* Full gesture list */}
      <Card>
        <CardHeader><CardTitle>All Supported Signs</CardTitle></CardHeader>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {GESTURES.map((g, i) => (
            <button
              key={g.label}
              onClick={() => setCurrent(i)}
              className={`p-3 rounded-xl text-center transition-all outline-none focus-visible:ring-2 focus-visible:ring-brand-primary ${
                i === current
                  ? "bg-brand-primary text-white"
                  : "bg-slate-50 hover:bg-indigo-50 text-content-secondary hover:text-brand-primary"
              }`}
              aria-current={i === current}
            >
              <div className="text-2xl font-bold">{g.label}</div>
              <div className="text-[10px] mt-1 opacity-70 leading-tight">{g.hint.split(",")[0]}</div>
            </button>
          ))}
        </div>
      </Card>
    </div>
  );
}
