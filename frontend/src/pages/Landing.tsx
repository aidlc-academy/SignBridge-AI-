import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Zap, Camera, FileText, Volume2, Mic, Globe,
  BookOpen, MessageSquare, AlertTriangle, ChevronRight,
  Shield, Clock, Users, Menu, X, ArrowRight,
} from "lucide-react";
import { Button } from "../components/ui/Button";

// ── Animated communication bridge ─────────────────────────────────────────

function CommBridge() {
  const [active, setActive] = useState(0);
  const steps = ["A", "→", "Hello", "→", "🔊"];

  useEffect(() => {
    const id = setInterval(() => setActive((p) => (p + 1) % steps.length), 900);
    return () => clearInterval(id);
  }, [steps.length]);

  return (
    <div
      className="flex items-center justify-center gap-3 py-6"
      aria-label="Communication pipeline animation"
      aria-live="polite"
    >
      {steps.map((s, i) => (
        <div
          key={i}
          className={`
            transition-all duration-300 ease-out select-none
            ${i === active
              ? "scale-125 opacity-100"
              : i < active
                ? "opacity-40 scale-100"
                : "opacity-20 scale-95"}
            ${s === "→" ? "text-brand-accent text-xl font-bold" : ""}
            ${s !== "→" ? "w-12 h-12 rounded-2xl flex items-center justify-center font-bold text-lg" : ""}
            ${s !== "→" && i === active
              ? "bg-brand-primary text-white shadow-lg shadow-indigo-200"
              : s !== "→"
                ? "bg-white border border-surface-border text-content-secondary"
                : ""}
          `}
          aria-hidden={i !== active}
        >
          {s}
        </div>
      ))}
    </div>
  );
}

// ── Feature data ───────────────────────────────────────────────────────────

const FEATURES = [
  { icon: <Camera size={20} />, title: "Real-Time Gesture Recognition",  desc: "MediaPipe detects 21 hand landmarks per frame; a trained RandomForest classifies the gesture." },
  { icon: <FileText size={20} />, title: "Sign-to-Text Conversion",       desc: "Stabilized gestures build an editable message with confidence scores." },
  { icon: <Volume2 size={20} />, title: "Text-to-Speech",                 desc: "Speak composed messages using your browser's built-in synthesis engine." },
  { icon: <Mic size={20} />, title: "Speech-to-Text",                     desc: "A hearing person can respond verbally; their speech is transcribed in real time." },
  { icon: <Globe size={20} />, title: "Multilingual Output",              desc: "English interface with Tamil display translations for supported phrases." },
  { icon: <BookOpen size={20} />, title: "Practice Mode",                 desc: "Step through each sign with live model feedback." },
  { icon: <MessageSquare size={20} />, title: "Conversation History",     desc: "Opt-in text-only history. No frames or audio stored." },
  { icon: <AlertTriangle size={20} />, title: "Emergency Phrases",        desc: "One-tap cards for critical messages with speech and copy." },
];

const PIPELINE = [
  { step: "01", label: "Camera",      desc: "Browser MediaDevices API captures frames" },
  { step: "02", label: "Detect",      desc: "MediaPipe finds 21 hand landmarks" },
  { step: "03", label: "Classify",    desc: "RandomForest predicts the gesture class" },
  { step: "04", label: "Stabilize",   desc: "Sliding window majority vote confirms" },
  { step: "05", label: "Output",      desc: "Text appears; optionally spoken aloud" },
];

const VOCAB = ["A","B","C","D","I","L","O","V","Y","1"];

const FAQS = [
  {
    q: "Which sign language does SignBridge AI support?",
    a: "Phase 1 supports 10 American Sign Language (ASL) static handshapes. Indian Sign Language support will be added when suitable training data is available.",
  },
  {
    q: "Does it understand the full ASL alphabet?",
    a: "No. The model covers 10 unambiguous static signs: A, B, C, D, I, L, O, V, Y, and numeral 1. Motion-dependent signs (J, Z) require multi-frame tracking, which will be added in a later phase.",
  },
  {
    q: "Does the app store my camera footage?",
    a: "No. Frames are processed in memory on the backend and discarded immediately. Conversation history is opt-in and stores text only.",
  },
  {
    q: "Does speech recognition work offline?",
    a: "The browser SpeechRecognition API may route audio to a cloud service depending on your browser and OS. It is not guaranteed to be offline or private.",
  },
  {
    q: "What browsers are supported?",
    a: "Sign recognition and TTS work in any modern browser. STT works best in Chrome and Edge; Firefox users receive a manual-entry fallback.",
  },
];

// ── Navigation ─────────────────────────────────────────────────────────────

function LandingNav() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 bg-white/90 backdrop-blur-sm border-b border-surface-border">
      <div className="max-w-6xl mx-auto px-5 h-16 flex items-center justify-between gap-4">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 focus-visible:ring-2 focus-visible:ring-brand-primary rounded-lg outline-none">
          <div className="w-8 h-8 rounded-lg gradient-brand flex items-center justify-center">
            <Zap size={16} className="text-white" />
          </div>
          <span className="font-bold text-content-primary">SignBridge AI</span>
        </Link>

        {/* Desktop links */}
        <nav className="hidden md:flex items-center gap-6 text-sm text-content-secondary" aria-label="Landing navigation">
          {[["#features","Features"],["#how-it-works","How It Works"],["#vocabulary","Vocabulary"],["#faq","FAQ"]].map(([href, label]) => (
            <a
              key={href}
              href={href}
              className="hover:text-content-primary transition-colors focus-visible:ring-2 focus-visible:ring-brand-primary rounded outline-none px-1"
            >
              {label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Link to="/app" className="hidden md:block">
            <Button size="sm" icon={<ArrowRight size={14} />}>Launch App</Button>
          </Link>

          {/* Mobile hamburger */}
          <button
            className="md:hidden p-2 rounded-xl text-content-secondary hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-brand-primary outline-none"
            onClick={() => setMenuOpen((o) => !o)}
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
          >
            {menuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden border-t border-surface-border bg-white px-5 py-4 space-y-3 animate-slide-up">
          {[["#features","Features"],["#how-it-works","How It Works"],["#vocabulary","Vocabulary"],["#faq","FAQ"]].map(([href, label]) => (
            <a
              key={href}
              href={href}
              className="block text-sm text-content-secondary hover:text-content-primary py-1"
              onClick={() => setMenuOpen(false)}
            >
              {label}
            </a>
          ))}
          <Link to="/app" className="block pt-2" onClick={() => setMenuOpen(false)}>
            <Button className="w-full" icon={<ArrowRight size={14} />}>Launch Application</Button>
          </Link>
        </div>
      )}
    </header>
  );
}

// ── Landing page ───────────────────────────────────────────────────────────

export default function Landing() {
  return (
    <div className="min-h-screen bg-surface-bg">
      <LandingNav />

      {/* ── Hero ── */}
      <section className="relative overflow-hidden">
        {/* Subtle radial glow — respects reduced motion via CSS */}
        <div
          className="absolute inset-0 pointer-events-none select-none"
          aria-hidden="true"
        >
          <div className="absolute top-[-20%] left-1/2 -translate-x-1/2 w-[80vw] h-[60vh] rounded-full bg-gradient-to-br from-indigo-100/60 via-purple-50/40 to-teal-50/30 blur-3xl" />
        </div>

        <div className="relative max-w-6xl mx-auto px-5 pt-20 pb-16 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-brand-primary text-xs font-medium mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-primary animate-pulse-slow" aria-hidden="true" />
            ASL Recognition · Phase 2 · 10 Supported Signs
          </div>

          <h1 className="text-5xl md:text-6xl font-bold text-content-primary leading-tight mb-5">
            Every Gesture<br className="hidden sm:block" />{" "}
            <span className="gradient-brand-text">Deserves a Voice.</span>
          </h1>

          <p className="text-lg md:text-xl text-content-secondary max-w-2xl mx-auto mb-10 leading-relaxed">
            An AI-powered communication assistant that transforms supported sign-language
            gestures into text and speech, helping people communicate more easily.
          </p>

          <div className="flex items-center justify-center gap-3 flex-wrap">
            <Link to="/app/sign">
              <Button size="lg" icon={<Camera size={18} />}>Start Communicating</Button>
            </Link>
            <a href="#features">
              <Button size="lg" variant="secondary" icon={<ChevronRight size={18} />}>
                Explore Features
              </Button>
            </a>
          </div>

          {/* Animated comm bridge hero card */}
          <div className="mt-14 max-w-lg mx-auto">
            <div className="glass-card p-6">
              <p className="text-xs text-content-secondary mb-1 text-left">Live recognition pipeline</p>
              <CommBridge />
              <div className="flex items-center justify-between mt-2 text-xs text-content-secondary">
                <span>Gesture →</span>
                <span>Recognition →</span>
                <span>→ Voice</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section id="features" className="max-w-6xl mx-auto px-5 py-20">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-content-primary mb-3">Everything You Need</h2>
          <p className="text-content-secondary max-w-xl mx-auto">
            A complete communication toolkit built on real computer vision and machine learning —
            no fakes, no static mockups.
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURES.map((f) => (
            <article key={f.title} className="card p-5 hover:shadow-md transition-shadow duration-150">
              <div className="w-10 h-10 rounded-xl bg-indigo-50 text-brand-primary flex items-center justify-center mb-3" aria-hidden="true">
                {f.icon}
              </div>
              <h3 className="font-semibold text-content-primary text-sm mb-1">{f.title}</h3>
              <p className="text-xs text-content-secondary leading-relaxed">{f.desc}</p>
            </article>
          ))}
        </div>
      </section>

      {/* ── How It Works ── */}
      <section id="how-it-works" className="bg-white border-y border-surface-border py-20">
        <div className="max-w-6xl mx-auto px-5">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-content-primary mb-3">How It Works</h2>
            <p className="text-content-secondary max-w-xl mx-auto">
              A genuine computer-vision pipeline. Each step is real software, not a simulation.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
            {PIPELINE.map((p, i) => (
              <div key={p.step} className="relative">
                <div className="card p-4 h-full">
                  <span className="text-2xl font-black gradient-brand-text block mb-2" aria-hidden="true">{p.step}</span>
                  <h3 className="font-semibold text-content-primary text-sm mb-1">{p.label}</h3>
                  <p className="text-xs text-content-secondary leading-relaxed">{p.desc}</p>
                </div>
                {i < PIPELINE.length - 1 && (
                  <ChevronRight
                    size={16}
                    className="hidden sm:block absolute -right-2 top-1/2 -translate-y-1/2 text-surface-border z-10"
                    aria-hidden="true"
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Vocabulary ── */}
      <section id="vocabulary" className="max-w-6xl mx-auto px-5 py-20">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-content-primary mb-3">Supported Vocabulary</h2>
          <p className="text-content-secondary max-w-xl mx-auto">
            Phase 1 supports 10 ASL static handshapes with verified training data.
            Motion-dependent signs (J, Z) will be added once multi-frame tracking is integrated.
          </p>
        </div>
        <div className="flex flex-wrap justify-center gap-3" role="list" aria-label="Supported ASL signs">
          {VOCAB.map((v) => (
            <div
              key={v}
              role="listitem"
              className="w-14 h-14 rounded-2xl bg-white border-2 border-indigo-100 flex items-center justify-center text-2xl font-bold gradient-brand-text shadow-card hover:border-brand-primary hover:shadow-md transition-all duration-150 cursor-default"
              aria-label={`Sign ${v}`}
            >
              {v}
            </div>
          ))}
        </div>
        <p className="text-center text-xs text-content-secondary mt-5">
          ASL · American Sign Language · Static poses only · Phase 1 vocabulary
        </p>
      </section>

      {/* ── Benefits ── */}
      <section className="bg-white border-y border-surface-border py-20">
        <div className="max-w-6xl mx-auto px-5">
          <h2 className="text-3xl font-bold text-content-primary text-center mb-12">
            Built for Real Accessibility
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                icon: <Shield size={20} />,
                iconBg: "bg-teal-50 text-brand-accent",
                title: "Privacy First",
                desc: "Frames are processed in memory and discarded immediately. No video stored. History is opt-in text only.",
              },
              {
                icon: <Clock size={20} />,
                iconBg: "bg-indigo-50 text-brand-primary",
                title: "Real-Time Processing",
                desc: "Landmark extraction and inference complete in well under 200 ms per frame on typical hardware.",
              },
              {
                icon: <Users size={20} />,
                iconBg: "bg-purple-50 text-purple-600",
                title: "Two-Way Communication",
                desc: "Both participants get a dedicated panel — sign recognition on one side, speech recognition on the other.",
              },
            ].map((b) => (
              <div key={b.title} className="flex gap-4">
                <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center ${b.iconBg}`} aria-hidden="true">
                  {b.icon}
                </div>
                <div>
                  <h3 className="font-semibold text-content-primary mb-1">{b.title}</h3>
                  <p className="text-sm text-content-secondary leading-relaxed">{b.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Technology overview ── */}
      <section className="max-w-6xl mx-auto px-5 py-20">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-content-primary mb-3">Technology Stack</h2>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Frontend",   stack: "React 19 · Vite · TypeScript · Tailwind CSS" },
            { label: "Backend",    stack: "FastAPI · Python 3.11 · SQLAlchemy · SQLite" },
            { label: "Vision",     stack: "OpenCV · MediaPipe Hands · 21 landmarks/frame" },
            { label: "ML",         stack: "Scikit-learn · RandomForest · joblib · 63 features" },
          ].map((t) => (
            <div key={t.label} className="card p-4">
              <h3 className="text-xs font-semibold text-content-secondary uppercase tracking-wide mb-2">{t.label}</h3>
              <p className="text-sm text-content-primary leading-relaxed">{t.stack}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── FAQ ── */}
      <section id="faq" className="bg-white border-y border-surface-border py-20">
        <div className="max-w-3xl mx-auto px-5">
          <h2 className="text-3xl font-bold text-content-primary text-center mb-10">
            Frequently Asked Questions
          </h2>
          <div className="space-y-4">
            {FAQS.map((f) => (
              <details key={f.q} className="card p-5 group">
                <summary className="font-semibold text-content-primary text-sm cursor-pointer list-none flex items-center justify-between gap-2 focus-visible:ring-2 focus-visible:ring-brand-primary rounded outline-none">
                  {f.q}
                  <ChevronRight
                    size={16}
                    className="text-content-secondary flex-shrink-0 transition-transform group-open:rotate-90"
                    aria-hidden="true"
                  />
                </summary>
                <p className="text-sm text-content-secondary leading-relaxed mt-3">{f.a}</p>
              </details>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="max-w-6xl mx-auto px-5 py-20 text-center">
        <h2 className="text-3xl font-bold text-content-primary mb-4">
          Ready to Start Communicating?
        </h2>
        <p className="text-content-secondary mb-8 max-w-md mx-auto">
          Launch the application, allow camera access, and begin recognizing your first gestures.
        </p>
        <Link to="/app">
          <Button size="lg" icon={<ArrowRight size={18} />}>Open SignBridge AI</Button>
        </Link>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-surface-border py-10">
        <div className="max-w-6xl mx-auto px-5 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg gradient-brand flex items-center justify-center">
              <Zap size={12} className="text-white" />
            </div>
            <span className="font-bold text-content-primary text-sm">SignBridge AI</span>
          </div>
          <p className="text-xs text-content-secondary text-center max-w-sm">
            Not a replacement for qualified sign-language interpreters or emergency services (112 / 911).
          </p>
          <nav aria-label="Footer navigation" className="flex items-center gap-4 text-xs text-content-secondary">
            <Link to="/app" className="hover:text-content-primary transition-colors">App</Link>
            <a href="#faq" className="hover:text-content-primary transition-colors">FAQ</a>
            <span>Phase 2</span>
          </nav>
        </div>
      </footer>
    </div>
  );
}
