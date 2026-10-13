import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Camera, Mic, BookOpen, AlertTriangle, TrendingUp, MessageCircle, Layers, Target } from "lucide-react";
import { Card, CardHeader, CardTitle, Badge } from "../components/ui/Card";
import { Skeleton, StatusBanner } from "../components/ui/Primitives";
import { useConversationStore, useRecognitionStore, useAnalyticsStore } from "../store";
import { getHealth, getModelStatus } from "../services/api";

// ── Stat card ──────────────────────────────────────────────────────────────

function StatCard({
  label, value, description, icon, iconBg, loading,
}: {
  label: string;
  value: string | number;
  description: string;
  icon: React.ReactNode;
  iconBg: string;
  loading?: boolean;
}) {
  return (
    <Card className="flex items-start gap-4">
      <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center ${iconBg}`} aria-hidden="true">
        {icon}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-content-secondary uppercase tracking-wide mb-1">{label}</p>
        {loading ? (
          <Skeleton className="h-7 w-16 mb-1" />
        ) : (
          <p className="text-3xl font-bold text-content-primary leading-none mb-1">{value}</p>
        )}
        <p className="text-xs text-content-secondary">{description}</p>
      </div>
    </Card>
  );
}

// ── Quick action card ──────────────────────────────────────────────────────

function QuickAction({
  to, icon, iconBg, label, desc,
}: {
  to: string;
  icon: React.ReactNode;
  iconBg: string;
  label: string;
  desc: string;
}) {
  return (
    <Link to={to} className="block group outline-none focus-visible:ring-2 focus-visible:ring-brand-primary rounded-2xl">
      <Card className="flex items-start gap-4 hover:shadow-md transition-shadow duration-150 cursor-pointer h-full">
        <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center ${iconBg}`} aria-hidden="true">
          {icon}
        </div>
        <div>
          <p className="font-semibold text-content-primary text-sm group-hover:text-brand-primary transition-colors">
            {label}
          </p>
          <p className="text-xs text-content-secondary mt-0.5">{desc}</p>
        </div>
      </Card>
    </Link>
  );
}

// ── Dashboard ──────────────────────────────────────────────────────────────

export default function Dashboard() {
  const { conversations, activeSessionMessages } = useConversationStore();
  const { currentMessage } = useRecognitionStore();
  const { attempts, accepted } = useAnalyticsStore();

  const [backendStatus, setBackendStatus] = useState<"checking" | "ok" | "error">("checking");
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);

  useEffect(() => {
    // Check backend health non-blocking
    getHealth()
      .then(() => setBackendStatus("ok"))
      .catch(() => setBackendStatus("error"));

    getModelStatus()
      .then((s) => setModelLoaded(s.loaded))
      .catch(() => setModelLoaded(false));
  }, []);

  const acceptanceRate = attempts > 0 ? Math.round((accepted / attempts) * 100) : null;

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-content-primary mb-0.5">Dashboard</h1>
        <p className="text-sm text-content-secondary">
          Select an action below to begin communicating.
        </p>
      </div>

      {/* Backend / model status banners */}
      {backendStatus === "error" && (
        <StatusBanner type="error">
          Backend is not reachable. Start the FastAPI server:{" "}
          <code className="font-mono text-xs">uvicorn app.main:app --reload --port 8000</code>
        </StatusBanner>
      )}
      {backendStatus === "ok" && modelLoaded === false && (
        <StatusBanner type="warning">
          Recognition model not trained yet. Sign recognition will return "model unavailable" until you train a model in Phase 5.
        </StatusBanner>
      )}

      {/* Stats grid */}
      <section aria-label="Session statistics">
        <h2 className="text-xs font-semibold text-content-secondary uppercase tracking-wide mb-3">Overview</h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Saved Conversations"
            value={conversations.length}
            description="Stored in history"
            icon={<MessageCircle size={18} />}
            iconBg="bg-indigo-50 text-brand-primary"
          />
          <StatCard
            label="Supported Signs"
            value={10}
            description="ASL static gestures"
            icon={<Layers size={18} />}
            iconBg="bg-teal-50 text-brand-accent"
          />
          <StatCard
            label="Recognition Attempts"
            value={attempts}
            description="This session only"
            icon={<TrendingUp size={18} />}
            iconBg="bg-amber-50 text-status-warning"
          />
          <StatCard
            label="Acceptance Rate"
            value={acceptanceRate !== null ? `${acceptanceRate}%` : "—"}
            description={attempts > 0 ? `${accepted} of ${attempts} accepted` : "No attempts yet"}
            icon={<Target size={18} />}
            iconBg="bg-green-50 text-status-success"
          />
        </div>
      </section>

      {/* Active session preview */}
      {(activeSessionMessages.length > 0 || currentMessage) && (
        <Card className="border-brand-primary/20 bg-gradient-to-r from-indigo-50/50 to-teal-50/30">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Active Session</CardTitle>
              <Badge variant="info">Live</Badge>
            </div>
          </CardHeader>
          {currentMessage && (
            <p className="text-sm text-content-primary">
              Current message:{" "}
              <span className="font-semibold gradient-brand-text">{currentMessage}</span>
            </p>
          )}
          {activeSessionMessages.length > 0 && (
            <p className="text-xs text-content-secondary mt-1">
              {activeSessionMessages.length} message{activeSessionMessages.length !== 1 ? "s" : ""} in session
            </p>
          )}
        </Card>
      )}

      {/* Quick actions */}
      <section aria-label="Quick actions">
        <h2 className="text-xs font-semibold text-content-secondary uppercase tracking-wide mb-3">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <QuickAction to="/app/sign"      icon={<Camera size={18} />}        iconBg="bg-indigo-50 text-brand-primary" label="Start Sign Recognition"   desc="Open webcam and begin recognizing gestures" />
          <QuickAction to="/app/speech"    icon={<Mic size={18} />}           iconBg="bg-teal-50 text-brand-accent"   label="Start Speech Recognition"  desc="Use your microphone to capture spoken response" />
          <QuickAction to="/app/practice"  icon={<BookOpen size={18} />}      iconBg="bg-amber-50 text-status-warning" label="Practice Gestures"         desc="Rehearse supported signs with live feedback" />
          <QuickAction to="/app/emergency" icon={<AlertTriangle size={18} />} iconBg="bg-red-50 text-status-error"    label="Emergency Phrases"         desc="Quick-access cards for critical communication" />
        </div>
      </section>

      {/* Phase roadmap */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <CardTitle>Development Phases</CardTitle>
            <Badge variant="info">Phase 2</Badge>
          </div>
        </CardHeader>
        <div className="space-y-2">
          {[
            { phase: 1, label: "Setup & Architecture",              status: "done" },
            { phase: 2, label: "Frontend Polish & Navigation",      status: "active" },
            { phase: 3, label: "Camera Integration",                status: "next" },
            { phase: 4, label: "OpenCV + MediaPipe Hand Tracking",  status: "future" },
            { phase: 5, label: "Dataset Collection + Training",     status: "future" },
            { phase: 6, label: "Real-Time Recognition",             status: "future" },
          ].map((p) => (
            <div key={p.phase} className="flex items-center gap-3">
              <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                p.status === "done"   ? "bg-status-success" :
                p.status === "active" ? "bg-brand-primary animate-pulse-slow" :
                p.status === "next"   ? "bg-status-warning" :
                "bg-slate-200"
              }`} aria-hidden="true" />
              <span className={`text-sm ${p.status === "active" ? "font-semibold text-content-primary" : "text-content-secondary"}`}>
                Phase {p.phase} — {p.label}
              </span>
              {p.status === "done" && <span className="text-xs text-status-success ml-auto">✓ Complete</span>}
              {p.status === "active" && <span className="text-xs text-brand-primary ml-auto font-medium">In Progress</span>}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
