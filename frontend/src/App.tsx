import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { Spinner } from "./components/ui/Card";

// ── Route-level lazy imports (code splitting) ─────────────────────────────
const Landing        = lazy(() => import("./pages/Landing"));
const Dashboard      = lazy(() => import("./pages/Dashboard"));
const SignRecognition = lazy(() => import("./pages/SignRecognition"));
const SpeechToText   = lazy(() => import("./pages/SpeechToText"));
const PracticeMode   = lazy(() => import("./pages/PracticeMode"));
const Emergency      = lazy(() => import("./pages/Emergency"));
const History        = lazy(() => import("./pages/History"));
const Analytics      = lazy(() => import("./pages/Analytics"));
const Settings       = lazy(() => import("./pages/Settings"));

// ── Fallback shown while a lazy chunk loads ────────────────────────────────
function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64" aria-busy="true" aria-label="Loading page…">
      <Spinner size={28} />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageLoader />}>
        <Routes>
          {/* Public landing */}
          <Route path="/" element={<Landing />} />

          {/* Application shell with nested routes */}
          <Route path="/app" element={<AppShell />}>
            <Route index element={<Dashboard />} />
            <Route path="sign"      element={<SignRecognition />} />
            <Route path="speech"    element={<SpeechToText />} />
            <Route path="practice"  element={<PracticeMode />} />
            <Route path="emergency" element={<Emergency />} />
            <Route path="history"   element={<History />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="settings"  element={<Settings />} />
          </Route>

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
