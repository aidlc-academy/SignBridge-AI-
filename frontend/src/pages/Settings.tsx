import { Settings as SettingsIcon, Volume2, Eye, Shield, Sliders } from "lucide-react";
import { Card, CardHeader, CardTitle } from "../components/ui/Card";
import { PageHeader } from "../components/ui/Primitives";
import { useSettingsStore } from "../store";
import { useTTS } from "../hooks/useSpeech";

function SettingsSection({
  icon, title, children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <span className="text-content-secondary" aria-hidden="true">{icon}</span>
          <CardTitle>{title}</CardTitle>
        </div>
      </CardHeader>
      <div className="space-y-5">{children}</div>
    </Card>
  );
}

function RangeField({
  label, description, min, max, step, value, onChange, displayValue,
}: {
  label: string;
  description?: string;
  min: number; max: number; step: number;
  value: number;
  onChange: (v: number) => void;
  displayValue: string;
}) {
  return (
    <label className="block">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-content-primary">{label}</span>
        <span className="text-sm font-semibold text-brand-primary">{displayValue}</span>
      </div>
      {description && <p className="text-xs text-content-secondary mb-2">{description}</p>}
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-brand-primary cursor-pointer"
        aria-label={label}
      />
      <div className="flex justify-between text-xs text-content-secondary mt-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </label>
  );
}

function ToggleField({
  label, description, checked, onChange,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer">
      <div className="relative flex-shrink-0 mt-0.5">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only"
        />
        <div
          className={`w-10 h-5 rounded-full transition-colors duration-200 ${checked ? "bg-brand-primary" : "bg-slate-200"}`}
          aria-hidden="true"
        >
          <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${checked ? "translate-x-5" : ""}`} />
        </div>
      </div>
      <div>
        <p className="text-sm font-medium text-content-primary">{label}</p>
        {description && <p className="text-xs text-content-secondary mt-0.5">{description}</p>}
      </div>
    </label>
  );
}

export default function Settings() {
  const { settings, updateSettings } = useSettingsStore();
  const { voices } = useTTS();

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-5">
      <PageHeader
        title="Settings"
        description="Configure recognition, speech, privacy, and accessibility preferences."
        icon={<SettingsIcon size={20} />}
        iconColor="bg-slate-100 text-content-secondary"
      />

      {/* Recognition */}
      <SettingsSection icon={<Sliders size={16} />} title="Recognition">
        <RangeField
          label="Confidence Threshold"
          description="Predictions below this value are rejected. Lower = more permissive; higher = more strict."
          min={0.5} max={0.99} step={0.01}
          value={settings.confidenceThreshold}
          onChange={(v) => updateSettings({ confidenceThreshold: v })}
          displayValue={settings.confidenceThreshold.toFixed(2)}
        />
      </SettingsSection>

      {/* Speech */}
      <SettingsSection icon={<Volume2 size={16} />} title="Text-to-Speech">
        <label className="block">
          <span className="text-sm font-medium text-content-primary block mb-1">Voice</span>
          <p className="text-xs text-content-secondary mb-2">
            Available voices depend on your operating system and browser.
            Tamil voices are rare and not guaranteed.
          </p>
          <select
            value={settings.voiceName}
            onChange={(e) => updateSettings({ voiceName: e.target.value })}
            className="w-full text-sm border border-surface-border rounded-xl px-3 py-2.5 bg-white text-content-primary focus:ring-2 focus:ring-brand-primary focus:outline-none"
            aria-label="Select TTS voice"
          >
            <option value="">System default</option>
            {voices.map((v) => (
              <option key={v.name} value={v.name}>
                {v.name} ({v.lang})
              </option>
            ))}
          </select>
          {voices.length === 0 && (
            <p className="text-xs text-content-secondary mt-1">No voices detected. Browser may load them asynchronously.</p>
          )}
        </label>

        <RangeField
          label="Speech Rate"
          description="How fast the voice speaks. 1.0× is normal speed."
          min={0.5} max={2} step={0.1}
          value={settings.speechRate}
          onChange={(v) => updateSettings({ speechRate: v })}
          displayValue={`${settings.speechRate.toFixed(1)}×`}
        />
      </SettingsSection>

      {/* Privacy */}
      <SettingsSection icon={<Shield size={16} />} title="Privacy">
        <ToggleField
          label="Save conversation history"
          description="When enabled, confirmed messages can be saved to local history. No raw frames or audio are ever stored."
          checked={settings.saveHistoryEnabled}
          onChange={(v) => updateSettings({ saveHistoryEnabled: v })}
        />
      </SettingsSection>

      {/* Accessibility */}
      <SettingsSection icon={<Eye size={16} />} title="Accessibility">
        <ToggleField
          label="Reduce motion"
          description="Disables animations and transitions. The app also respects your OS reduced-motion preference automatically."
          checked={settings.reducedMotion}
          onChange={(v) => updateSettings({ reducedMotion: v })}
        />
      </SettingsSection>

      <p className="text-xs text-content-secondary text-center">
        Settings are saved in your browser's local storage and persist between sessions.
      </p>
    </div>
  );
}
