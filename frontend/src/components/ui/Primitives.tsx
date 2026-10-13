import { clsx } from "clsx";

// ── Skeleton ──────────────────────────────────────────────────────────────

interface SkeletonProps {
  className?: string;
  lines?: number;
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={clsx(
        "animate-pulse bg-slate-200 rounded-lg",
        className
      )}
      aria-hidden="true"
    />
  );
}

export function SkeletonCard({ lines = 3 }: SkeletonProps) {
  return (
    <div className="card p-5 space-y-3" aria-busy="true" aria-label="Loading…">
      <Skeleton className="h-4 w-1/3" />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className={clsx("h-3", i === lines - 1 ? "w-2/3" : "w-full")} />
      ))}
    </div>
  );
}

export function SkeletonText({ lines = 2, className }: SkeletonProps) {
  return (
    <div className={clsx("space-y-2", className)} aria-busy="true">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className={clsx("h-3", i === lines - 1 ? "w-3/4" : "w-full")} />
      ))}
    </div>
  );
}

// ── EmptyState ────────────────────────────────────────────────────────────

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div
      className={clsx(
        "flex flex-col items-center justify-center py-16 px-6 text-center",
        "card bg-slate-50 border-dashed",
        className
      )}
      role="status"
    >
      {icon && (
        <div className="w-14 h-14 rounded-2xl bg-white border border-surface-border flex items-center justify-center text-slate-300 mb-4 shadow-card">
          {icon}
        </div>
      )}
      <h3 className="text-sm font-semibold text-content-primary mb-1">{title}</h3>
      {description && (
        <p className="text-sm text-content-secondary max-w-xs leading-relaxed mb-4">
          {description}
        </p>
      )}
      {action}
    </div>
  );
}

// ── PageHeader ────────────────────────────────────────────────────────────

interface PageHeaderProps {
  title: string;
  description?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  icon?: React.ReactNode;
  iconColor?: string;
}

export function PageHeader({
  title,
  description,
  badge,
  actions,
  icon,
  iconColor = "bg-indigo-50 text-brand-primary",
}: PageHeaderProps) {
  return (
    <div className="flex items-start justify-between gap-4 mb-6 flex-wrap">
      <div className="flex items-start gap-3 min-w-0">
        {icon && (
          <div className={clsx("flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center", iconColor)}>
            {icon}
          </div>
        )}
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-2xl font-bold text-content-primary">{title}</h1>
            {badge}
          </div>
          {description && (
            <p className="text-sm text-content-secondary mt-0.5">{description}</p>
          )}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
    </div>
  );
}

// ── StatusBanner ──────────────────────────────────────────────────────────

interface StatusBannerProps {
  type: "info" | "warning" | "error" | "success";
  children: React.ReactNode;
  className?: string;
}

const bannerMap = {
  info:    "bg-indigo-50 border-indigo-200 text-brand-primary",
  warning: "bg-amber-50 border-amber-200 text-status-warning",
  error:   "bg-red-50 border-red-200 text-status-error",
  success: "bg-green-50 border-green-200 text-status-success",
};

export function StatusBanner({ type, children, className }: StatusBannerProps) {
  return (
    <div
      className={clsx(
        "flex items-start gap-2.5 px-4 py-3 rounded-xl border text-sm font-medium",
        bannerMap[type],
        className
      )}
      role={type === "error" ? "alert" : "status"}
    >
      {children}
    </div>
  );
}

// ── ComingSoon badge ──────────────────────────────────────────────────────

export function ComingSoonBadge({ phase }: { phase: number }) {
  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-content-secondary border border-surface-border">
      Phase {phase}
    </span>
  );
}
