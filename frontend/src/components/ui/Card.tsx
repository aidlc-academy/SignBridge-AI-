import { clsx } from "clsx";

// ── Card ───────────────────────────────────────────────────────────────────

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  glass?: boolean;
}

export function Card({ glass, className, children, ...props }: CardProps) {
  return (
    <div
      className={clsx(glass ? "glass-card" : "card", "p-5", className)}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={clsx("mb-4", className)} {...props}>
      {children}
    </div>
  );
}

export function CardTitle({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={clsx("text-base font-semibold text-content-primary", className)}
      {...props}
    >
      {children}
    </h3>
  );
}

// ── Badge ──────────────────────────────────────────────────────────────────

type BadgeVariant = "default" | "success" | "warning" | "error" | "info";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

const badgeVariantMap: Record<BadgeVariant, string> = {
  default: "bg-slate-100 text-content-secondary",
  success: "bg-green-100 text-status-success",
  warning: "bg-amber-100 text-status-warning",
  error:   "bg-red-100 text-status-error",
  info:    "bg-indigo-100 text-brand-primary",
};

export function Badge({ variant = "default", className, children, ...props }: BadgeProps) {
  return (
    <span
      className={clsx("badge", badgeVariantMap[variant], className)}
      {...props}
    >
      {children}
    </span>
  );
}

// ── StatusDot ──────────────────────────────────────────────────────────────

type DotStatus = "active" | "idle" | "error" | "warning";

const dotMap: Record<DotStatus, string> = {
  active:  "bg-status-success",
  idle:    "bg-slate-300",
  error:   "bg-status-error",
  warning: "bg-status-warning",
};

export function StatusDot({ status = "idle" }: { status?: DotStatus }) {
  return (
    <span
      className={clsx(
        "inline-block w-2 h-2 rounded-full",
        dotMap[status],
        status === "active" && "animate-pulse-slow"
      )}
      aria-hidden="true"
    />
  );
}

// ── Spinner ────────────────────────────────────────────────────────────────

export function Spinner({ size = 20, className }: { size?: number; className?: string }) {
  return (
    <svg
      className={clsx("animate-spin text-brand-primary", className)}
      width={size}
      height={size}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-label="Loading"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
    </svg>
  );
}
