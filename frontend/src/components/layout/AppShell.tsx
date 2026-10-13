import { useState, useEffect, useRef, useCallback } from "react";
import { NavLink, Outlet, useLocation, Link } from "react-router-dom";
import { clsx } from "clsx";
import {
  LayoutDashboard, Hand, Mic, BookOpen, AlertTriangle,
  History, BarChart2, Settings, ChevronLeft, ChevronRight,
  Zap, Menu, X, ChevronRight as Chevron,
} from "lucide-react";

interface NavItem {
  to: string;
  label: string;
  icon: React.ReactNode;
  end?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/app",           label: "Dashboard",        icon: <LayoutDashboard size={18} />, end: true },
  { to: "/app/sign",      label: "Sign to Text",      icon: <Hand size={18} /> },
  { to: "/app/speech",    label: "Speech to Text",    icon: <Mic size={18} /> },
  { to: "/app/practice",  label: "Practice Mode",     icon: <BookOpen size={18} /> },
  { to: "/app/emergency", label: "Emergency Phrases", icon: <AlertTriangle size={18} /> },
  { to: "/app/history",   label: "History",           icon: <History size={18} /> },
  { to: "/app/analytics", label: "Analytics",         icon: <BarChart2 size={18} /> },
  { to: "/app/settings",  label: "Settings",          icon: <Settings size={18} /> },
];

// Map path segments → human readable label for breadcrumbs
const ROUTE_LABELS: Record<string, string> = {
  app:       "Dashboard",
  sign:      "Sign to Text",
  speech:    "Speech to Text",
  practice:  "Practice Mode",
  emergency: "Emergency Phrases",
  history:   "History",
  analytics: "Analytics",
  settings:  "Settings",
};

function useBreadcrumbs() {
  const { pathname } = useLocation();
  const parts = pathname.split("/").filter(Boolean); // ['app', 'sign']
  return parts.map((seg, i) => ({
    label: ROUTE_LABELS[seg] ?? seg,
    to:    "/" + parts.slice(0, i + 1).join("/"),
    current: i === parts.length - 1,
  }));
}

// ── Sidebar nav link ───────────────────────────────────────────────────────

function SideNavLink({
  item,
  collapsed,
  onClick,
}: {
  item: NavItem;
  collapsed: boolean;
  onClick?: () => void;
}) {
  return (
    <NavLink
      to={item.to}
      end={item.end}
      onClick={onClick}
      className={({ isActive }) =>
        clsx(
          "flex items-center gap-3 px-3 py-2 rounded-xl text-sm font-medium transition-colors duration-150 outline-none",
          "focus-visible:ring-2 focus-visible:ring-brand-primary focus-visible:ring-inset",
          isActive
            ? "bg-indigo-50 text-brand-primary"
            : "text-content-secondary hover:bg-slate-50 hover:text-content-primary"
        )
      }
      title={collapsed ? item.label : undefined}
      aria-label={item.label}
    >
      <span className="flex-shrink-0" aria-hidden="true">{item.icon}</span>
      {!collapsed && <span className="truncate">{item.label}</span>}
    </NavLink>
  );
}

// ── Mobile overlay drawer ──────────────────────────────────────────────────

function MobileDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const firstLinkRef = useRef<HTMLAnchorElement>(null);

  // Trap focus inside drawer when open
  useEffect(() => {
    if (open) {
      firstLinkRef.current?.focus();
    }
  }, [open]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (open) document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Prevent body scroll when open
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [open]);

  return (
    <>
      {/* Backdrop */}
      <div
        className={clsx(
          "fixed inset-0 z-40 bg-black/40 backdrop-blur-sm transition-opacity duration-200",
          open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Drawer panel */}
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
        className={clsx(
          "fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-surface-border",
          "flex flex-col transition-transform duration-200 ease-out",
          open ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Drawer header */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-surface-border">
          <Link to="/" className="flex items-center gap-2" onClick={onClose}>
            <div className="w-8 h-8 rounded-lg gradient-brand flex items-center justify-center">
              <Zap size={16} className="text-white" />
            </div>
            <span className="font-bold text-content-primary text-sm">SignBridge AI</span>
          </Link>
          <button
            onClick={onClose}
            className="p-2 rounded-xl text-content-secondary hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-brand-primary outline-none"
            aria-label="Close navigation menu"
          >
            <X size={18} />
          </button>
        </div>

        {/* Nav links */}
        <nav className="flex-1 overflow-y-auto scrollbar-thin py-3 px-2">
          <ul className="space-y-0.5" role="list">
            {NAV_ITEMS.map((item, i) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  onClick={onClose}
                  ref={i === 0 ? firstLinkRef : undefined}
                  className={({ isActive }) =>
                    clsx(
                      "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors outline-none",
                      "focus-visible:ring-2 focus-visible:ring-brand-primary focus-visible:ring-inset",
                      isActive
                        ? "bg-indigo-50 text-brand-primary"
                        : "text-content-secondary hover:bg-slate-50 hover:text-content-primary"
                    )
                  }
                >
                  <span aria-hidden="true">{item.icon}</span>
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-surface-border">
          <p className="text-xs text-content-secondary">SignBridge AI · Phase 2</p>
        </div>
      </aside>
    </>
  );
}

// ── App header bar ─────────────────────────────────────────────────────────

function AppHeader({
  onMenuClick,
  sidebarCollapsed,
  onToggleSidebar,
}: {
  onMenuClick: () => void;
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
}) {
  const crumbs = useBreadcrumbs();

  return (
    <header className="sticky top-0 z-30 h-14 bg-white/90 backdrop-blur-sm border-b border-surface-border flex items-center px-4 gap-3">
      {/* Mobile hamburger */}
      <button
        onClick={onMenuClick}
        className="md:hidden p-2 rounded-xl text-content-secondary hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-brand-primary outline-none"
        aria-label="Open navigation menu"
        aria-expanded={false}
      >
        <Menu size={20} />
      </button>

      {/* Desktop sidebar toggle */}
      <button
        onClick={onToggleSidebar}
        className="hidden md:flex p-2 rounded-xl text-content-secondary hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-brand-primary outline-none"
        aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
      </button>

      {/* Breadcrumbs */}
      <nav aria-label="Breadcrumb" className="flex items-center gap-1 min-w-0">
        <ol className="flex items-center gap-1 text-sm min-w-0">
          <li>
            <Link
              to="/app"
              className="text-content-secondary hover:text-content-primary transition-colors focus-visible:ring-2 focus-visible:ring-brand-primary rounded outline-none px-1"
            >
              Home
            </Link>
          </li>
          {crumbs.slice(1).map((crumb) => (
            <li key={crumb.to} className="flex items-center gap-1 min-w-0">
              <Chevron size={14} className="text-surface-border flex-shrink-0" aria-hidden="true" />
              {crumb.current ? (
                <span className="font-medium text-content-primary truncate" aria-current="page">
                  {crumb.label}
                </span>
              ) : (
                <Link
                  to={crumb.to}
                  className="text-content-secondary hover:text-content-primary transition-colors focus-visible:ring-2 focus-visible:ring-brand-primary rounded outline-none px-1 truncate"
                >
                  {crumb.label}
                </Link>
              )}
            </li>
          ))}
        </ol>
      </nav>
    </header>
  );
}

// ── App Shell ──────────────────────────────────────────────────────────────

export function AppShell() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { pathname } = useLocation();

  // Close mobile drawer on route change
  useEffect(() => {
    setDrawerOpen(false);
  }, [pathname]);

  const toggleSidebar = useCallback(() => setSidebarCollapsed((c) => !c), []);
  const openDrawer    = useCallback(() => setDrawerOpen(true), []);
  const closeDrawer   = useCallback(() => setDrawerOpen(false), []);

  return (
    <div className="flex h-screen overflow-hidden bg-surface-bg">
      {/* ── Desktop sidebar ── */}
      <aside
        className={clsx(
          "hidden md:flex flex-col flex-shrink-0 h-full bg-white border-r border-surface-border",
          "transition-[width] duration-200 ease-in-out overflow-hidden",
          sidebarCollapsed ? "w-16" : "w-60"
        )}
        aria-label="Application navigation"
      >
        {/* Logo */}
        <div className="flex items-center h-16 px-4 border-b border-surface-border flex-shrink-0">
          <Link to="/" className="flex items-center gap-2 min-w-0 outline-none focus-visible:ring-2 focus-visible:ring-brand-primary rounded-lg">
            <div className="flex-shrink-0 w-8 h-8 rounded-lg gradient-brand flex items-center justify-center">
              <Zap size={16} className="text-white" />
            </div>
            {!sidebarCollapsed && (
              <span className="font-bold text-content-primary text-sm truncate">
                SignBridge AI
              </span>
            )}
          </Link>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto scrollbar-thin py-3 px-2" aria-label="Main navigation">
          <ul className="space-y-0.5" role="list">
            {NAV_ITEMS.map((item) => (
              <li key={item.to}>
                <SideNavLink item={item} collapsed={sidebarCollapsed} />
              </li>
            ))}
          </ul>
        </nav>

        {/* Phase badge */}
        {!sidebarCollapsed && (
          <div className="px-4 py-3 border-t border-surface-border">
            <span className="text-xs text-content-secondary">Phase 2 · ASL 10 signs</span>
          </div>
        )}
      </aside>

      {/* ── Mobile drawer ── */}
      <MobileDrawer open={drawerOpen} onClose={closeDrawer} />

      {/* ── Main area ── */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <AppHeader
          onMenuClick={openDrawer}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={toggleSidebar}
        />
        <main className="flex-1 overflow-y-auto scrollbar-thin" id="main-content" tabIndex={-1}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
