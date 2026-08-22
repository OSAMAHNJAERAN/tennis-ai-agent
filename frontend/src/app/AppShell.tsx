import * as Dialog from "@radix-ui/react-dialog";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  Article,
  ChartLineUp,
  Circuitry,
  CourtBasketball,
  Gauge,
  ListMagnifyingGlass,
  MagnifyingGlass,
  MapPinLine,
  PlayCircle,
  Rows,
  SidebarSimple,
  TennisBall,
  UserFocus,
  X,
} from "@phosphor-icons/react";
import { useState, type ComponentType } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useAnalysis } from "./AnalysisContext";
import { RunPicker } from "../components/RunPicker";
import { DataError, SkeletonPage } from "../components/ui";
import { cn } from "../lib/cn";

type Icon = ComponentType<{ size?: number; weight?: "regular" | "fill" | "duotone"; "aria-hidden"?: boolean }>;

const navigation: Array<{ to: string; label: string; icon: Icon }> = [
  { to: "/", label: "Overview", icon: Gauge },
  { to: "/matches", label: "Matches", icon: Rows },
  { to: "/analysis", label: "Match analysis", icon: PlayCircle },
  { to: "/court", label: "Court view", icon: CourtBasketball },
  { to: "/players", label: "Players", icon: UserFocus },
  { to: "/ball", label: "Ball tracking", icon: TennisBall },
  { to: "/events", label: "Events", icon: ListMagnifyingGlass },
  { to: "/line-calls", label: "Line calls", icon: MapPinLine },
  { to: "/reports", label: "Reports", icon: Article },
  { to: "/system", label: "System", icon: Circuitry },
];

function Brand({ collapsed = false }: { collapsed?: boolean }) {
  return (
    <div className="brand">
      <span className="brand-mark" aria-hidden="true"><span /></span>
      {!collapsed ? (
        <span className="brand-copy">
          <strong>T88J709</strong>
          <span>Tennis Vision</span>
        </span>
      ) : null}
    </div>
  );
}

function NavItems({ collapsed = false, onNavigate, search }: { collapsed?: boolean; onNavigate?: () => void; search: string }) {
  return (
    <nav aria-label="Primary" className="primary-nav">
      <span className="nav-label">{collapsed ? "" : "Workspace"}</span>
      {navigation.map(({ to, label, icon: Icon }) => {
        const link = (
          <NavLink
            to={{ pathname: to, search }}
            end={to === "/"}
            onClick={onNavigate}
            className={({ isActive }) => cn("nav-item", isActive && "nav-item-active")}
          >
            <Icon size={19} weight="duotone" aria-hidden={true} />
            {!collapsed ? <span>{label}</span> : null}
          </NavLink>
        );
        return collapsed ? (
          <Tooltip.Root key={to} delayDuration={250}>
            <Tooltip.Trigger asChild>{link}</Tooltip.Trigger>
            <Tooltip.Portal><Tooltip.Content className="tooltip" side="right" sideOffset={8}>{label}</Tooltip.Content></Tooltip.Portal>
          </Tooltip.Root>
        ) : <div key={to}>{link}</div>;
      })}
    </nav>
  );
}

export function AppShell() {
  const [collapsed, setCollapsed] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { run, loading, error, retry, selectedRunId, selectedEventId } = useAnalysis();
  const sharedSearch = `?run=${encodeURIComponent(selectedRunId)}${selectedEventId == null ? "" : `&event=${selectedEventId}`}`;
  const isMac = typeof navigator !== "undefined" && /Mac|iPod|iPhone|iPad/.test(navigator.userAgent);

  return (
    <Tooltip.Provider>
      <a className="skip-link" href="#main-content">Skip to workspace</a>
      <div className={cn("app-shell", collapsed && "app-shell-collapsed")}>
        <aside className="desktop-sidebar">
          <div className="sidebar-top">
            <Brand collapsed={collapsed} />
            <button
              type="button"
              className="icon-button sidebar-toggle"
              aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
              onClick={() => setCollapsed((value) => !value)}
            >
              <SidebarSimple size={19} weight="duotone" aria-hidden="true" />
            </button>
          </div>
          <NavItems collapsed={collapsed} search={sharedSearch} />
          <div className="sidebar-foot">
            <span className="live-dot" aria-hidden="true" />
            {!collapsed ? <span><strong>Offline snapshot</strong><small>No runtime network</small></span> : null}
          </div>
        </aside>

        <div className="app-column">
          <header className="topbar">
            <Dialog.Root open={drawerOpen} onOpenChange={setDrawerOpen}>
              <Dialog.Trigger asChild>
                <button className="icon-button mobile-menu" type="button" aria-label="Open navigation">
                  <SidebarSimple size={20} weight="duotone" aria-hidden="true" />
                </button>
              </Dialog.Trigger>
              <Dialog.Portal>
                <Dialog.Overlay className="dialog-overlay" />
                <Dialog.Content className="mobile-drawer" aria-describedby={undefined}>
                  <Dialog.Title className="sr-only">Navigation</Dialog.Title>
                  <div className="drawer-header">
                    <Brand />
                    <Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close navigation"><X size={20} /></button></Dialog.Close>
                  </div>
                  <NavItems search={sharedSearch} onNavigate={() => setDrawerOpen(false)} />
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>
            <div className="topbar-context">
              <ChartLineUp size={18} weight="duotone" aria-hidden="true" />
              <span><strong>{run?.summary.label ?? "Analysis workspace"}</strong><small>Research console</small></span>
            </div>
            <button className="search-button" type="button" disabled title="Snapshot search is not available in this demo">
              <MagnifyingGlass size={17} aria-hidden="true" />
              <span>Search artifacts</span>
              <kbd>{isMac ? "⌘" : "Ctrl"} K</kbd>
            </button>
            <RunPicker />
          </header>
          <main id="main-content" className="main-content" tabIndex={-1}>
            {loading ? <SkeletonPage /> : error ? <DataError message={error.message} onRetry={retry} /> : <Outlet />}
          </main>
        </div>
      </div>
    </Tooltip.Provider>
  );
}
