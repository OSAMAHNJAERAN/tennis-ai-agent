import * as Dialog from "@radix-ui/react-dialog";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  ChartLineUp,
  Cube,
  Crosshair,
  Gauge,
  Gear,
  PlayCircle,
  SidebarSimple,
  Sparkle,
  Trophy,
  UserFocus,
  X,
} from "@phosphor-icons/react";
import { useState, type ComponentType } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useAnalysis } from "./AnalysisContext";
import { RunPicker } from "../components/RunPicker";
import { DataError, SkeletonPage } from "../components/ui";
import { cn } from "../lib/cn";

type Icon = ComponentType<{ size?: number; weight?: "regular" | "fill" | "duotone" | "bold"; "aria-hidden"?: boolean }>;

const primaryNavigation: Array<{ to: string; label: string; icon: Icon; badge?: string }> = [
  { to: "/overview", label: "Overview", icon: Gauge },
  { to: "/match-center", label: "Match Center", icon: PlayCircle },
  { to: "/court", label: "3D Court", icon: Cube },
  { to: "/analytics", label: "Analytics", icon: ChartLineUp },
  { to: "/players", label: "Players", icon: UserFocus },
  { to: "/tournaments", label: "Tournaments", icon: Trophy },
  { to: "/scouting", label: "Scouting", icon: Crosshair },
  { to: "/agent", label: "Agent", icon: Sparkle, badge: "AI" },
];

const secondaryNavigation: Array<{ to: string; label: string; icon: Icon }> = [
  { to: "/settings", label: "Settings", icon: Gear },
];

function Brand({ collapsed = false }: { collapsed?: boolean }) {
  return (
    <div className="brand">
      <span className="brand-mark" aria-hidden="true">
        <Sparkle size={18} weight="fill" className="brand-sparkle" />
      </span>
      {!collapsed ? (
        <span className="brand-copy">
          <strong>Astra Tennis</strong>
          <span>Court intelligence</span>
        </span>
      ) : null}
    </div>
  );
}

function NavItems({ collapsed = false, onNavigate, search }: { collapsed?: boolean; onNavigate?: () => void; search: string }) {
  return (
    <div className="sidebar-nav-container">
      <nav aria-label="Primary" className="primary-nav">
        <span className="nav-label">{collapsed ? "" : "Main Menu"}</span>
        {primaryNavigation.map(({ to, label, icon: Icon, badge }) => {
          const link = (
            <NavLink
              to={{ pathname: to, search }}
              end={to === "/"}
              onClick={onNavigate}
              className={({ isActive }) => cn("nav-item", isActive && "nav-item-active")}
            >
              <Icon size={19} weight="duotone" aria-hidden={true} />
              {!collapsed ? (
                <div className="nav-text-row">
                  <span>{label}</span>
                  {badge ? <span className="nav-badge-pill">{badge}</span> : null}
                </div>
              ) : null}
            </NavLink>
          );
          return collapsed ? (
            <Tooltip.Root key={to} delayDuration={250}>
              <Tooltip.Trigger asChild>{link}</Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content className="tooltip" side="right" sideOffset={8}>
                  {label}
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          ) : (
            <div key={to}>{link}</div>
          );
        })}
      </nav>

      <div className="sidebar-bottom-nav">
        {secondaryNavigation.map(({ to, label, icon: Icon }) => {
          const link = (
            <NavLink
              to={{ pathname: to, search }}
              onClick={onNavigate}
              className={({ isActive }) => cn("nav-item nav-item-secondary", isActive && "nav-item-active")}
            >
              <Icon size={18} weight="duotone" aria-hidden={true} />
              {!collapsed ? <span>{label}</span> : null}
            </NavLink>
          );
          return collapsed ? (
            <Tooltip.Root key={to} delayDuration={250}>
              <Tooltip.Trigger asChild>{link}</Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content className="tooltip" side="right" sideOffset={8}>
                  {label}
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          ) : (
            <div key={to}>{link}</div>
          );
        })}
      </div>
    </div>
  );
}

export function AppShell() {
  const [collapsed, setCollapsed] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { run, loading, error, retry, selectedRunId, selectedEventId } = useAnalysis();
  const sharedSearch = `?run=${encodeURIComponent(selectedRunId)}${selectedEventId == null ? "" : `&event=${selectedEventId}`}`;

  return (
    <Tooltip.Provider>
      <a className="skip-link" href="#main-content">Skip to workspace</a>
      <div className={cn("app-shell astra-shell", collapsed && "app-shell-collapsed")}>
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
            {!collapsed ? (
              <span>
                <strong>Analysis workspace</strong>
                <small>Single-camera evidence</small>
              </span>
            ) : null}
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
                <Dialog.Content className="mobile-drawer astra-portal" aria-describedby={undefined}>
                  <Dialog.Title className="sr-only">Navigation</Dialog.Title>
                  <div className="drawer-header">
                    <Brand />
                    <Dialog.Close asChild>
                      <button className="icon-button" type="button" aria-label="Close navigation">
                        <X size={20} />
                      </button>
                    </Dialog.Close>
                  </div>
                  <NavItems search={sharedSearch} onNavigate={() => setDrawerOpen(false)} />
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>
            <div className="topbar-context">
              <Sparkle size={18} weight="fill" className="text-lime" aria-hidden="true" />
              <span>
                <strong>{run?.summary.label ?? "Tennis AI Agent"}</strong>
                <small>Session Telemetry</small>
              </span>
            </div>
            <div className="topbar-spacer" />
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
