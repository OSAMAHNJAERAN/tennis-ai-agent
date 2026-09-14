import { lazy, Suspense } from "react";
import { createBrowserRouter, Navigate, useLocation } from "react-router-dom";
import { AnalysisProvider } from "./AnalysisContext";
import { AppShell } from "./AppShell";
import RouteError from "./RouteError";
import { SkeletonPage } from "../components/ui";

const Overview = lazy(() => import("../pages/Overview"));
const Matches = lazy(() => import("../pages/Matches"));
const MatchAnalysis = lazy(() => import("../pages/MatchAnalysis"));
const Analytics = lazy(() => import("../pages/Analytics"));
const CourtView = lazy(() => import("../pages/CourtView"));
const Players = lazy(() => import("../pages/Players"));
const Tournaments = lazy(() => import("../pages/Tournaments"));
const Scouting = lazy(() => import("../pages/Scouting"));
const Agent = lazy(() => import("../pages/Agent"));
const BallTracking = lazy(() => import("../pages/BallTracking"));
const Events = lazy(() => import("../pages/Events"));
const LineCalls = lazy(() => import("../pages/LineCalls"));
const Reports = lazy(() => import("../pages/Reports"));
const System = lazy(() => import("../pages/System"));

function Root() {
  return (
    <AnalysisProvider>
      <AppShell />
    </AnalysisProvider>
  );
}

const wrap = (element: React.ReactNode) => <Suspense fallback={<SkeletonPage />}>{element}</Suspense>;

function LandingRedirect() {
  const { search } = useLocation();
  return <Navigate to={{ pathname: "/match-center", search }} replace />;
}

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <RouteError />,
    children: [
      /* Primary 7 Core Views */
      { index: true, element: <LandingRedirect /> },
      { path: "overview", element: wrap(<Overview />) },
      { path: "match-center", element: wrap(<MatchAnalysis />) },
      { path: "analysis", element: wrap(<MatchAnalysis />) },
      { path: "analytics", element: wrap(<Analytics />) },
      { path: "players", element: wrap(<Players />) },
      { path: "tournaments", element: wrap(<Tournaments />) },
      { path: "scouting", element: wrap(<Scouting />) },
      { path: "agent", element: wrap(<Agent />) },
      { path: "settings", element: wrap(<System />) },

      /* Secondary / Specialized Analysis Views */
      { path: "matches", element: wrap(<Matches />) },
      { path: "court", element: wrap(<CourtView />) },
      { path: "ball", element: wrap(<BallTracking />) },
      { path: "events", element: wrap(<Events />) },
      { path: "line-calls", element: wrap(<LineCalls />) },
      { path: "reports", element: wrap(<Reports />) },
      { path: "system", element: wrap(<System />) },
      { path: "*", element: <RouteError /> },
    ],
  },
]);
