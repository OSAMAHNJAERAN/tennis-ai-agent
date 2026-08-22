import { lazy, Suspense } from "react";
import { createBrowserRouter } from "react-router-dom";
import { AnalysisProvider } from "./AnalysisContext";
import { AppShell } from "./AppShell";
import RouteError from "./RouteError";
import { SkeletonPage } from "../components/ui";

const Overview = lazy(() => import("../pages/Overview"));
const Matches = lazy(() => import("../pages/Matches"));
const MatchAnalysis = lazy(() => import("../pages/MatchAnalysis"));
const CourtView = lazy(() => import("../pages/CourtView"));
const Players = lazy(() => import("../pages/Players"));
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

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: wrap(<Overview />) },
      { path: "matches", element: wrap(<Matches />) },
      { path: "analysis", element: wrap(<MatchAnalysis />) },
      { path: "court", element: wrap(<CourtView />) },
      { path: "players", element: wrap(<Players />) },
      { path: "ball", element: wrap(<BallTracking />) },
      { path: "events", element: wrap(<Events />) },
      { path: "line-calls", element: wrap(<LineCalls />) },
      { path: "reports", element: wrap(<Reports />) },
      { path: "system", element: wrap(<System />) },
      { path: "*", element: <RouteError /> },
    ],
  },
]);

