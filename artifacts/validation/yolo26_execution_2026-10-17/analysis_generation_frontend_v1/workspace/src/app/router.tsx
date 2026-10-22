import { lazy, Suspense } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AppShell } from './AppShell';
import RouteError from './RouteError';
import { SkeletonPage } from '../components/ui';
import ProductHome, { MatchCenter } from '../pages/ProductHome';
const LegacyAnalysis = lazy(() => import('../pages/ProductWorkspace').then(module => ({ default: module.LegacyAnalysis })));
const ProductWorkspace = lazy(() => import('../pages/ProductWorkspace'));
const Upload = lazy(() => import('../pages/Upload'));
const ProductSettings = lazy(() => import('../pages/ProductSettings'));
const ProductExamples = lazy(() => import('../pages/ProductExamples'));
const wrap = (element: React.ReactNode) => <Suspense fallback={<SkeletonPage label="Loading workspace" />}>{element}</Suspense>;
export const router = createBrowserRouter([{ path: '/', element: <AppShell />, errorElement: <RouteError />, children: [
  { index: true, element: <Navigate to="/overview" replace /> },
  { path: 'overview', element: <ProductHome /> },
  { path: 'match-center', element: <MatchCenter /> },
  { path: 'matches', element: <Navigate to="/match-center" replace /> },
  { path: 'matches/:matchId', element: wrap(<ProductWorkspace />) },
  { path: 'upload', element: wrap(<Upload />) },
  ...['live/new', 'players', 'players/:playerId', 'insights', 'agent', 'scouting'].map(path => ({ path, element: <Navigate to="/match-center" replace /> })),
  { path: 'settings', element: wrap(<ProductSettings />) },
  { path: 'system', element: <Navigate to="/settings" replace /> },
  { path: 'examples', element: wrap(<ProductExamples />) },
  { path: 'examples/:exampleId', element: wrap(<ProductExamples />) },
  { path: 'tournaments', element: <Navigate to="/match-center" replace /> },
  ...['analysis', 'analysis/:analysisId', 'analytics', 'court', 'ball', 'events', 'line-calls', 'reports'].map(path => ({ path, element: wrap(<LegacyAnalysis />) })),
  { path: '*', element: <RouteError /> },
]}]);
