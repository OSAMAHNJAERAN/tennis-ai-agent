import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useMatch, useSearchParams } from "react-router-dom";
import type { AnalysisRun, AnalysisRunSummary } from "../data/contracts";
import { analysisRepository, snapshot } from "../data/repository";

interface AnalysisContextValue {
  runs: AnalysisRunSummary[];
  run: AnalysisRun | null;
  selectedRunId: string;
  selectedEventId: number | null;
  loading: boolean;
  error: Error | null;
  selectRun: (id: string) => void;
  selectEvent: (id: number | null) => void;
  retry: () => void;
  reloadGeneration: (generation: string) => void;
  reloadCurrent: () => void;
  correctionsReadOnly: boolean;
}

const AnalysisContext = createContext<AnalysisContextValue | null>(null);

export function AnalysisProvider({ children, runId, readOnly = false }: { children: ReactNode; runId?: string; readOnly?: boolean }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const match = useMatch('/analysis/:analysisId');
  const selectedRunId = runId ?? match?.params.analysisId ?? searchParams.get("run") ?? snapshot.defaultRunId;
  const selectedEventRaw = searchParams.get("event");
  const generation = searchParams.get('generation') ?? undefined;
  const selectionKey = `${selectedRunId}:${generation ?? ''}`;
  const selectedEventId = selectedEventRaw && /^\d+$/.test(selectedEventRaw) ? Number(selectedEventRaw) : null;
  const [loadState, setLoadState] = useState<{
    id: string;
    runs: AnalysisRunSummary[];
    run: AnalysisRun | null;
    error: Error | null;
  }>({ id: "", runs: [], run: null, error: null });
  const [reloadKey, setReloadKey] = useState(0);
  const loading = loadState.id !== selectionKey;
  const runs = loadState.runs;
  const run = loading ? null : loadState.run;
  const error = loading ? null : loadState.error;

  useEffect(() => {
    let active = true;
    Promise.all([analysisRepository.listRuns(), generation ? analysisRepository.getRun(selectedRunId, generation) : analysisRepository.getRun(selectedRunId)])
      .then(([nextRuns, nextRun]) => {
        if (!active) return;
        setLoadState({ id: selectionKey, runs: nextRuns, run: nextRun, error: null });
      })
      .catch((caught: unknown) => {
        if (!active) return;
        setLoadState({
          id: selectionKey,
          runs: snapshot.runs,
          run: null,
          error: caught instanceof Error ? caught : new Error("Unable to load analysis data."),
        });
      });
    return () => {
      active = false;
    };
  }, [reloadKey, selectedRunId, generation, selectionKey]);

  const updateSearch = useCallback(
    (updates: Record<string, string | null>) => {
      setSearchParams((current) => {
        const next = new URLSearchParams(current);
        for (const [key, value] of Object.entries(updates)) {
          if (value == null) next.delete(key);
          else next.set(key, value);
        }
        return next;
      });
    },
    [setSearchParams],
  );

  const value = useMemo<AnalysisContextValue>(
    () => ({
      runs,
      run,
      selectedRunId,
      selectedEventId,
      loading,
      error,
      selectRun: (id) => updateSearch({ run: id, event: null, generation: null }),
      selectEvent: (id) => updateSearch({ event: id == null ? null : String(id) }),
      retry: () => setReloadKey((key) => key + 1),
      reloadGeneration: (token) => { if (/^[a-f0-9]{64}$/.test(token)) updateSearch({ generation: token, event: null }); },
      reloadCurrent: () => { updateSearch({ generation: null, event: null }); setReloadKey(key => key + 1); },
      correctionsReadOnly: readOnly,
    }),
    [error, loading, run, runs, selectedEventId, selectedRunId, updateSearch, readOnly],
  );

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}

export function useAnalysis() {
  const value = useContext(AnalysisContext);
  if (!value) throw new Error("useAnalysis must be used inside AnalysisProvider.");
  return value;
}
