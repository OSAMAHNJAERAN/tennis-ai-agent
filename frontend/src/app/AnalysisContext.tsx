import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useSearchParams } from "react-router-dom";
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
}

const AnalysisContext = createContext<AnalysisContextValue | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedRunId = searchParams.get("run") ?? snapshot.defaultRunId;
  const selectedEventRaw = searchParams.get("event");
  const selectedEventId = selectedEventRaw && /^\d+$/.test(selectedEventRaw) ? Number(selectedEventRaw) : null;
  const [loadState, setLoadState] = useState<{
    id: string;
    runs: AnalysisRunSummary[];
    run: AnalysisRun | null;
    error: Error | null;
  }>({ id: "", runs: [], run: null, error: null });
  const [reloadKey, setReloadKey] = useState(0);
  const loading = loadState.id !== selectedRunId;
  const runs = loadState.runs;
  const run = loading ? null : loadState.run;
  const error = loading ? null : loadState.error;

  useEffect(() => {
    let active = true;
    Promise.all([analysisRepository.listRuns(), analysisRepository.getRun(selectedRunId)])
      .then(([nextRuns, nextRun]) => {
        if (!active) return;
        setLoadState({ id: selectedRunId, runs: nextRuns, run: nextRun, error: null });
      })
      .catch((caught: unknown) => {
        if (!active) return;
        setLoadState({
          id: selectedRunId,
          runs: snapshot.runs,
          run: null,
          error: caught instanceof Error ? caught : new Error("Unable to load analysis data."),
        });
      });
    return () => {
      active = false;
    };
  }, [reloadKey, selectedRunId]);

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
      selectRun: (id) => updateSearch({ run: id, event: null }),
      selectEvent: (id) => updateSearch({ event: id == null ? null : String(id) }),
      retry: () => setReloadKey((key) => key + 1),
    }),
    [error, loading, run, runs, selectedEventId, selectedRunId, updateSearch],
  );

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}

export function useAnalysis() {
  const value = useContext(AnalysisContext);
  if (!value) throw new Error("useAnalysis must be used inside AnalysisProvider.");
  return value;
}
