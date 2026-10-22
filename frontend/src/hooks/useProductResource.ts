import { useCallback, useEffect, useState } from 'react';
import { productJson } from '../data/productApi';

export function useProductResource<T>(url: string, interval = 0) {
  const [state, setState] = useState<{ url: string; data: T | null; error: string; checkedAt?: string }>({ url: '', data: null, error: '' });
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision(v => v + 1), []);
  const replace = useCallback((data: T) => setState({ url, data, error: '' }), [url]);
  useEffect(() => {
    let active = true; let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try { const data = await productJson<T>(url); if (active) setState({ url, data, error: '', checkedAt: new Date().toISOString() }); }
      catch (error) { if (active) setState(old => ({ url, data: old.url === url ? old.data : null, error: error instanceof Error ? error.message : 'Unable to load records.' })); }
      finally { if (active && interval) timer = setTimeout(load, interval); }
    };
    void load();
    return () => { active = false; clearTimeout(timer); };
  }, [url, interval, revision]);
  return { data: state.url === url ? state.data : null, error: state.url === url ? state.error : '', checkedAt: state.checkedAt, refresh, replace };
}
