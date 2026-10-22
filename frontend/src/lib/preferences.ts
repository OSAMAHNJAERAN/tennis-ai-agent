export function readPreference(key: string, fallback: string): string {
  try { return window.localStorage.getItem(key) ?? fallback; }
  catch { return fallback; }
}

export function savePreference(key: string, value: string): boolean {
  try { window.localStorage.setItem(key, value); return true; }
  catch { return false; }
}
