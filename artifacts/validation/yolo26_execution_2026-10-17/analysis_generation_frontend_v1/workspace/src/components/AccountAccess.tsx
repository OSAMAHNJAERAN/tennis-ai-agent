import { useEffect, useState } from 'react';
import { z } from 'zod';
import { apiJson } from '../data/analysisApi';
import { mutate } from '../data/productApi';

const SessionSchema = z.object({ authenticated: z.boolean(), required: z.boolean(), user: z.object({ id: z.string(), username: z.string() }).nullable() });

export function AccountAccess({ onChange }: { onChange?: () => void }) {
  const [session, setSession] = useState<z.infer<typeof SessionSchema> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [register, setRegister] = useState(false);
  const [pending, setPending] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    apiJson('/api/auth/session', controller.signal).then(data => setSession(SessionSchema.parse(data))).catch(() => { /* Older local servers have no accounts. */ });
    return () => controller.abort();
  }, []);
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setPending(true); setError(null);
    try {
      await mutate('/api/auth/' + (register ? 'register' : 'login'), { username, password });
      setSession(SessionSchema.parse(await apiJson('/api/auth/session'))); setPassword(''); onChange?.();
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Sign-in failed.'); }
    finally { setPending(false); }
  }
  if (!session?.required) return null;
  return <section className="upload-account" aria-label="Your account"><h2>{session.authenticated ? `Signed in as ${session.user?.username}` : 'Keep your match private'}</h2>
    {session.authenticated ? <button type="button" className="astra-text-button" disabled={pending} onClick={async () => {
      setPending(true); try { await mutate('/api/auth/logout', {}); setSession({ authenticated: false, required: true, user: null }); onChange?.(); }
      catch { setError('Sign-out failed. Retry to close your session.'); } finally { setPending(false); }
    }}>Sign out</button> : <form onSubmit={event => void submit(event)}><p>Sign in to upload videos and return to your private analyses.</p>
      <label>Username<input value={username} autoComplete="username" required minLength={3} maxLength={64} onChange={event => setUsername(event.target.value)} /></label>
      <label>Password<input type="password" value={password} autoComplete={register ? 'new-password' : 'current-password'} required minLength={12} onChange={event => setPassword(event.target.value)} /></label>
      <button type="submit" className="upload-submit" disabled={pending}>{pending ? 'Connecting…' : register ? 'Create account' : 'Sign in'}</button>
      <button type="button" className="astra-text-button" onClick={() => { setRegister(!register); setError(null); }}>{register ? 'Use existing account' : 'Create an account'}</button>
    </form>}{error ? <p role="alert">{error}</p> : null}</section>;
}
