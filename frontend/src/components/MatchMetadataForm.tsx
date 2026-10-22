import { useState } from 'react';
import { useProductResource } from '../hooks/useProductResource';
import { mutate, type MatchMetadata, type PlayerProfile } from '../data/productApi';

export function MatchMetadataForm({ value, onChange, disabled = false }: { value: MatchMetadata; onChange: (value: MatchMetadata) => void; disabled?: boolean }) {
  const profiles = useProductResource<PlayerProfile[]>('/api/players');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  function participant(index: number, name: string | null, profile_id: string | null) {
    onChange({ ...value, role_assignment: 'unconfirmed', participants: value.participants.map((p, i) => i === index ? { name, profile_id } : p) });
  }
  async function create(index: number) {
    const name = value.participants[index]?.name;
    if (!name?.trim()) return;
    setSaving(true); setError('');
    try { const profile = await mutate<PlayerProfile>('/api/players', { name }); participant(index, profile.name, profile.id); profiles.refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : 'Unable to create profile.'); }
    finally { setSaving(false); }
  }
  return <fieldset className="product-fields" disabled={disabled || saving}><legend>Match details <span className="text-muted">· optional</span></legend>
    {value.participants.map((p, index) => <div className="participant-field" key={index}><label>Player {index + 1}<input maxLength={120} placeholder="Name unknown" value={p.name ?? ''} disabled={Boolean(p.profile_id)} onChange={e => participant(index, e.target.value || null, null)} /></label>
      <label>Saved profile<select value={p.profile_id ?? ''} onChange={e => { const selected = profiles.data?.find(item => item.id === e.target.value); participant(index, selected?.name ?? p.name, selected?.id ?? null); }}><option value="">No linked profile</option>{profiles.data?.map(profile => <option key={profile.id} value={profile.id}>{profile.name} · {profile.id.slice(-6)}</option>)}</select></label>
      {!p.profile_id && p.name?.trim() ? <button type="button" className="astra-text-button" onClick={() => void create(index)}>Create a new profile for {p.name}</button> : null}</div>)}
    <label>Tournament<input maxLength={160} value={value.tournament ?? ''} placeholder="Optional" onChange={e => onChange({ ...value, tournament: e.target.value || null })} /></label>
    <label>Match date<input type="date" value={value.match_date ?? ''} onChange={e => onChange({ ...value, match_date: e.target.value || null })} /></label>
    <label>Match type<select value="singles" onChange={() => undefined}><option value="singles">Singles</option><option value="doubles" disabled>Doubles — video analysis not supported</option></select></label>
    <p className="product-hint">Unknown names stay unknown. No identity source is connected; names you enter are user-confirmed. Linking a profile is a separate, explicit choice.</p>
    {error || profiles.error ? <p role="alert">{error || profiles.error}</p> : null}
  </fieldset>;
}
