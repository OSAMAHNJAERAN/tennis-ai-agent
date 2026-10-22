import { readPreference, savePreference } from '../lib/preferences';
import { useState } from 'react';
import { useProductResource } from '../hooks/useProductResource';
import { Card, PageHeader } from '../components/ui';
import { ProductError } from './ProductHome';
import { AccountAccess } from '../components/AccountAccess';

export default function ProductSettings() {
  const [storageError, setStorageError] = useState(false);
  const health = useProductResource<{ worker_available: boolean; max_upload_bytes: number; max_duration_seconds: number }>('/api/health', 10000);
  const [view, setView] = useState(() => readPreference('astra.replay.view', 'video'));
  const [rate, setRate] = useState(() => readPreference('astra.replay.rate', '1'));
  return <div className="page product-page">
    <PageHeader eyebrow="WORKSPACE SETTINGS" title="Service and replay" description="Analysis availability and preferences for this browser." />
    <ProductError error={health.error} retry={health.refresh} />
    <AccountAccess onChange={health.refresh} />
    <Card className="product-panel"><h2>Analysis service</h2>
      <p>Connection: {health.error ? 'Unavailable' : health.data ? 'Connected' : 'Checking…'}</p>
      <p>Processing worker: {health.data && !health.error ? health.data.worker_available ? 'Available' : 'Not running' : 'Unknown'}</p>
      {health.data ? <><p>Upload limit: {(health.data.max_upload_bytes / 1024 ** 2).toFixed(0)} MB</p><p>Configured duration ceiling: {health.data.max_duration_seconds / 60} minutes</p><p>MP4, MOV and AVI, where decodable · Singles from a mostly static elevated baseline view</p></> : null}
      <p className="product-hint">The configured ceiling is a validation limit. Full-match processing capacity and independent measurement accuracy are still being qualified.</p>
      <p className="product-hint">Last checked: {health.checkedAt ? new Date(health.checkedAt).toLocaleTimeString() : 'Not yet checked'}</p>
    </Card>
    <Card className="product-panel"><h2>Replay preferences</h2><div className="product-inline">
      <label>Default view<select value={view} onChange={e => { setView(e.target.value); setStorageError(!savePreference('astra.replay.view', e.target.value)); }}><option value="video">Video</option><option value="court">Mini court</option><option value="3d">3D court</option><option value="heatmap">Heatmap</option></select></label>
      <label>Playback speed<select value={rate} onChange={e => { setRate(e.target.value); setStorageError(!savePreference('astra.replay.rate', e.target.value)); }}>{['0.25', '0.5', '1', '1.5', '2'].map(v => <option key={v} value={v}>{v}×</option>)}</select></label>
    </div><p className="product-hint">{storageError ? 'Browser storage is unavailable. Preferences could not be saved.' : 'Saved automatically in this browser. Applied when opening an analysis.'}</p></Card>
  </div>;
}
