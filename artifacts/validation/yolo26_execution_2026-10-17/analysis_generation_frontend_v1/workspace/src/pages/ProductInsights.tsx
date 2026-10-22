import { Link } from 'react-router-dom';
import { useProductResource } from '../hooks/useProductResource';
import { matchTitle, type ProductMatch } from '../data/productApi';
import { PageHeader, Card } from '../components/ui';
import { ProductError } from './ProductHome';
import { WinProbability } from '../components/WinProbability';

export default function ProductInsights() {
  const resource = useProductResource<ProductMatch[]>('/api/matches', 10000);
  return <div className="page product-page"><PageHeader eyebrow="EVIDENCE FIRST" title="AI Insights" description="Review observations, statistical estimates and your notes in the context of each match." />
    <ProductError error={resource.error} retry={resource.refresh} />
    <Card className="product-panel"><h2>Predictions follow confirmed points</h2><p>Manual scoring provides a statistical estimate from the score and earlier points. Independent calibration remains pending. Video forecasts require qualified point outcomes.</p></Card>
    {!resource.data && !resource.error ? <p role="status">Loading matches…</p> : null}
    {resource.data?.length === 0 ? <p className="product-empty">Upload a match or start a manual score log to build an evidence history.</p> : null}
    {resource.data?.filter(match => match.source === 'manual' || match.status === 'COMPLETED').map(match => <Card className="product-panel" key={match.id}><h2>{matchTitle(match)}</h2>
      <p>{match.source === 'manual' ? `${match.score?.statistics.reduce((n, p) => n + p.points_won, 0) ?? 0} points recorded. Review point and service/return statistics.` : 'Review player positioning observations with the source video and selected frame interval.'}</p>
      {match.source === 'manual' ? <WinProbability match={match} /> : null}
      <p className="product-hint">Source: {match.source === 'manual' ? 'manual score log' : 'recorded video telemetry'} · Updated {new Date(match.updated_at).toLocaleString()}</p>
      <Link className="button" to={`/matches/${match.id}?section=insights`}>Review insights & notes</Link>
    </Card>)}
  </div>;
}
