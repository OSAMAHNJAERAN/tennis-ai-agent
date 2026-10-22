import { Link, useParams, useSearchParams } from 'react-router-dom';
import { AnalysisProvider, useAnalysis } from '../app/AnalysisContext';
import { MediaSourceCredit } from '../components/MediaSourceCredit';
import { DataError, PageHeader, SkeletonPage } from '../components/ui';
import { usePublicDemos } from '../hooks/usePublicDemos';
import { PublicDemoGallery } from './ProductHome';
import SpatialAnalysis from './SpatialAnalysis';
import SessionAnalytics from './LiveRunAnalytics';
import Reports from './Reports';

function ExampleAnalysis() {
  const { run, loading, error, retry } = useAnalysis();
  const [params, setParams] = useSearchParams();
  const requested = params.get('section');
  const section = requested === 'statistics' || requested === 'report' ? requested : 'replay';
  if (loading) return <SkeletonPage />;
  if (error || !run) return <DataError message={error?.message ?? 'This example analysis is unavailable.'} onRetry={retry} />;
  return <><nav className="product-workspace-tabs" aria-label="Example sections">{[['replay', 'Replay'], ['statistics', 'Movement & tracking'], ['report', 'Report & exports']].map(([value, title]) => <button className="button" key={value} aria-pressed={section === value} onClick={() => setParams(current => { const next = new URLSearchParams(current); next.set('section', value!); return next; })}>{title}</button>)}</nav><section className="product-workspace-section" hidden={section !== 'replay'}><SpatialAnalysis active={section === 'replay'} /></section>{section === 'statistics' ? <SessionAnalytics /> : null}{section === 'report' ? <Reports /> : null}</>;
}

export default function ProductExamples() {
  const { exampleId } = useParams();
  const catalog = usePublicDemos();
  const selected = catalog.demos.find(demo => demo.id === exampleId);
  return <div className="page product-page"><Link className="product-breadcrumb" to="/overview">← Home</Link>{catalog.loading ? <SkeletonPage label="Loading example analyses" /> : catalog.error ? <DataError message={catalog.error} onRetry={catalog.refresh} /> : exampleId ? selected ? <><PageHeader eyebrow="EXAMPLE ANALYSIS" title={selected.title} description="A real processed tennis clip. Review its visible evidence, uncertain estimates, and downloadable results." /><MediaSourceCredit source={{ ...selected.attribution, excerpt: selected.excerpt }} expanded /><p className="product-demo-note">This demonstration is separate from your uploaded videos. Its measurements have not been independently validated.</p><AnalysisProvider key={selected.analysisId} runId={selected.analysisId}><ExampleAnalysis /></AnalysisProvider><Link className="product-breadcrumb" to="/examples">View all examples</Link></> : <><h1>Example unavailable</h1><p className="product-empty">This clip is not in the current example collection. <Link to="/examples">View available examples</Link>.</p></> : <><PageHeader eyebrow="TENNIS VIDEO EXAMPLES" title="Explore real analyzed play" description="Each clip has its own processed video, saved analysis, and source credit." />{catalog.demos.length ? <PublicDemoGallery demos={catalog.demos} /> : <p className="product-empty">Public example videos are not available yet. <Link to="/upload">Upload your own video</Link> to explore the analysis.</p>}<p className="product-demo-note">Example outputs demonstrate the upload-to-report experience. They are not independent evidence of measurement accuracy.</p></>}</div>;
}
