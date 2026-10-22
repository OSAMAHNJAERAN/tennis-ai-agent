import { safeCreditUrl, type MediaAttribution, type MediaSource } from '../data/publicDemos';

function CreditLink({ url, children }: { url: string; children: string }) {
  const safeUrl = safeCreditUrl(url);
  return safeUrl ? <a href={safeUrl} target="_blank" rel="noopener noreferrer">{children}</a> : <span>{children}</span>;
}

export function MediaSourceCredit({ source, expanded = false }: { source: MediaAttribution | MediaSource; expanded?: boolean }) {
  const changes = Array.isArray(source.modifications) ? source.modifications.join('; ') : source.modifications;
  const excerpt = 'excerpt' in source ? source.excerpt : null;
  return <div className={`product-media-credit${expanded ? ' product-media-credit-expanded' : ''}`}>
    <p>Video by {source.author} · <CreditLink url={source.sourcePage}>Original source</CreditLink> · <CreditLink url={source.licenseUrl}>{source.license}</CreditLink></p>
    {changes ? <p>{changes}</p> : null}
    {source.derivativeLicense ? <p>Annotated version: {source.derivativeLicense}</p> : null}
    {expanded && excerpt ? <p>Source excerpt: {excerpt.startSeconds.toFixed(2)}–{excerpt.endSeconds.toFixed(2)} seconds. Playback starts at the beginning of this excerpt.</p> : null}
  </div>;
}
