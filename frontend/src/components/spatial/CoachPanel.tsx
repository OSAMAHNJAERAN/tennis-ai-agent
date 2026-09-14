import { ArrowUpRight, PaperPlaneTilt, Sparkle } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisRun } from "../../data/contracts";
import { buildCoachEvidence, type FrameRange, type NormalizedTelemetry } from "../../lib/tennisTelemetry";

const QUESTIONS = ["How can P1 improve recovery?", "Compare court positioning", "What should I review next?"];

export function CoachPanel({ run, telemetry, range }: { run: AnalysisRun; telemetry: NormalizedTelemetry; range: FrameRange }) {
  const evidence = useMemo(() => buildCoachEvidence(run, telemetry, range), [range, run, telemetry]);
  const [question, setQuestion] = useState(QUESTIONS[0]!);
  const [response, setResponse] = useState<{ text: string; model: string } | null>(null);
  const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => { controller.current?.abort(); controller.current = null; }, []);
  const p1 = evidence.players[0]!; const p2 = evidence.players[1]!;
  const enough = p1.total >= 15 && p2.total >= 15;
  const lead = !enough ? "More tracking is needed." : p1.behindBaseline >= 50 ? "Start with recovery depth." : "Watch the center corridor.";
  const observation = !enough ? "This interval has too few mapped positions for a useful player comparison." : p1.behindBaseline >= 50 ? `P1 spends ${p1.behindBaseline.toFixed(0)}% of tracked frames behind the baseline. Review the recovery position after each contact before deciding whether to step forward.` : `P1 occupies the center corridor in ${p1.center.toFixed(0)}% of tracked frames, compared with ${p2.center.toFixed(0)}% for P2. Review how each player recovers after contact.`;

  async function askCoach(event: React.FormEvent) {
    event.preventDefault(); if (!question.trim() || busy) return;
    controller.current?.abort(); const abort = new AbortController(); controller.current = abort;
    setBusy(true); setError(""); setResponse(null);
    const timeout = window.setTimeout(() => abort.abort(), 45000);
    try {
      const result = await fetch("/api/coach", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: question.trim(), evidence }), signal: abort.signal });
      if (!result.ok) {
        if ([503, 404, 502].includes(result.status)) throw new Error("Astra GPT is not connected. Start the coach service and configure its endpoint, model, and server API key. Local observations remain available.");
        if (result.status === 429) throw new Error("The coach is busy. Please try again shortly.");
        throw new Error("The coach could not complete this request. Please try again.");
      }
      const data: unknown = await result.json();
      if (!data || typeof data !== "object" || !("text" in data) || typeof data.text !== "string" || !("model" in data) || typeof data.model !== "string") throw new Error("The coach returned an invalid response.");
      setResponse({ text: data.text, model: data.model });
    } catch (caught) {
      if (controller.current === abort) setError(abort.signal.aborted ? "The coach request timed out. Please try again." : caught instanceof Error ? caught.message : "Coach connection unavailable.");
    } finally { window.clearTimeout(timeout); if (controller.current === abort) setBusy(false); }
  }

  return <aside className="astra-coach">
    <header><span className="coach-emblem"><Sparkle size={20} weight="fill" /></span><div><h2>Astra coach</h2><span>From movement to meaning</span></div><span className="coach-ai-label">AI</span></header>
    <div className="coach-observation"><div className="coach-source"><span />Local telemetry observation</div><h3>{lead}</h3><p>{observation}</p><div className="coach-evidence-tags"><span>{evidence.range.durationSeconds.toFixed(1)}s interval</span><span>{p1.total + p2.total} samples</span></div></div>
    <div className="coach-next"><span className="astra-eyebrow">Next review</span><p>{enough ? "Compare both players immediately after a contact. Look for open space before attributing a tactical advantage." : "Choose an analysis with player trajectories, then select a longer interval."}</p><ArrowUpRight size={17} /></div>
    <div className="coach-ask"><h3>Ask Astra GPT</h3><p>Send this interval’s telemetry to your configured coach.</p><div className="coach-prompts">{QUESTIONS.map(value => <button key={value} type="button" onClick={() => setQuestion(value)} aria-pressed={question === value}>{value}<ArrowUpRight size={13} /></button>)}</div>
      <form onSubmit={askCoach}><label className="sr-only" htmlFor="coach-question">Question for Astra GPT</label><textarea id="coach-question" maxLength={1000} value={question} onChange={e => setQuestion(e.target.value)} rows={2} /><button className="astra-button coach-submit" type="submit" disabled={busy || !question.trim()}><span>{busy ? "Reading the court…" : "Generate coaching"}</span><PaperPlaneTilt size={16} /></button></form>
      <div aria-live="polite">{error ? <p className="coach-error" role="status">{error}</p> : null}{response ? <article className="coach-response"><span>Astra GPT · {response.model}</span><p>{response.text}</p></article> : null}</div>
    </div>
    <footer>Single-camera evidence. Short intervals describe positioning; they do not establish why a point was won or lost.</footer>
  </aside>;
}
