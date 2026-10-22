import { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDown, ArrowRight, ArrowUpRight, Pause, Play } from '@phosphor-icons/react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';
import { useReducedMotion } from '../hooks/useReducedMotion';

gsap.registerPlugin(ScrollTrigger, useGSAP);

export function CourtEmblem({ compact = false }: { compact?: boolean }) {
  return <svg className={`society-court-emblem${compact ? ' is-compact' : ''}`} viewBox="0 0 400 440" fill="none" aria-hidden="true">
    <g className="emblem-orbit" stroke="currentColor" strokeWidth="1"><circle cx="200" cy="220" r="171" /><circle cx="200" cy="220" r="181" strokeDasharray="1 10" /></g>
    <g stroke="currentColor" strokeWidth="1.4"><rect x="108" y="71" width="184" height="298" /><path d="M131 71V369M269 71V369M108 220H292M131 150H269M131 290H269M200 150V290" /><path d="M200 61V71M200 369V379M98 220H108M292 220H302" /></g>
    <circle className="emblem-ball" cx="200" cy="220" r="55" fill="var(--society-lime)" />
    <g stroke="var(--society-blue)" strokeWidth="2"><path d="M160 180C192 180 195 207 196 221C197 249 217 261 240 261M240 179C209 179 207 207 206 220C205 249 183 261 160 261" /></g>
    <path className="emblem-path" d="M143 332C118 235 282 260 253 109" stroke="var(--society-lime)" strokeWidth="1.5" strokeDasharray="5 6" />
    <g fill="currentColor"><rect x="139" y="328" width="8" height="8" /><rect x="249" y="105" width="8" height="8" /><path d="M43 38h16v4H43zM49 32h4v16h-4zM341 394h16v4h-16zM347 388h4v16h-4z" /></g>
  </svg>;
}

export function SocietyHero() {
  return <header className="society-hero">
    <div className="society-hero-meta"><span>[ AI TENNIS INTELLIGENCE ]</span><span>THE COURT IS JUST THE BEGINNING.</span></div>
    <div className="society-hero-composition">
      <div className="society-hero-copy">
        <h1><span className="society-reveal-line"><span>See the game.</span></span><span className="society-reveal-line"><span>See it <em>differently.</em></span></span></h1>
        <div className="society-hero-bottom"><p>Your footage. A new perspective.<br />Follow the ball, read the movement, and discover the evidence behind your game.</p><div className="society-hero-actions"><Link className="society-cta" to="/upload"><span>Analyze your match</span><ArrowUpRight size={22} aria-hidden="true" /></Link><Link className="society-text-link" to="/examples">Explore the replay <ArrowRight size={18} aria-hidden="true" /></Link></div></div>
      </div>
      <div className="society-hero-art"><div className="society-art-caption"><span>COURT / COMPUTER VISION</span><span>+</span></div><CourtEmblem /><div className="society-art-footer"><span>A different angle on every rally.</span><span>[ ASTRA ]</span></div></div>
    </div>
    <a className="society-scroll-cue" href="#match-intelligence"><ArrowDown size={16} aria-hidden="true" /><span>Discover the detail</span><span>SCROLL TO EXPLORE</span></a>
  </header>;
}

export function SocietyMarquee({ paused, onToggle }: { paused: boolean; onToggle: () => void }) {
  const items = ['PLAYER MOVEMENT', 'BALL TRACKING', 'COURT PERSPECTIVE', 'MATCH EVIDENCE'];
  return <div className="society-marquee" aria-label="Player movement, ball tracking, court perspective, match evidence"><div aria-hidden="true">{[0, 1].map(copy => <span key={copy}>{items.map(item => <span key={item}>{item}<i>+</i></span>)}</span>)}</div><button type="button" className="society-motion-toggle" aria-pressed={paused} aria-label={paused ? 'Resume moving graphics' : 'Pause moving graphics'} onClick={onToggle}>{paused ? <Play size={14} aria-hidden="true" /> : <Pause size={14} aria-hidden="true" />}<span>{paused ? 'Resume' : 'Pause'}</span></button></div>;
}

const chapters = [
  { title: 'Bring your game.', label: 'THE FOOTAGE', detail: 'Start with a recorded singles match. A clear, mostly static camera gives the system the best view of the court and both players.', href: '/upload', action: 'Upload your video' },
  { title: 'Change your perspective.', label: 'THE REPLAY', detail: 'Move between the original footage, annotated video, and available court views. Step through the frames and inspect what the tracking actually sees.', href: '/examples', action: 'Explore an analysis' },
  { title: 'Keep the evidence.', label: 'THE TAKEAWAY', detail: 'Review movement estimates, visible tracks, and unavailable measurements. Export the report, annotated video, and underlying data from a saved analysis.', href: '/match-center', action: 'Open your library' },
];

export function SocietyWorkflow() {
  const [selected, setSelected] = useState(0);
  return <section className="society-workflow" aria-label="How video analysis works">
    <div className="society-workflow-heading"><span className="society-label">[ FROM FOOTAGE TO UNDERSTANDING ]</span><h2>The details<br />make the <em>difference.</em></h2><p>One video. More ways to see your game.</p></div>
    <div className="society-chapters">{chapters.map((chapter, index) => <article key={chapter.label} className={`society-chapter${selected === index ? ' is-selected' : ''}`}>
      <h3><button type="button" aria-expanded={selected === index} aria-controls={`chapter-${index}`} onClick={() => setSelected(index)}><span className="society-label">{chapter.label}</span><span className="society-chapter-title">{chapter.title}</span><span className="society-chapter-sign" aria-hidden="true">{selected === index ? '−' : '+'}</span></button></h3>
      <div id={`chapter-${index}`} className="society-chapter-content" hidden={selected !== index}><p>{chapter.detail}</p><Link to={chapter.href}>{chapter.action}<ArrowUpRight size={20} aria-hidden="true" /></Link></div>
    </article>)}</div>
  </section>;
}

export function SocietyAction() {
  return <section className="society-action"><span className="society-label">[ EVERY MATCH HAS MORE TO TELL ]</span><h2>Let's look<br /><em>a little closer.</em></h2><Link className="society-cta" to="/upload"><span>Analyze your match</span><ArrowUpRight size={24} aria-hidden="true" /></Link><p>Recorded tennis. Real observations. Your next perspective.</p></section>;
}

export function useSocietyMotion() {
  const scope = useRef<HTMLDivElement>(null);
  const reducedMotion = useReducedMotion();
  useGSAP(() => {
    if (reducedMotion || !scope.current) return;
    const media = gsap.matchMedia();
    gsap.from('.society-reveal-line > span', { yPercent: 110, rotation: 2, duration: 1.05, stagger: .12, ease: 'power4.out', clearProps: 'transform' });
    gsap.from('.society-hero-art', { opacity: 0, y: 32, duration: 1.1, delay: .2, ease: 'power3.out', clearProps: 'transform,opacity' });
    gsap.utils.toArray<HTMLElement>('.society-section-heading').forEach(element => gsap.from(element, { y: 28, opacity: 0, duration: .7, ease: 'power3.out', scrollTrigger: { trigger: element, start: 'top 94%', once: true }, clearProps: 'transform,opacity' }));
    const words = scope.current.querySelectorAll('.society-manifesto-word');
    if (words.length) gsap.fromTo(words, { opacity: .2 }, { opacity: 1, stagger: .12, ease: 'none', scrollTrigger: { trigger: '.society-manifesto', start: 'top 85%', end: 'bottom 65%', scrub: .5 } });
    media.add('(min-width: 1024px)', () => {
      const title = scope.current?.querySelector('.society-workflow-heading');
      if (title) ScrollTrigger.create({ trigger: '.society-workflow', start: 'top 145px', end: 'bottom 650px', pin: title, pinSpacing: false, invalidateOnRefresh: true });
    });
    const refresh = requestAnimationFrame(() => ScrollTrigger.refresh());
    let resizeTimer: ReturnType<typeof setTimeout>;
    let previousHeight = 0;
    const observer = new ResizeObserver(entries => {
      const height = entries[0]?.contentRect.height ?? 0;
      if (Math.abs(height - previousHeight) < 1) return;
      previousHeight = height;
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => ScrollTrigger.refresh(), 150);
    });
    observer.observe(scope.current);
    return () => { cancelAnimationFrame(refresh); clearTimeout(resizeTimer); observer.disconnect(); media.revert(); };
  }, { scope, dependencies: [reducedMotion], revertOnUpdate: true });
  return scope;
}
