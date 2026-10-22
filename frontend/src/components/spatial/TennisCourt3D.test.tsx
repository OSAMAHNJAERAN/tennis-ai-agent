import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { StrictMode, useState } from 'react';
// @ts-expect-error Vitest provides this Node builtin; the browser app intentionally has no Node types.
import { createRequire } from 'node:module';
import { _roots } from '@react-three/fiber';
import * as THREE from 'three';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { snapshot } from '../../data/repository';
import { normalizeTelemetry } from '../../lib/tennisTelemetry';
import TennisCourt3D from './TennisCourt3D';
import { TelemetryDashboard } from './TelemetryDashboard';

type FakeRenderer = {
  domElement: HTMLCanvasElement;
  render: ReturnType<typeof vi.fn>;
  setSize: ReturnType<typeof vi.fn>;
  setPixelRatio: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
  forceContextLoss: ReturnType<typeof vi.fn>;
  getContext: ReturnType<typeof vi.fn>;
  renderLists: { dispose: ReturnType<typeof vi.fn> };
  shadowMap: { enabled: boolean; type: number; needsUpdate: boolean };
  xr: {
    enabled: boolean;
    isPresenting: boolean;
    addEventListener: ReturnType<typeof vi.fn>;
    removeEventListener: ReturnType<typeof vi.fn>;
    setAnimationLoop: ReturnType<typeof vi.fn>;
  };
};

const rendererState = vi.hoisted(() => ({ fail: false, contextLost: false, instances: [] as FakeRenderer[] }));

// Keep Fiber's actual Canvas, measurement, reconciler, root store, and effect ordering.
// Only deterministic browser layout/observation and the GPU renderer are substituted.
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');
  return {
    ...actual,
    WebGLRenderer: vi.fn(function (options: THREE.WebGLRendererParameters) {
      if (rendererState.fail) throw new Error('Test GPU context creation refused');
      const renderer: FakeRenderer = {
        domElement: options.canvas as HTMLCanvasElement,
        render: vi.fn(), setSize: vi.fn(), setPixelRatio: vi.fn(),
        dispose: vi.fn(), forceContextLoss: vi.fn(),
        getContext: vi.fn(() => ({ isContextLost: () => rendererState.contextLost })),
        renderLists: { dispose: vi.fn() },
        shadowMap: { enabled: false, type: 0, needsUpdate: false },
        xr: { enabled: false, isPresenting: false, addEventListener: vi.fn(),
          removeEventListener: vi.fn(), setAnimationLoop: vi.fn() },
      };
      rendererState.instances.push(renderer);
      return renderer;
    }),
  };
});

// Installed Fiber is external CommonJS in Vitest and requires Three directly.
// Substitute its renderer binding too, without replacing Canvas or configure.
const installedThree = createRequire(import.meta.url)('three') as typeof import('three');

const base = snapshot.details.phase5_scoring_1!;
const run = { ...base, trajectories: null, events: [], summary: { ...base.summary, frames: 3 } };
const telemetry = normalizeTelemetry(run);
const playhead = { current: 1 };

function courtProps(onFallback: () => void = vi.fn()) {
  return { run, telemetry, playhead, frame: 1, playing: false, trails: false,
    arc: false, reducedMotion: true, onFallback };
}

async function settle() {
  await act(async () => {
    await Promise.resolve();
    // Canvas requests scroll-aware measurement; its ResizeObserver callback is
    // debounced by 50 ms in the installed react-use-measure implementation.
    await vi.advanceTimersByTimeAsync(60);
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe('WebGL court failure and renderer ownership', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    rendererState.fail = false;
    rendererState.contextLost = false;
    rendererState.instances.length = 0;
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    vi.stubGlobal('WebGL2RenderingContext', class {});
    vi.stubGlobal('ResizeObserver', class {
      constructor(private readonly callback: ResizeObserverCallback) {}
      observe() { queueMicrotask(() => this.callback([], this as unknown as ResizeObserver)); }
      unobserve() {}
      disconnect() {}
    });
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      width: 640, height: 360, left: 0, top: 0, right: 640, bottom: 360, x: 0, y: 0,
      toJSON: () => ({}),
    });
    vi.spyOn(installedThree, 'WebGLRenderer').mockImplementation(function (options) {
      return new THREE.WebGLRenderer(options);
    });
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(1);
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      fillStyle: '', font: '', textAlign: 'center', beginPath: vi.fn(), roundRect: vi.fn(),
      fill: vi.fn(), fillText: vi.fn(), clearRect: vi.fn(),
    } as unknown as CanvasRenderingContext2D);
  });

  afterEach(async () => {
    await act(async () => cleanup());
    await act(async () => { await vi.advanceTimersByTimeAsync(510); });
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('offers 2D recovery without camera controls when the WebGL API is absent', () => {
    vi.stubGlobal('WebGL2RenderingContext', undefined);
    const onFallback = vi.fn();
    render(<TennisCourt3D {...courtProps(onFallback)} />);

    expect(screen.getByRole('heading', { name: '3D rendering unavailable' })).toBeInTheDocument();
    expect(THREE.WebGLRenderer).not.toHaveBeenCalled();
    expect(screen.queryByRole('group', { name: 'Camera angle' })).not.toBeInTheDocument();
    expect(screen.queryByRole('group', { name: '3D camera controls' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open 2D tactical view' }));
    expect(onFallback).toHaveBeenCalledOnce();
  });

  it('catches actual-Canvas renderer construction before async configure can continue', async () => {
    rendererState.fail = true;
    const rootCount = _roots.size;
    const unhandled: unknown[] = [];
    const onUnhandled = (event: PromiseRejectionEvent) => { unhandled.push(event.reason); };
    window.addEventListener('unhandledrejection', onUnhandled);
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
      render(<TennisCourt3D {...courtProps()} />);
      await settle();

      expect(THREE.WebGLRenderer).toHaveBeenCalled();
      expect(screen.getByRole('heading', { name: '3D rendering unavailable' })).toBeInTheDocument();
      expect(screen.queryByRole('group', { name: 'Camera angle' })).not.toBeInTheDocument();
      expect(document.querySelector('.court3d-host canvas')).not.toBeInTheDocument();
      expect(_roots.size).toBe(rootCount);
      expect(unhandled).toEqual([]);
    } finally {
      consoleError.mockRestore();
      window.removeEventListener('unhandledrejection', onUnhandled);
    }
  });

  it('owns one renderer for the real Canvas across StrictMode reattachment and disposes on detachment', async () => {
    const props = courtProps();
    const view = render(<StrictMode><TennisCourt3D {...props} /></StrictMode>);
    await settle();
    const canvas = document.querySelector('.court3d-host canvas') as HTMLCanvasElement;
    const renderer = rendererState.instances[0]!;

    expect(canvas).toBeInstanceOf(HTMLCanvasElement);
    expect(THREE.WebGLRenderer).toHaveBeenCalledOnce();
    expect(renderer.domElement).toBe(canvas);
    expect(_roots.get(canvas)?.store.getState().gl).toBe(renderer);
    const requestedContexts = vi.mocked(HTMLCanvasElement.prototype.getContext).mock.calls.map(call => call[0]);
    expect(requestedContexts).not.toContain('webgl2');
    expect(requestedContexts).not.toContain('webgl');
    expect(renderer.dispose).not.toHaveBeenCalled();

    view.rerender(<StrictMode><TennisCourt3D {...props} frame={2} /></StrictMode>);
    await settle();
    expect(THREE.WebGLRenderer).toHaveBeenCalledOnce();
    expect(renderer.dispose).not.toHaveBeenCalled();

    view.unmount();
    await settle();
    expect(canvas.isConnected).toBe(false);
    expect(renderer.dispose).toHaveBeenCalledOnce();
    await act(async () => { await vi.advanceTimersByTimeAsync(510); });
    expect(renderer.renderLists.dispose).toHaveBeenCalled();
    expect(renderer.forceContextLoss).toHaveBeenCalled();
    expect(_roots.has(canvas)).toBe(false);
  });

  it('removes the lost-context listener and lets the caller recover without changing the evidence frame', async () => {
    const added = vi.spyOn(HTMLCanvasElement.prototype, 'addEventListener');
    const removed = vi.spyOn(HTMLCanvasElement.prototype, 'removeEventListener');
    function RecoverableCourt() {
      const [view, setView] = useState<'3d' | '2d'>('3d');
      return view === '3d'
        ? <TennisCourt3D {...courtProps(() => setView('2d'))} />
        : <output aria-label="2D recovery frame">{playhead.current}</output>;
    }
    render(<RecoverableCourt />);
    await settle();
    const canvas = document.querySelector('.court3d-host canvas') as HTMLCanvasElement;
    const registrations = added.mock.calls.filter(call => call[0] === 'webglcontextlost');
    expect(registrations).toHaveLength(1);
    const listener = registrations[0]![1];
    const lost = new Event('webglcontextlost', { cancelable: true });
    act(() => { canvas.dispatchEvent(lost); });
    await settle();

    expect(lost.defaultPrevented).toBe(true);
    expect(screen.getByRole('heading', { name: '3D rendering unavailable' })).toBeInTheDocument();
    expect(screen.queryByRole('group', { name: '3D camera controls' })).not.toBeInTheDocument();
    expect(removed).toHaveBeenCalledWith('webglcontextlost', listener);
    fireEvent.click(screen.getByRole('button', { name: 'Open 2D tactical view' }));
    expect(screen.getByLabelText('2D recovery frame')).toHaveTextContent('1');
    expect(playhead.current).toBe(1);
  });

  it('preserves invalid-court abstention instead of attempting a WebGL scene', async () => {
    const invalid = { ...run, court: run.court ? { ...run.court, isValid: false } : null,
      trajectories: { ...base.trajectories!, player1CourtPositions: [[0, 0], null, null] as Array<[number, number] | null> } };
    render(<TelemetryDashboard run={invalid} selectedEventId={null} onSelectEvent={vi.fn()} />);
    fireEvent.click(screen.getByRole('tab', { name: '3D Court' }));
    await settle();

    expect(screen.getByRole('heading', { name: 'Court replay unavailable' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: '3D rendering unavailable' })).not.toBeInTheDocument();
    expect(THREE.WebGLRenderer).not.toHaveBeenCalled();
  });

  it('recovers when the context was already lost before the monitor subscribed', async () => {
    rendererState.contextLost = true;
    const onFallback = vi.fn();
    render(<TennisCourt3D {...courtProps(onFallback)} />);
    await settle();

    expect(THREE.WebGLRenderer).toHaveBeenCalledOnce();
    expect(screen.getByRole('heading', { name: '3D rendering unavailable' })).toBeInTheDocument();
    expect(screen.queryByRole('group', { name: '3D camera controls' })).not.toBeInTheDocument();
    expect(document.querySelector('.court3d-host canvas')).not.toBeInTheDocument();
    expect(rendererState.instances[0]!.getContext).toHaveBeenCalled();
    expect(rendererState.instances[0]!.dispose).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole('button', { name: 'Open 2D tactical view' }));
    expect(onFallback).toHaveBeenCalledOnce();
    expect(playhead.current).toBe(1);
  });
});
