import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useReplay } from './useReplay';

afterEach(() => vi.restoreAllMocks());

describe('replay clock ownership', () => {
  it('holds telemetry without decoded media observations, including playback stalls', () => {
    const raf = vi.spyOn(window, 'requestAnimationFrame');
    const { result } = renderHook(() => useReplay(300, 30, 0, 'media'));
    act(() => result.current.toggle());
    expect(result.current.playing).toBe(true);
    expect(raf).not.toHaveBeenCalled();
    expect(result.current.frame).toBe(0);
    act(() => result.current.observeTime(1.5));
    expect(result.current.frame).toBe(45);
    expect(result.current.seekRevision).toBe(0);
    act(() => result.current.setRate(2));
    expect(result.current.frame).toBe(45);
    expect(raf).not.toHaveBeenCalled();
  });

  it('separates seek commands from presented timestamps and steps exactly one frame', () => {
    const { result } = renderHook(() => useReplay(300, 30, 0, 'media'));
    act(() => result.current.seek(90));
    expect(result.current.seekRevision).toBe(1);
    act(() => result.current.observeTime(3));
    expect(result.current.seekRevision).toBe(1);
    // Decoder timestamps may sit infinitesimally below the nominal frame boundary.
    act(() => result.current.observeTime(3 - Number.EPSILON * 3));
    expect(result.current.frame).toBe(90);
    act(() => result.current.step(1));
    expect(result.current.frame).toBe(90);
    expect(result.current.commandTime).toBeCloseTo(91 / 30);
    act(() => result.current.observeTime(91 / 30));
    expect(result.current.frame).toBe(91);
    expect(result.current.seekRevision).toBe(2);
    expect(result.current.playing).toBe(false);
  });

  it('holds media evidence during a pending seek and accumulates steps from the requested target', () => {
    const fps = 24000 / 1001;
    const { result } = renderHook(() => useReplay(469, fps, 0, 'media'));
    act(() => result.current.observeTime(30 / fps));
    act(() => result.current.seek(90));
    expect(result.current.frame).toBe(30);
    expect(result.current.time).toBeCloseTo(30 / fps);
    expect(result.current.commandTime).toBeCloseTo(90 / fps);
    act(() => { result.current.step(1); result.current.step(1); });
    expect(result.current.frame).toBe(30);
    expect(result.current.commandTime).toBeCloseTo(92 / fps);
    expect(result.current.seekRevision).toBe(3);
    act(() => result.current.observeTime(92 / fps));
    expect(result.current.frame).toBe(92);
    expect(result.current.commandTime).toBeCloseTo(92 / fps);
    act(() => result.current.step(-1));
    expect(result.current.frame).toBe(92);
    expect(result.current.commandTime).toBeCloseTo(91 / fps);
    act(() => result.current.observeTime(91 / fps));
    expect(result.current.frame).toBe(91);
  });

  it('retains the video observation when returning from a different simulation position', () => {
    const { result, rerender } = renderHook(({ clock }: { clock: 'media' | 'simulation' }) =>
      useReplay(469, 30, 0, clock), { initialProps: { clock: 'media' } });
    act(() => result.current.observeTime(1));
    rerender({ clock: 'simulation' });
    expect(result.current.frame).toBe(30);
    act(() => result.current.seek(180));
    expect(result.current.frame).toBe(180);
    rerender({ clock: 'media' });
    expect(result.current.frame).toBe(30);
    expect(result.current.commandTime).toBe(6);
    act(() => result.current.observeTime(6));
    expect(result.current.frame).toBe(180);
    expect(result.current.time).toBe(6);
  });
});
