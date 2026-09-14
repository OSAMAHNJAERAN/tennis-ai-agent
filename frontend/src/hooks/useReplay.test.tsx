import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useReplay } from "./useReplay";

afterEach(() => vi.restoreAllMocks());
describe("shared replay clock", () => {
  it("advances with elapsed time and playback rate, then pauses at the last frame", () => {
    let tick: FrameRequestCallback = () => undefined;
    vi.spyOn(window, "requestAnimationFrame").mockImplementation(callback => { tick = callback; return 1; });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => undefined);
    const { result } = renderHook(() => useReplay(7, 30));
    act(() => { result.current.setRate(2); result.current.toggle(); });
    act(() => tick(0)); act(() => tick(50));
    expect(result.current.frame).toBeCloseTo(3);
    act(() => tick(100));
    expect(result.current.frame).toBe(6); expect(result.current.playing).toBe(false);
    act(() => result.current.toggle()); expect(result.current.frame).toBe(0);
  });
  it("clamps seeks and frame steps to the source bounds", () => {
    const { result } = renderHook(() => useReplay(214, 30));
    act(() => result.current.seek(999)); expect(result.current.frame).toBe(213);
    act(() => result.current.step(-1)); expect(result.current.frame).toBe(212);
    act(() => result.current.seek(-5)); expect(result.current.frame).toBe(0);
    act(() => result.current.seek(Number.NaN)); expect(result.current.frame).toBe(0);
  });
  it("does not play an empty clip", () => {
    const { result } = renderHook(() => useReplay(0, 30));
    act(() => result.current.toggle()); expect(result.current.playing).toBe(false);
  });
});
