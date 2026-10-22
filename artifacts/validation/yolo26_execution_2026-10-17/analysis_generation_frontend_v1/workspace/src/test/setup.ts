import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverMock as typeof ResizeObserver;

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  }),
});

Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
  value: vi.fn(() => ({
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    strokeRect: vi.fn(),
    fillRect: vi.fn(),
    fillText: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    set lineWidth(_value: number) {},
    set font(_value: string) {},
    set fillStyle(_value: string) {},
    set strokeStyle(_value: string) {},
  })),
});

Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
  configurable: true,
  value: vi.fn(() => 1),
});

Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", {
  configurable: true,
  value: vi.fn(),
});
