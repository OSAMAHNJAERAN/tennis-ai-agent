export type Point2D = readonly [number, number];

export type BoundingBox = readonly [number, number, number, number];

export type Size2D = {
  width: number;
  height: number;
};

export type MediaFit = "contain" | "cover";

export type MediaRect = {
  x: number;
  y: number;
  width: number;
  height: number;
  scale: number;
};

export type DisplayBoundingBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type CoordinateTransform = {
  source: Size2D;
  container: Size2D;
  mediaRect: MediaRect;
  point: (value: Point2D) => Point2D;
  normalizedPoint: (value: Point2D) => Point2D;
  box: (value: BoundingBox) => DisplayBoundingBox;
};

const EMPTY_RECT: MediaRect = { x: 0, y: 0, width: 0, height: 0, scale: 0 };

function hasArea(size: Size2D) {
  return Number.isFinite(size.width) && Number.isFinite(size.height) && size.width > 0 && size.height > 0;
}

export function calculateMediaRect(source: Size2D, container: Size2D, fit: MediaFit = "contain"): MediaRect {
  if (!hasArea(source) || !hasArea(container)) return EMPTY_RECT;

  const scaleX = container.width / source.width;
  const scaleY = container.height / source.height;
  const scale = fit === "cover" ? Math.max(scaleX, scaleY) : Math.min(scaleX, scaleY);
  const width = source.width * scale;
  const height = source.height * scale;

  return {
    x: (container.width - width) / 2,
    y: (container.height - height) / 2,
    width,
    height,
    scale,
  };
}

export function createCoordinateTransform(
  source: Size2D,
  container: Size2D,
  fit: MediaFit = "contain",
): CoordinateTransform {
  const mediaRect = calculateMediaRect(source, container, fit);
  const point = ([x, y]: Point2D): Point2D => [
    mediaRect.x + x * mediaRect.scale,
    mediaRect.y + y * mediaRect.scale,
  ];

  return {
    source,
    container,
    mediaRect,
    point,
    normalizedPoint: ([x, y]) => [mediaRect.x + x * mediaRect.width, mediaRect.y + y * mediaRect.height],
    box: ([x1, y1, x2, y2]) => ({
      x: mediaRect.x + Math.min(x1, x2) * mediaRect.scale,
      y: mediaRect.y + Math.min(y1, y2) * mediaRect.scale,
      width: Math.abs(x2 - x1) * mediaRect.scale,
      height: Math.abs(y2 - y1) * mediaRect.scale,
    }),
  };
}

export function canvasBitmapSize(cssSize: Size2D, devicePixelRatio: number): Size2D {
  const dpr = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0 ? devicePixelRatio : 1;
  return {
    width: Math.max(1, Math.round(cssSize.width * dpr)),
    height: Math.max(1, Math.round(cssSize.height * dpr)),
  };
}
