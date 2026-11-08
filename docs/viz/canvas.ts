// A HiDPI canvas layer behind the stage's SVG. Use it for dense marks (heatmaps, block grids, particles)
// and keep labels, arrows and anything interactive in the SVG on top.

import type { Scene } from "./scene";

export interface CanvasLayer {
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  /** Drawing width in user units: the viewBox width if one was given, else CSS pixels. */
  width: number;
  /** Drawing height in user units. */
  height: number;
  clear(): void;
}

export interface CanvasLayerOptions {
  /**
   * Draw in the same coordinates as the stage's SVG by passing its viewBox size, e.g. [760, 400].
   * The layer then scales those units to the stage and the device pixel ratio for you.
   */
  viewBox?: [width: number, height: number];
}

export function canvasLayer(scene: Scene, opts: CanvasLayerOptions = {}): CanvasLayer {
  const canvas = document.createElement("canvas");
  canvas.className = "viz-canvas";
  canvas.setAttribute("aria-hidden", "true");
  scene.stage.prepend(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas is not available");

  const layer: CanvasLayer = {
    canvas,
    ctx,
    width: 0,
    height: 0,
    clear() {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.restore();
    },
  };

  scene.onResize(({ width, height }) => {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    const [w, h] = opts.viewBox ?? [width, height];
    ctx.setTransform((dpr * width) / w, 0, 0, (dpr * height) / h, 0, 0);
    layer.width = w;
    layer.height = h;
  });

  return layer;
}
