// Reference scene used by pages/_template.html. Shows the house pattern:
// model (plain state object) → GSAP timeline with labelled steps → render(state) on canvas + SVG.

import { range, scaleBand } from "d3";
import { canvasLayer, type Scene } from "..";

const W = 640; // must match the SVG viewBox in the page
const H = 220;
const COLS = 16;
const ROWS = 4;
const TOTAL = COLS * ROWS;

export default function exampleKvFill(scene: Scene) {
  // Model: the only thing the timeline animates.
  const state = { used: 0 };

  // View: dense marks on canvas, text in the page's SVG on top.
  const layer = canvasLayer(scene, { viewBox: [W, H] });
  const count = scene.stage.querySelector<SVGTextElement>("[data-ref=count]")!;
  const x = scaleBand<number>().domain(range(COLS)).range([20, W - 20]).paddingInner(0.18);
  const y = scaleBand<number>().domain(range(ROWS)).range([48, H - 20]).paddingInner(0.25);

  scene.onRender(() => {
    const { ctx } = layer;
    const t = scene.theme;
    const used = Math.round(state.used);
    layer.clear();
    ctx.lineWidth = 1.25;
    for (let i = 0; i < TOTAL; i++) {
      ctx.beginPath();
      ctx.roundRect(x(i % COLS)!, y(Math.floor(i / COLS))!, x.bandwidth(), y.bandwidth(), 4);
      ctx.fillStyle = i < used ? t.accent : t.bgSoft;
      ctx.strokeStyle = i < used ? t.accent : t.border;
      ctx.fill();
      ctx.stroke();
    }
    count.textContent = `${used} / ${TOTAL} blocks used`;
  });

  // Each label is one step in the player. Anchor each step's tweens at its label:
  // addLabel("x", "+=0.5") alone doesn't move the next tween, so the gap would vanish.
  scene.tl
    .addLabel("prefill")
    .to(state, { used: 40, duration: 1.2, ease: "power2.out" }, "prefill")
    .addLabel("decode", "+=0.5")
    .to(state, { used: 48, duration: 2.4, ease: "none" }, "decode")
    .addLabel("finish", "+=0.5")
    .to(state, { used: 8, duration: 0.8, ease: "power2.in" }, "finish");
}
