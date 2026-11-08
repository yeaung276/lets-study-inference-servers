// Ranks figure on pages/concepts.html.
// Step 1: each node's GPUs appear (hardware). Step 2: one process starts per GPU and attaches to it.
// Hovering (or focusing) a process shows its global rank and local rank in a tooltip.

import type { Scene } from "..";

const GPUS_PER_NODE = 4;

export default function ranks(scene: Scene) {
  const { stage, tl } = scene;

  // ---- Animation: SVG-only scene, so tweens target the elements directly.
  tl.addLabel("GPUs")
    .from(".gpu", { opacity: 0, y: 8, stagger: 0.06 }, "GPUs")
    .addLabel("One process per GPU", "+=0.4")
    .from(".proc", { opacity: 0, y: -10, stagger: 0.08 }, "One process per GPU")
    .from(".drive", { opacity: 0, stagger: 0.08 }, "One process per GPU+=0.2");

  return rankHover(stage);
}

/** Hover (or focus) a `.proc` in the stage to show its global rank and local rank. Returns the cleanup. */
export function rankHover(stage: Scene["stage"]): () => void {
  // ---- Hover tooltip: an HTML box positioned over the stage.
  const tip = document.createElement("div");
  tip.className = "viz-tip";
  tip.hidden = true;
  stage.append(tip);

  const procs = [...stage.querySelectorAll<SVGGElement>(".proc")];

  const show = (proc: SVGGElement) => {
    const g = Number(proc.dataset.global);
    const l = Number(proc.dataset.local);
    const n = Number(proc.dataset.node);
    tip.innerHTML =
      `<div><span>global rank</span> <b>${g}</b></div>` +
      `<div><span>local rank</span> <b>${l}</b></div>` +
      `<div class="viz-tip-sub">${g} = node ${n} × ${GPUS_PER_NODE} + ${l}</div>`;
    const box = proc.getBoundingClientRect();
    const host = stage.getBoundingClientRect();
    tip.style.left = `${box.left - host.left + box.width / 2}px`;
    tip.style.top = `${box.top - host.top}px`;
    tip.hidden = false;
    for (const p of procs) p.classList.toggle("is-active", p === proc);
  };

  const hide = () => {
    tip.hidden = true;
    for (const p of procs) p.classList.remove("is-active");
  };

  const off: (() => void)[] = [];
  for (const proc of procs) {
    const enter = () => show(proc);
    proc.addEventListener("pointerenter", enter);
    proc.addEventListener("focus", enter);
    proc.addEventListener("pointerleave", hide);
    proc.addEventListener("blur", hide);
    off.push(() => {
      proc.removeEventListener("pointerenter", enter);
      proc.removeEventListener("focus", enter);
      proc.removeEventListener("pointerleave", hide);
      proc.removeEventListener("blur", hide);
    });
  }

  return () => {
    for (const fn of off) fn();
    tip.remove();
  };
}
