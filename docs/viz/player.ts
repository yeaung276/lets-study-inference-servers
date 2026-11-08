// The player bar under an animated figure: play/pause, previous/next step (timeline labels), scrubber.

import type { Scene } from "./scene";

const ICONS = {
  play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>',
  pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5h3v11H4zm5 0h3v11H9z"/></svg>',
  prev: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 2.5h2v11H3zm10 0v11L6 8z"/></svg>',
  next: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M11 2.5h2v11h-2zM3 2.5v11L10 8z"/></svg>',
};

export function attachPlayer(scene: Scene): () => void {
  const steps = scene.steps;
  const bar = document.createElement("div");
  bar.className = "viz-controls";
  bar.innerHTML = `
    <button type="button" class="viz-btn" data-act="toggle"></button>
    <button type="button" class="viz-btn" data-act="prev" aria-label="Previous step">${ICONS.prev}</button>
    <button type="button" class="viz-btn" data-act="next" aria-label="Next step">${ICONS.next}</button>
    <input class="viz-scrub" type="range" min="0" max="1000" step="1" value="0" aria-label="Animation position">
    <span class="viz-step" aria-live="polite"></span>`;

  const button = (act: string) => bar.querySelector<HTMLButtonElement>(`[data-act="${act}"]`)!;
  const toggle = button("toggle");
  const prev = button("prev");
  const next = button("next");
  const scrub = bar.querySelector<HTMLInputElement>(".viz-scrub")!;
  const label = bar.querySelector<HTMLElement>(".viz-step")!;
  if (steps.length === 0) prev.hidden = next.hidden = label.hidden = true;

  toggle.addEventListener("click", () => scene.toggle());
  prev.addEventListener("click", () => scene.step(-1));
  next.addEventListener("click", () => scene.step(1));
  scrub.addEventListener("input", () => scene.seek(scrub.valueAsNumber / 1000));
  scene.stage.after(bar);

  // Runs every frame while playing, so only touch the DOM when something changed.
  let playing: boolean | undefined;
  let position = -1;
  let stepText: string | undefined;
  scene.onRender(() => {
    if (scene.playing !== playing) {
      playing = scene.playing;
      toggle.innerHTML = playing ? ICONS.pause : ICONS.play;
      toggle.setAttribute("aria-label", playing ? "Pause" : "Play");
    }
    const p = Math.round(scene.tl.progress() * 1000);
    if (p !== position) scrub.value = String((position = p));
    const i = scene.stepIndex;
    const text = steps[i] ? `${i + 1}/${steps.length} · ${steps[i].name}` : "";
    if (text !== stepText) label.textContent = stepText = text;
  });

  return () => bar.remove();
}
