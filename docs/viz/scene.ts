// A scene is one animated figure: a stage (SVG and/or canvas layers), one paused master GSAP timeline,
// and a list of render functions that redraw the view from the current state.
//
//   mount("#kv-paging", (scene) => {
//     const state = { used: 0 };                         // model: the only thing the timeline animates
//     scene.onRender(() => draw(state, scene.theme));    // view: redrawn from state on every timeline update
//     scene.tl.addLabel("prefill").to(state, { used: 20 })
//             .addLabel("decode").to(state, { used: 26 });   // labels become the player's steps
//   });

import { gsap } from "gsap";
import { attachPlayer } from "./player";
import { onThemeChange, readTheme, type Theme } from "./theme";

export interface SceneOptions {
  /** Start playing the first time the figure scrolls into view. Default true. Off under prefers-reduced-motion. */
  autoplay?: boolean;
  /** Repeat forever with a 1 s pause between runs. Default false. */
  loop?: boolean;
  /** Show the player bar: play/pause, previous/next step, scrubber. Default true. */
  controls?: boolean;
  /** Width / height of the stage. Only needed when no in-flow SVG sizes it (canvas- or WebGL-only scenes). */
  aspect?: number;
  /** Defaults merged into every tween on the master timeline. */
  defaults?: gsap.TweenVars;
}

/** Builds the scene. Runs once inside a gsap.context scoped to the stage, so selector strings only match inside it. */
export type SceneBuilder = (scene: Scene) => void | (() => void);

export interface Size {
  width: number;
  height: number;
}

export interface Step {
  name: string;
  time: number;
}

const EPS = 1e-3;

export class Scene {
  readonly figure: HTMLElement;
  readonly stage: HTMLElement;
  /** The master timeline. Paused; the scene and its player drive it. Don't set its onUpdate/onComplete. */
  readonly tl: gsap.core.Timeline;
  readonly reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  /** Current theme tokens, refreshed before renderers run after a theme change. */
  theme: Theme = readTheme();
  /** Stage size in CSS pixels. */
  size: Size;

  #renderers: (() => void)[] = [];
  #resizers: ((size: Size) => void)[] = [];
  #cleanup: (() => void)[] = [];
  #driver: gsap.core.Tween | undefined;
  #autoplayPending: boolean;

  constructor(figure: HTMLElement, build: SceneBuilder, opts: SceneOptions = {}) {
    this.figure = figure;
    this.stage = figure.querySelector<HTMLElement>(":scope > .viz-stage") ?? createStage(figure);
    figure.classList.add("viz");
    if (opts.aspect) this.stage.style.aspectRatio = String(opts.aspect);
    const rect = this.stage.getBoundingClientRect();
    this.size = { width: rect.width, height: rect.height };
    this.#autoplayPending = opts.autoplay !== false && !this.reducedMotion;

    this.tl = gsap.timeline({
      paused: true,
      repeat: opts.loop ? -1 : 0,
      repeatDelay: 1,
      defaults: { duration: 0.6, ease: "power2.inOut", ...opts.defaults },
      onUpdate: () => this.render(),
      onComplete: () => this.render(),
    });

    const ctx = gsap.context(() => {}, this.stage);
    try {
      ctx.add(() => {
        const cleanup = build(this);
        if (cleanup) this.#cleanup.push(cleanup);
      });
    } catch (err) {
      ctx.revert(); // put the static markup back
      this.tl.kill();
      throw err;
    }
    this.#cleanup.push(() => ctx.revert());

    if (this.reducedMotion) this.tl.progress(1); // show the finished figure; steps and scrubber still work
    if (opts.controls !== false) this.#cleanup.push(attachPlayer(this));
    this.#observe();
    this.render();
  }

  /** Redraw every layer from the current state. Called on timeline updates, resizes and theme changes. */
  render(): void {
    for (const fn of this.#renderers) fn();
  }

  onRender(fn: () => void): void {
    this.#renderers.push(fn);
  }

  /** Called now (if the stage has a size) and whenever the stage resizes. Renderers run right after. */
  onResize(fn: (size: Size) => void): void {
    this.#resizers.push(fn);
    if (this.size.width > 0) fn(this.size);
  }

  get playing(): boolean {
    return !!this.#driver?.isActive() || (!this.tl.paused() && this.tl.isActive());
  }

  /** Timeline labels in time order. Each label is one step in the player. */
  get steps(): Step[] {
    return Object.entries(this.tl.labels)
      .map(([name, time]) => ({ name, time }))
      .sort((a, b) => a.time - b.time);
  }

  /** Index into `steps` of the step the playhead is in, or -1 before the first label. */
  get stepIndex(): number {
    const now = this.tl.time();
    return this.steps.findLastIndex((s) => s.time <= now + EPS);
  }

  play(): void {
    this.#autoplayPending = false;
    this.#run();
  }

  pause(): void {
    this.#autoplayPending = false;
    this.#stopDriver();
    this.tl.pause();
    this.render();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  /** Jump to a point in the timeline, 0–1, and pause there. */
  seek(progress: number): void {
    this.#autoplayPending = false;
    this.#stopDriver();
    this.tl.pause().progress(progress);
    this.render();
  }

  /** Move to the next or previous label. Forward plays the step at normal speed; back rewinds quickly. */
  step(dir: 1 | -1): void {
    this.#autoplayPending = false;
    const now = this.tl.time();
    const target =
      dir > 0
        ? (this.steps.find((s) => s.time > now + EPS)?.time ?? this.tl.duration())
        : (this.steps.findLast((s) => s.time < now - EPS)?.time ?? 0);
    this.#stopDriver();
    this.tl.pause();
    if (this.reducedMotion) {
      this.tl.time(target);
      this.render();
      return;
    }
    this.#driver = this.tl.tweenTo(target, {
      duration: dir > 0 ? target - now : Math.min(now - target, 0.5),
      ease: dir > 0 ? "none" : "power2.out",
      onComplete: () => this.render(),
    });
    this.render();
  }

  destroy(): void {
    this.#stopDriver();
    this.tl.kill();
    for (const fn of this.#cleanup.splice(0).reverse()) fn();
    this.#renderers = [];
    this.#resizers = [];
  }

  #run(): void {
    this.#stopDriver();
    if (this.tl.repeat() !== -1 && this.tl.progress() >= 1) this.tl.restart();
    else this.tl.play();
    this.render();
  }

  #stopDriver(): void {
    this.#driver?.kill();
    this.#driver = undefined;
  }

  #observe(): void {
    const resize = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (width === this.size.width && height === this.size.height) return;
      this.size = { width, height };
      for (const fn of this.#resizers) fn(this.size);
      this.render();
    });
    resize.observe(this.stage);

    // Autoplay on first view; pause while scrolled away and pick up again on return.
    let resume = false;
    const visibility = new IntersectionObserver(
      ([entry]) => {
        if (!entry) return;
        if (entry.isIntersecting) {
          if (this.#autoplayPending || resume) this.#run();
          this.#autoplayPending = resume = false;
        } else if (this.playing && !this.#driver) {
          resume = true;
          this.tl.pause();
          this.render();
        }
      },
      { threshold: 0.25 },
    );
    visibility.observe(this.figure);

    const unTheme = onThemeChange((theme) => {
      this.theme = theme;
      this.render();
    });

    this.#cleanup.push(() => resize.disconnect(), () => visibility.disconnect(), unTheme);
  }
}

/** Mount a scene on a figure. Returns undefined (and leaves the static figure alone) if the target is missing or the build throws. */
export function mount(target: string | HTMLElement, build: SceneBuilder, opts?: SceneOptions): Scene | undefined {
  const figure = typeof target === "string" ? document.querySelector<HTMLElement>(target) : target;
  if (!figure) {
    console.warn(`[viz] nothing matches ${String(target)}`);
    return undefined;
  }
  try {
    return new Scene(figure, build, opts);
  } catch (err) {
    console.error(`[viz] scene on ${String(target)} failed to build`, err);
    return undefined;
  }
}

function createStage(figure: HTMLElement): HTMLElement {
  const stage = document.createElement("div");
  stage.className = "viz-stage";
  for (const child of [...figure.children]) if (child.tagName !== "FIGCAPTION") stage.append(child);
  figure.prepend(stage);
  return stage;
}
