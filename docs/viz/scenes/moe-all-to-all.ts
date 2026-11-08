// MoE all-to-all figure on pages/concepts.html.
// Follows rank 0's tokens through a DeepEP-style dispatch and combine: 4 ranks, 8 experts (2 per rank), top-2
// routing, one direct link between every pair of ranks.
// Steps: overview → tokens → zoom R0 → gate → notify → dispatch → every rank → gpu_barrier → zoom R1 →
// copy epilogue → grouped GEMM → combine → gpu_barrier → output.
// Model/view: the timeline tweens `s`; render() places every element from it. All counts, slots and buffer
// contents are derived from ROUTES, so the numbers on screen always agree with each other.

import type { Scene } from "..";

const NS = "http://www.w3.org/2000/svg";
const W = 960;
const H = 710;
const N = 4; // ranks
const EPR = 2; // experts per rank
const BIG = { w: 440, h: 550 };

type P = [number, number];
interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

// ---------------------------------------------------------------- data

/** Each token's two chosen experts. Tokens a1–a4 live on R0, b1–b4 on R1, c1–c4 on R2, d1–d4 on R3. */
const ROUTES: Record<string, readonly [number, number]> = {
  a1: [2, 3], a2: [0, 5], a3: [3, 6], a4: [1, 4],
  b1: [0, 2], b2: [2, 3], b3: [1, 7], b4: [3, 5],
  c1: [2, 6], c2: [0, 1], c3: [5, 3], c4: [7, 4],
  d1: [3, 1], d2: [6, 7], d3: [2, 0], d4: [5, 6],
};
const FOCUS = "a1";
/** Each rank's tokens get their own letter: they are that rank's own requests, unrelated to other ranks' tokens. */
const LETTERS = "abcd";

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const tokensOf = (r: number) => range(4).map((i) => `${LETTERS[r]}${i + 1}`);
const homeOf = (t: string) => LETTERS.indexOf(t[0]!);
const rankOf = (e: number) => Math.floor(e / EPR);
const route = (t: string) => ROUTES[t]!;
/** Destination ranks of a token, deduplicated: a token goes to a rank once even if two of its experts live there. */
const destRanks = (t: string) => [...new Set(route(t).map(rankOf))];
/** Combine row on the home rank for the partial that sender rank `sr` returns for token `t`. With more ranks than
 *  top-k choices (4 > 2), DeepEP indexes the combine buffer by top-k slot, not by rank: the row is the slot of the
 *  token's first choice that lives on `sr`. Two senders never share a row, as each holds different choices. */
const combineSlot = (t: string, sr: number) => route(t).findIndex((e) => rankOf(e) === sr);

interface RankData {
  rankCount: number[]; // tokens this rank sends to each rank
  expertCount: number[]; // per global expert
  dstSlot: Map<string, P[]>; // token → [(dst rank, slot)]
  recv: string[][]; // recv[src] = tokens src wrote into this rank's recv_buffer, in slot order
  recvX: string[][]; // per local expert, in arrival order (src, then slot)
}

function derive(): RankData[] {
  const data: RankData[] = range(N).map(() => ({
    rankCount: Array<number>(N).fill(0),
    expertCount: Array<number>(N * EPR).fill(0),
    dstSlot: new Map(),
    recv: range(N).map(() => []),
    recvX: range(EPR).map(() => []),
  }));
  for (let src = 0; src < N; src++) {
    const me = data[src]!;
    for (const t of tokensOf(src)) {
      for (const e of route(t)) me.expertCount[e]!++;
      const slots: P[] = [];
      for (const dst of destRanks(t)) {
        slots.push([dst, me.rankCount[dst]!++]);
        data[dst]!.recv[src]!.push(t);
      }
      me.dstSlot.set(t, slots);
    }
  }
  for (let r = 0; r < N; r++) {
    const me = data[r]!;
    for (const row of me.recv)
      for (const t of row) for (const e of route(t)) if (rankOf(e) === r) me.recvX[e - r * EPR]!.push(t);
  }
  return data;
}

const DATA = derive();
const recvRankCount = (r: number) => DATA[r]!.recv.map((row) => row.length);
const recvExpertCount = (r: number) => DATA[r]!.recvX.map((g) => g.length);
const inclusive = (xs: number[]) => xs.map((_, i) => xs.slice(0, i + 1).reduce((a, b) => a + b, 0));

// ---------------------------------------------------------------- layout

const L0: Box[] = range(N).map((i) => ({ x: 20 + i * 236, y: 260, w: 212, h: 100 }));
const L1: Box[] = [
  { x: 20, y: 40, ...BIG },
  { x: 600, y: 40, w: 340, h: 150 },
  { x: 600, y: 210, w: 340, h: 150 },
  { x: 600, y: 380, w: 340, h: 150 },
];
const L2: Box[] = [
  { x: 20, y: 40, ...BIG },
  { x: 500, y: 40, ...BIG },
  { x: 500, y: 608, w: 210, h: 90 },
  { x: 730, y: 608, w: 210, h: 90 },
];

// Positions inside a big rank box (local coordinates).
const GRID = { x: 56, y: 254, dx: 50, dy: 16, w: 46, h: 13 };
const RECVX = { x: 52, y: 364, dx: 36, dy: 28, w: 32, h: 20 };
const TOK = { x: 34, y: 58, dx: 42, w: 36, h: 18 };
const OUT = { x: 206, y: 518, dx: 54, w: 48, h: 18 };
const B1_Y = 336;
const B2_Y = 490;

const STEPS: [string, string, string][] = [
  ["4 ranks, 8 experts", "4 ranks, each holding 2 of the 8 experts", "Experts E0–E7 are split evenly across ranks R0–R3. Every pair of ranks has a direct link."],
  ["Tokens arrive on R0", "Tokens a1–a4 arrive on R0, their home rank", "They are R0's batch for this step: a chunk of one prompt, or one new token from each of 4 requests. Every rank has its own batch from its own requests, so R1's tokens are b1–b4, R2's c1–c4, R3's d1–d4."],
  ["Zoom into R0", "Zoom into R0", "Inside: the gate, the variables this rank keeps, recv_buffer, recv_x, the GPU, and the output."],
  ["Gate", "Gate (router): each token picks its top-2 experts", "The chosen experts light up, on R0 and on the other ranks. topk_idx records the choices."],
  ["notify", "notify: R0 tells every rank how many tokens are coming", "dispatch_impl's notify warps count per rank and per expert, then gin.put_value writes the counts into each peer."],
  ["dispatch", "dispatch: R0 writes each token into its destinations' recv_buffer", "Same kernel as notify, running at the same time. One copy per rank: a1 goes to R1 once, for both E2 and E3."],
  ["Every rank does the same", "Every rank does the same with its own tokens", "R0's recv_buffer fills from R1–R3. Once every count is in, R0 computes psum_rank and psum_expert."],
  ["gpu_barrier (dispatch)", "gpu_barrier: wait until all 4 ranks have finished sending", "No rank reads its recv_buffer until every rank has arrived here."],
  ["Zoom into R1", "Zoom into R1, which holds E2 and E3", "R1 went through the same steps in parallel. Its recv_buffer holds a1 and a3 from R0."],
  ["dispatch_copy_epilogue", "dispatch_copy_epilogue: regroup rows by expert into recv_x", "A token is copied once per local expert it chose: on R1, a1 lands under both E2 and E3."],
  ["Grouped GEMM", "Grouped GEMM on the GPU (DeepGEMM)", "Each expert runs on its group of rows: down( SwiGLU(gate_up(x)) · w ), with w the token's gate weight."],
  ["combine", "combine: each rank sums a token's rows and sends one partial home", "a1's E2 and E3 results become one partial (a1Σ), written into R0's recv_buffer[k0][a1]: row k0 because E2, a1's first choice, is on R1. The shared expert runs meanwhile."],
  ["gpu_barrier (combine)", "gpu_barrier: wait until every partial has arrived", "recv_buffer is reused for combine. Here there are more ranks (4) than choices per token (2), so rows are top-k slots: a partial goes to the row of the first choice its sender holds. Columns are the home rank's tokens."],
  ["combine_reduce_epilogue", "combine_reduce_epilogue: out[t] = Σ partials + shared_out[t]", "Each column of recv_buffer is summed with the shared expert's output into the final token."],
];

// ---------------------------------------------------------------- helpers

const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const lerpP = (a: P, b: P, t: number): P => [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];
const lerpBox = (a: Box, b: Box, t: number): Box => ({
  x: lerp(a.x, b.x, t),
  y: lerp(a.y, b.y, t),
  w: lerp(a.w, b.w, t),
  h: lerp(a.h, b.h, t),
});
const ease = (p: number) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);

type Attrs = Record<string, string | number>;

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Attrs, parent: Element): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  parent.append(node);
  return node;
}

function text(parent: Element, attrs: Attrs, content = ""): SVGTextElement {
  const t = el("text", attrs, parent);
  t.textContent = content;
  return t;
}

/** Set attributes only when they change: render runs every frame. */
function set(node: Element, attrs: Attrs): void {
  for (const [k, v] of Object.entries(attrs)) {
    const s = typeof v === "number" ? String(Math.round(v * 100) / 100) : v;
    if (node.getAttribute(k) !== s) node.setAttribute(k, s);
  }
}

function setFont(node: SVGElement, px: number): void {
  const v = `${Math.round(px * 10) / 10}px`;
  if (node.style.fontSize !== v) node.style.fontSize = v;
}

function setText(node: Element, s: string): void {
  if (node.textContent !== s) node.textContent = s;
}

/** A labelled rounded rect: tokens, expert chips, buffer cells. Positioned by its centre. */
class Chip {
  readonly g: SVGGElement;
  readonly rect: SVGRectElement;
  readonly label: SVGTextElement;
  constructor(parent: Element, content: string) {
    this.g = el("g", {}, parent);
    this.rect = el("rect", { rx: 4 }, this.g);
    this.label = text(this.g, { "text-anchor": "middle", class: "t-chip" }, content);
  }
  place([cx, cy]: P, w: number, h: number, opacity: number, hot: boolean, font = 10): void {
    set(this.g, { opacity: clamp(opacity), display: opacity <= 0.001 ? "none" : "inline" });
    if (opacity <= 0.001) return;
    set(this.rect, { x: cx - w / 2, y: cy - h / 2, width: w, height: h, class: hot ? "b-focus" : "b" });
    set(this.label, { x: cx, y: cy + font * 0.35, class: hot ? "t-chip t-hot" : "t-chip" });
    setFont(this.label, font);
  }
}

// ---------------------------------------------------------------- scene

export default function moeAllToAll(scene: Scene) {
  const { stage, tl } = scene;

  // Model: the only thing the timeline animates.
  const s = {
    intro: 0, tokens: 0, focus0: 0, gate: 0, notify: 0, fill: 0, others: 0, barrier1: 0,
    focus1: 0, copy: 0, gemm: 0, combine: 0, barrier2: 0, output: 0,
  };
  type Phase = keyof typeof s;
  const between = (a: Phase, b: Phase) => s[a] > 0 && s[b] === 0;

  // The static figure in the page stays as the no-JavaScript fallback; the scene draws its own SVG.
  const fallback = stage.querySelector<SVGSVGElement>(":scope > svg");
  const svg = el("svg", {
    class: "diagram",
    viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label":
      "Animated MoE all-to-all with 4 ranks and 8 experts. Rank 0's tokens go through the gate, notify, dispatch, " +
      "a GPU barrier, the copy epilogue and grouped GEMM on rank 1, combine back to rank 0, a second barrier and the output.",
  }, stage);
  const defs = el("defs", {}, svg);
  for (const [id, cls] of [["a2av-ah", "ah"], ["a2av-ah-hot", "ah-hot"]] as const) {
    const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    el("path", { class: cls, d: "M0,0 L10,5 L0,10 z" }, m);
  }

  // ---- layers, back to front
  const meshLayer = el("g", {}, svg);
  const rankLayer = el("g", {}, svg);
  const lineLayer = el("g", {}, svg);
  const chipLayer = el("g", {}, svg);

  // What the current step does, in words, under the diagram.
  const desc = document.createElement("div");
  desc.className = "viz-desc";
  desc.setAttribute("aria-live", "polite");
  const title = desc.appendChild(document.createElement("strong"));
  const detail = desc.appendChild(document.createElement("span"));
  stage.append(desc);

  // ---- geometry from state
  const box = (r: number): Box => lerpBox(lerpBox(L0[r]!, L1[r]!, s.focus0), L2[r]!, s.focus1);
  const bigness = (r: number) => (r === 0 ? s.focus0 : r === 1 ? s.focus1 : 0);
  const inner = (r: number) => clamp((bigness(r) - 0.55) / 0.45);
  const abs = (r: number, [lx, ly]: P): P => {
    const b = box(r);
    return [b.x + lx, b.y + ly];
  };
  const center = (r: number): P => {
    const b = box(r);
    return [b.x + b.w / 2, b.y + b.h / 2];
  };
  const topEdge = (r: number): P => {
    const b = box(r);
    return [b.x + b.w / 2, b.y + 6];
  };
  const gridAt = (r: number, src: number, col: number): P =>
    abs(r, [GRID.x + col * GRID.dx + GRID.w / 2, GRID.y + src * GRID.dy + GRID.h / 2]);
  const recvxAt = (r: number, le: number, j: number): P =>
    abs(r, [RECVX.x + j * RECVX.dx + RECVX.w / 2, RECVX.y + le * RECVX.dy + RECVX.h / 2]);
  const outAt = (r: number, j: number): P => abs(r, [OUT.x + j * OUT.dx + OUT.w / 2, OUT.y + OUT.h / 2]);
  const tokenSmallAt = (r: number, j: number): P => {
    const b = box(r);
    return [b.x + b.w - 4 * 28 - 4 + j * 28 + 12, b.y + 15];
  };
  const tokenAt = (r: number, j: number): P => {
    const big = abs(r, [TOK.x + j * TOK.dx + TOK.w / 2, TOK.y + TOK.h / 2]);
    if (r === 0) {
      const l0 = L0[0]!;
      const over: P = [l0.x + l0.w / 2 - 63 + j * 42, l0.y - 66];
      return lerpP(over, big, ease(s.focus0));
    }
    return lerpP(tokenSmallAt(r, j), big, bigness(r));
  };
  const expertAt = (r: number, j: number): P => {
    const b = box(r);
    const small: P = [b.x + b.w / 2 - 28 + j * 56, b.y + 48];
    return lerpP(small, [b.x + 296 + j * 56, b.y + 62], bigness(r));
  };

  // ---- topology mesh (behind the ranks)
  const mesh = range(N).flatMap((i) => range(N).filter((j) => j > i).map((j) => ({ i, j, line: el("line", { class: "a" }, meshLayer) })));
  const meshLabel = text(meshLayer, { x: 770, y: 28, class: "t-lbl", "text-anchor": "middle" }, "a direct link between every pair of ranks");

  // ---- overview extras
  const fromAttn = text(rankLayer, { class: "t-lbl", "text-anchor": "middle" }, "from R0's attention layer");
  const fromAttnArrow = el("line", { class: "a", "marker-end": "url(#a2av-ah)" }, rankLayer);

  // ---- ranks
  interface RankView {
    frame: SVGRectElement;
    label: SVGTextElement;
    experts: Chip[];
    readouts: SVGTextElement[];
    badge: SVGTextElement;
    inner?: Inner;
  }
  interface Inner {
    g: SVGGElement;
    gate: SVGRectElement;
    expertsBox: SVGRectElement;
    vars: { name: string; value: () => string; hot: () => boolean; node: SVGTextElement }[];
    gridTag: SVGTextElement;
    gridNote: [SVGTextElement, SVGTextElement];
    gridRows: { label: SVGTextElement; cells: SVGRectElement[] }[];
    b1: SVGLineElement;
    b1Label: SVGTextElement;
    gpu: SVGRectElement;
    b2: SVGLineElement;
    b2Label: SVGTextElement;
    shared: SVGRectElement;
    sharedSub: SVGTextElement;
    out: SVGRectElement;
  }

  const ranks: RankView[] = range(N).map((r) => {
    const frame = el("rect", { class: "b-ext", rx: 10 }, rankLayer);
    const label = text(rankLayer, { class: "t-title", "text-anchor": "middle" }, `RANK ${r}`);
    const experts = range(EPR).map((j) => new Chip(chipLayer, `E${r * EPR + j}`));
    const readouts = range(4).map(() => text(rankLayer, { class: "t-mono" }));
    const badge = text(rankLayer, { class: "t-lbl t-hot", "text-anchor": "end" });
    const view: RankView = { frame, label, experts, readouts, badge };
    if (r <= 1) view.inner = buildInner(r);
    return view;
  });

  function block(g: Element, x: number, y: number, w: number, h: number): SVGRectElement {
    return el("rect", { x, y, width: w, height: h, rx: 7, class: "b" }, g);
  }

  function buildInner(r: number): Inner {
    const g = el("g", {}, rankLayer);
    const own = tokensOf(r);
    text(g, { x: 72, y: 18, class: "t-tag" }, r === 0 ? `· HOME RANK OF ${own[0]}–${own[3]}` : `· HOLDS E${r * EPR}, E${r * EPR + 1} · HOME RANK OF ${own[0]}–${own[3]}`);

    // Row A: gate and local experts
    const gate = block(g, 16, 30, 200, 56);
    text(g, { x: 116, y: 48, class: "t-title", "text-anchor": "middle" }, "Gate (router)");
    const expertsBox = block(g, 224, 30, 200, 56);
    text(g, { x: 324, y: 44, class: "t-tag", "text-anchor": "middle" }, "LOCAL EXPERTS");

    // Row B: variables
    block(g, 16, 96, 408, 130);
    text(g, { x: 28, y: 110, class: "t-tag" }, `VARIABLES ON RANK ${r}`);
    const d = DATA[r]!;
    const rrc = recvRankCount(r);
    const rec = recvExpertCount(r);
    const fmtR = (xs: number[]) => xs.map((v, i) => `R${i}:${v}`).join(" ");
    const fmtE = (xs: number[], base = 0) => xs.map((v, i) => `E${base + i}:${v}`).join(" ");
    const rows: [string, () => string, () => boolean][] = [
      ["topk_idx", () => {
        const n = Math.ceil(clamp(s.gate, 0, 4));
        return n === 0 ? "·" : own.slice(0, n).map((t) => `${t}:E${route(t)[0]},E${route(t)[1]}`).join(" ");
      }, () => between("gate", "notify")],
      ["rank_count", () => (s.notify > 0.15 ? fmtR(d.rankCount) : "·"), () => between("notify", "fill")],
      ["expert_count", () => (s.notify > 0.15 ? fmtE(d.expertCount) : "·"), () => between("notify", "fill")],
      ["dst_slot", () => (s.fill > 0.1 ? own.map((t) => `${t}→${d.dstSlot.get(t)!.map(([x, y]) => `R${x}#${y}`).join(",")}`).join(" ") : "·"),
        () => between("fill", "others")],
      ["recv_rank_count", () => (s.others > 0.5 ? fmtR(rrc) : s.notify > 0.15 ? rrc.map((v, i) => `R${i}:${i === r ? v : "·"}`).join(" ") : "·"),
        () => between("notify", "fill") || between("others", "barrier1")],
      ["recv_expert_count", () => (s.others > 0.5 ? fmtE(rec, r * EPR) : "·"), () => between("others", "barrier1")],
      ["psum_rank", () => (s.others > 0.75 ? `[${inclusive(rrc).join(", ")}]` : "·"), () => between("others", "barrier1")],
      ["psum_expert", () => (s.others > 0.75 ? `[0, ${inclusive(rec).join(", ")}]` : "·"), () => between("others", "barrier1")],
    ];
    const vars = rows.map(([name, value, hot], i) => {
      text(g, { x: 28, y: 125 + i * 13, class: "t-mono t-muted" }, name);
      return { name, value, hot, node: text(g, { x: 140, y: 125 + i * 13, class: "t-mono" }) };
    });

    // Row C: recv_buffer
    block(g, 16, 236, 408, 84);
    const gridTag = text(g, { x: 28, y: 248, class: "t-tag" });
    const gridRows = range(N).map((row) => ({
      label: text(g, { x: 28, y: GRID.y + row * GRID.dy + 10, class: "t-mono t-muted" }),
      cells: range(4).map((c) =>
        el("rect", { x: GRID.x + c * GRID.dx, y: GRID.y + row * GRID.dy, width: GRID.w, height: GRID.h, rx: 3, class: "b-slot" }, g)),
    }));
    const gridNote: [SVGTextElement, SVGTextElement] = [
      text(g, { x: 264, y: 278, class: "t-sub" }),
      text(g, { x: 264, y: 292, class: "t-sub" }),
    ];

    const b1 = el("line", { x1: 16, x2: 424, y1: B1_Y, y2: B1_Y, class: "a a-out" }, g);
    const b1Label = text(g, { x: 424, y: B1_Y - 4, class: "t-lbl", "text-anchor": "end" });

    // Row D: recv_x and the GPU
    block(g, 16, 344, 262, 74);
    text(g, { x: 28, y: 357, class: "t-tag" }, "recv_x · ROWS GROUPED BY EXPERT");
    for (let le = 0; le < EPR; le++)
      text(g, { x: 28, y: RECVX.y + le * RECVX.dy + 14, class: "t-mono t-muted" }, `E${r * EPR + le}`);
    const gpu = block(g, 286, 344, 138, 74);
    el("rect", { x: 341, y: 352, width: 28, height: 20, rx: 3, class: "b" }, g);
    for (const px of [347, 355, 363]) {
      el("line", { x1: px, x2: px, y1: 348, y2: 352, class: "a" }, g);
      el("line", { x1: px, x2: px, y1: 372, y2: 376, class: "a" }, g);
    }
    text(g, { x: 355, y: 366, class: "t-tag", "text-anchor": "middle" }, "GPU");
    text(g, { x: 355, y: 392, class: "t-title", "text-anchor": "middle" }, "Grouped GEMM");
    text(g, { x: 355, y: 408, class: "t-sub", "text-anchor": "middle" }, "DeepGEMM");

    // Row E: shared expert. It needs only this rank's own tokens, so it runs while combine is in flight,
    // before the second barrier; only the final sum waits for that barrier.
    const shared = block(g, 16, 426, 170, 52);
    text(g, { x: 101, y: 447, class: "t-title", "text-anchor": "middle" }, "Shared expert");
    const sharedSub = text(g, { x: 101, y: 465, class: "t-sub", "text-anchor": "middle" });
    text(g, { x: 198, y: 448, class: "t-sub" }, "every token, no gate choice;");
    text(g, { x: 198, y: 462, class: "t-sub" }, "runs during combine, fewer SMs");

    const b2 = el("line", { x1: 16, x2: 424, y1: B2_Y, y2: B2_Y, class: "a a-out" }, g);
    const b2Label = text(g, { x: 424, y: B2_Y - 4, class: "t-lbl", "text-anchor": "end" });
    el("line", { x1: 101, y1: 478, x2: 101, y2: 496, class: "a", "marker-end": "url(#a2av-ah)" }, g);

    // Row F: output, after the second barrier
    const out = block(g, 16, 498, 408, 46);
    text(g, { x: 28, y: 511, class: "t-tag" }, "out[t] = Σ PARTIALS + shared_out[t]");

    return { g, gate, expertsBox, vars, gridTag, gridNote, gridRows, b1, b1Label, gpu, b2, b2Label, shared, sharedSub, out };
  }

  // ---- arrows between R0 and the small ranks (layout L1)
  const arrows = range(N - 1).map(() => ({
    notify: el("line", { class: "a" }, lineLayer),
    notifyLabel: text(lineLayer, { class: "t-lbl", "text-anchor": "middle" }, "gin.put_value"),
    fill: el("line", { class: "a a-hot" }, lineLayer),
    fillLabel: text(lineLayer, { class: "t-lbl", "text-anchor": "middle" }, "tma_store_1d / gin.put"),
  }));
  // Gate: lines from the active token to its two experts.
  const routeLines = range(2).map(() => el("polyline", { class: "a a-hot", fill: "none" }, lineLayer));

  // ---- tokens (each rank's own)
  const tokenChips = range(N).map((r) => tokensOf(r).map((t) => new Chip(chipLayer, t)));

  // ---- pieces: chips that fly from one place to another and may stay there
  interface Piece {
    chip: Chip;
    from: () => P;
    to: () => P;
    move: [Phase, number, number];
    persist: boolean;
    until?: [Phase, number, number];
    show?: () => number;
    hot: () => boolean;
    size: [number, number];
    font: number;
  }
  const pieces: Piece[] = [];
  const add = (label: string, p: Omit<Piece, "chip">) => pieces.push({ ...p, chip: new Chip(chipLayer, label) });
  const stagger = (phase: Phase, n: number, start: number, len: number) => (k: number): [Phase, number, number] => {
    const step = n > 1 ? (1 - start - len) / (n - 1) : 0;
    return [phase, start + k * step, start + k * step + len];
  };
  const cell: [number, number] = [GRID.w, GRID.h];
  const clearForCombine: [Phase, number, number] = ["combine", 0, 0.15];

  // Dispatch from R0: one copy per destination rank.
  {
    const flights = tokensOf(0).flatMap((t, j) => DATA[0]!.dstSlot.get(t)!.map(([dst, slot]) => ({ t, j, dst, slot })));
    const win = stagger("fill", flights.length, 0.05, 0.4);
    flights.forEach(({ t, j, dst, slot }, k) =>
      add(t, {
        from: () => tokenAt(0, j),
        to: () => (dst === 0 ? gridAt(0, 0, slot) : (() => { const b = box(dst); return [b.x + 60, b.y + 92] as P; })()),
        move: win(k), persist: dst === 0, until: clearForCombine,
        hot: () => t === FOCUS, size: cell, font: 9,
      }));
  }
  // Every other rank's tokens that R0 holds experts for.
  {
    const flights = range(N - 1).flatMap((i) => DATA[0]!.recv[i + 1]!.map((t, slot) => ({ t, src: i + 1, slot })));
    const win = stagger("others", flights.length, 0.05, 0.4);
    flights.forEach(({ t, src, slot }, k) =>
      add(t, {
        from: () => tokenSmallAt(src, tokensOf(src).indexOf(t)),
        to: () => gridAt(0, src, slot),
        move: win(k), persist: true, until: clearForCombine,
        hot: () => false, size: cell, font: 9,
      }));
  }
  // R1's recv_buffer, already filled by the time it is zoomed into.
  for (let src = 0; src < N; src++)
    DATA[1]!.recv[src]!.forEach((t, slot) =>
      add(t, {
        from: () => gridAt(1, src, slot), to: () => gridAt(1, src, slot),
        move: ["focus1", 0, 0.001], persist: true, until: clearForCombine, show: () => inner(1),
        hot: () => t === FOCUS, size: cell, font: 9,
      }));
  // Copy epilogue on R0 and R1: recv_buffer rows → recv_x, once per local expert.
  {
    const flights = [0, 1].flatMap((r) =>
      DATA[r]!.recvX.flatMap((group, le) => group.map((t, j) => ({ r, le, j, t }))));
    const win = stagger("copy", flights.length, 0.03, 0.3);
    flights.forEach(({ r, le, j, t }, k) => {
      const src = homeOf(t);
      const slot = DATA[r]!.recv[src]!.indexOf(t);
      add(t, {
        from: () => gridAt(r, src, slot), to: () => recvxAt(r, le, j),
        move: win(k), persist: true,
        hot: () => t === FOCUS || s.gemm > 0.5, size: [RECVX.w, RECVX.h], font: 10,
      });
    });
  }
  // Combine: one partial per (token, expert rank), written into the home rank's recv_buffer[top-k slot][t].
  {
    const flights = range(N).flatMap((h) =>
      tokensOf(h).flatMap((t) => destRanks(t).filter((sr) => sr <= 1 || h <= 1).map((sr) => ({ t, h, sr }))));
    const win = stagger("combine", flights.length, 0.18, 0.35);
    flights.forEach(({ t, h, sr }, k) => {
      const local = route(t).filter((e) => rankOf(e) === sr).length;
      const cells = () =>
        DATA[sr]!.recvX.flatMap((group, le) => group.flatMap((x, j) => (x === t ? [recvxAt(sr, le, j)] : [])));
      add(local > 1 ? `${t}Σ` : t, {
        from: () => {
          if (sr > 1) return topEdge(sr);
          const cs = cells();
          return [cs.reduce((a, p) => a + p[0], 0) / cs.length, cs.reduce((a, p) => a + p[1], 0) / cs.length];
        },
        to: () => (h <= 1 ? gridAt(h, combineSlot(t, sr), tokensOf(h).indexOf(t)) : topEdge(h)),
        move: win(k), persist: h <= 1,
        hot: () => t === FOCUS, size: cell, font: 9,
      });
    });
  }
  // Output on R0 and R1: each recv_buffer column is reduced into one token.
  {
    const flights = [0, 1].flatMap((h) => tokensOf(h).map((t, j) => ({ h, t, j })));
    const win = stagger("output", flights.length, 0.05, 0.4);
    flights.forEach(({ h, t, j }, k) =>
      add(`${t}′`, {
        from: () => {
          const ps = destRanks(t).map((sr) => gridAt(h, combineSlot(t, sr), j));
          return [ps[0]![0], ps.reduce((a, p) => a + p[1], 0) / ps.length];
        },
        to: () => outAt(h, j),
        move: win(k), persist: true,
        hot: () => t === FOCUS, size: [OUT.w, OUT.h], font: 10,
      }));
  }

  // ---------------------------------------------------------------- view

  function activeToken(): string | undefined {
    if (s.gate <= 0) return undefined;
    if (s.gate < 4 && s.notify === 0) return tokensOf(0)[Math.min(3, Math.floor(s.gate))];
    return FOCUS;
  }

  function expertHot(e: number): boolean {
    if (s.gemm > 0 && s.combine === 0) return rankOf(e) <= 1 && (rankOf(e) === 0 ? s.focus0 : s.focus1) > 0.5;
    if (s.combine > 0) return false;
    const t = activeToken();
    return !!t && route(t).includes(e);
  }

  scene.onRender(() => {
    const step = STEPS[Math.max(0, scene.stepIndex)]!;
    setText(title, step[1]);
    setText(detail, step[2]);

    // Mesh
    const meshOn = clamp(s.notify * 4) * (1 - clamp(s.focus1 * 2)) * 0.6;
    for (const { i, j, line } of mesh) {
      const [x1, y1] = center(i);
      const [x2, y2] = center(j);
      set(line, { x1, y1, x2, y2, opacity: meshOn });
    }
    set(meshLabel, { opacity: meshOn / 0.6 });

    // Overview: tokens arriving from attention
    const attn = s.tokens * (1 - clamp(s.focus0 * 3));
    const l0 = L0[0]!;
    set(fromAttn, { x: l0.x + l0.w / 2, y: l0.y - 86, opacity: attn });
    set(fromAttnArrow, { x1: l0.x + l0.w / 2, y1: l0.y - 54, x2: l0.x + l0.w / 2, y2: l0.y - 4, opacity: attn });

    // Ranks
    ranks.forEach((v, r) => {
      const b = box(r);
      const big = bigness(r);
      set(v.frame, { x: b.x, y: b.y, width: b.w, height: b.h, opacity: s.intro });

      // Label: centred and large when small; small at the top-left when big.
      const size = lerp(13, 10.5, big);
      setFont(v.label, size);
      set(v.label, { y: lerp(b.y + 24, b.y + 18, big), opacity: s.intro });
      const len = v.label.getComputedTextLength();
      set(v.label, { x: lerp(b.x + b.w / 2, b.x + 14 + len / 2, big) });

      v.experts.forEach((chip, j) => chip.place(expertAt(r, j), 44, 24, s.intro, expertHot(r * EPR + j), 11));

      // Readouts on small ranks (layout L1 only)
      const ro = clamp(s.focus0 * 2 - 1) * (1 - big) * (1 - clamp(s.focus1 * 3));
      const lines = r === 0 ? [] : readouts(r);
      v.readouts.forEach((node, i) => {
        const line = lines[i];
        setText(node, line?.[0] ?? "");
        set(node, { x: b.x + 14, y: b.y + 80 + i * 16, opacity: line ? ro : 0, class: line?.[1] ? "t-mono t-hot" : "t-mono" });
      });

      // Badge for the second barrier on the small ranks of layout L2
      const arrived2 = r >= 2 && s.barrier2 > r * 0.25;
      setText(v.badge, arrived2 ? "✓ at gpu_barrier" : "");
      set(v.badge, { x: b.x + b.w - 10, y: b.y + 82 });

      if (v.inner) renderInner(r, v.inner);
    });

    // Arrows R0 ↔ small ranks
    const fade = (1 - clamp(s.barrier1 * 3)) * (1 - clamp(s.focus1 * 3));
    const both = s.others > 0;
    arrows.forEach((a, i) => {
      const d = i + 1;
      const b0 = box(0);
      const bd = box(d);
      const x1 = b0.x + b0.w;
      const x2 = bd.x;
      const cy = bd.y + bd.h / 2;
      const nOn = clamp(s.notify * 4) * fade;
      const fOn = clamp(s.fill * 4) * fade;
      set(a.notify, { x1, x2: x2 - 2, y1: cy - 12, y2: cy - 12, opacity: nOn, "marker-end": "url(#a2av-ah)", "marker-start": both ? "url(#a2av-ah)" : "none" });
      set(a.notifyLabel, { x: (x1 + x2) / 2, y: cy - 17, opacity: nOn });
      set(a.fill, { x1, x2: x2 - 2, y1: cy + 10, y2: cy + 10, opacity: fOn, "marker-end": "url(#a2av-ah-hot)", "marker-start": both ? "url(#a2av-ah-hot)" : "none" });
      set(a.fillLabel, { x: (x1 + x2) / 2, y: cy + 25, opacity: fOn });
    });

    // Gate: lines from the active token to its experts
    const t = activeToken();
    const gateOn = between("gate", "notify") ? 1 : 0;
    routeLines.forEach((line, k) => {
      if (!t || !gateOn) return set(line, { opacity: 0 });
      const [x1, y1] = tokenAt(0, tokensOf(0).indexOf(t));
      const e = route(t)[k]!;
      const [x2, y2] = expertAt(rankOf(e), e % EPR);
      // Local experts: straight across. Remote experts: up over R0's box, down the gap between R0 and the other
      // ranks, then in from the left, so the line never crosses E0/E1 or another rank.
      const top = box(0).y - 10;
      const gapX = (box(0).x + box(0).w + box(1).x) / 2;
      const points = rankOf(e) === 0
        ? [[x1 + 18, y1], [x2 - 22, y2]]
        : [[x1, y1 - 9], [x1, top], [gapX, top], [gapX, y2 - 18], [x2, y2 - 18], [x2, y2 - 12]];
      set(line, { points: points.map((q) => q.join(",")).join(" "), opacity: 0.8 });
    });

    // Tokens
    tokenChips.forEach((chips, r) =>
      chips.forEach((chip, j) => {
        const big = r === 0 ? 1 : bigness(r);
        let op: number;
        if (r === 0) op = s.tokens;
        else op = r === 1 ? Math.max(clamp(s.others * 4), inner(1)) : clamp(s.others * 4) * (1 - clamp(s.focus1 * 3));
        const tk = tokensOf(r)[j]!;
        chip.place(tokenAt(r, j), lerp(24, TOK.w, big), lerp(14, TOK.h, big), op, r === 0 && tk === t, lerp(8, 10, big));
      }));

    // Pieces
    for (const p of pieces) {
      const prog = clamp((s[p.move[0]] - p.move[1]) / (p.move[2] - p.move[1]));
      let op = prog <= 0 ? 0 : Math.min(1, prog * 8);
      if (!p.persist && prog > 0.85) op = prog >= 1 ? 0 : (1 - prog) / 0.15;
      if (p.until) op *= 1 - clamp((s[p.until[0]] - p.until[1]) / (p.until[2] - p.until[1]));
      if (p.show) op *= p.show();
      if (op <= 0.001) {
        p.chip.place([0, 0], 0, 0, 0, false);
        continue;
      }
      const pos = lerpP(p.from(), p.to(), ease(prog));
      p.chip.place(pos, p.size[0], p.size[1], op, p.hot(), p.font);
    }
  });

  /** Text lines on a small rank, with whether each line just changed. */
  function readouts(r: number): [string, boolean][] {
    const lines: [string, boolean][] = [];
    const fromR0 = DATA[r]!.recv[0]!;
    if (s.others > 0.6) {
      lines.push([`recv_rank_count = [${recvRankCount(r).join(", ")}]`, between("others", "barrier1")]);
      const total = recvRankCount(r).reduce((a, b) => a + b, 0);
      lines.push([`recv_buffer: ${total} tokens from 4 ranks`, between("others", "barrier1")]);
      lines.push([`psum_rank = [${inclusive(recvRankCount(r)).join(", ")}]`, between("others", "barrier1")]);
    } else {
      if (s.notify > 0.6) lines.push([`recv_rank_count[R0] = ${fromR0.length}`, between("notify", "fill")]);
      if (s.fill > 0.6) lines.push([`recv_buffer[R0] = ${fromR0.length ? fromR0.join(" ") : "—"}`, between("fill", "others")]);
    }
    if (s.barrier1 > r * 0.25) lines.push(["✓ at gpu_barrier", between("barrier1", "focus1")]);
    return lines;
  }

  function renderInner(r: number, v: Inner): void {
    const b = box(r);
    set(v.g, { transform: `translate(${b.x} ${b.y})`, opacity: inner(r), display: inner(r) <= 0.001 ? "none" : "inline" });
    if (inner(r) <= 0.001) return;

    set(v.gate, { class: between("gate", "notify") && r === 0 ? "b-focus" : "b" });
    set(v.expertsBox, { class: s.gemm > 0 && s.combine === 0 ? "b-focus" : "b" });
    for (const row of v.vars) {
      setText(row.node, row.value());
      set(row.node, { class: row.hot() ? "t-mono t-hot" : "t-mono" });
    }

    const combining = s.combine > 0.1;
    setText(v.gridTag, combining ? "recv_buffer[top-k slot][t] · REUSED BY COMBINE" : "recv_buffer[src][slot]");
    setText(v.gridNote[0], combining ? "row k: the sender holds" : "written by the senders;");
    setText(v.gridNote[1], combining ? "the token's choice k" : "this rank never calls recv");
    // Combine uses only TOPK rows (k0, k1); the rank rows R2, R3 are unused.
    v.gridRows.forEach((row, i) => {
      setText(row.label, combining ? (i < 2 ? `k${i}` : "") : `R${i}`);
      for (const c of row.cells) set(c, { opacity: combining && i >= 2 ? 0 : 1 });
    });

    const n1 = 1 + [1, 2, 3].filter((i) => s.barrier1 > i * 0.25).length;
    set(v.b1, { opacity: clamp(s.barrier1 * 4), class: between("barrier1", "focus1") ? "a a-out a-hot" : "a a-out" });
    setText(v.b1Label, `gpu_barrier · ${n1}/4 ranks arrived`);
    set(v.b1Label, { opacity: clamp(s.barrier1 * 4) });

    set(v.gpu, { class: between("gemm", "combine") ? "b-focus" : "b" });

    const n2 = 1 + [1, 2, 3].filter((i) => s.barrier2 > i * 0.25).length;
    set(v.b2, { opacity: clamp(s.barrier2 * 4), class: between("barrier2", "output") ? "a a-out a-hot" : "a a-out" });
    setText(v.b2Label, `gpu_barrier · ${n2}/4 ranks arrived`);
    set(v.b2Label, { opacity: clamp(s.barrier2 * 4) });

    set(v.shared, { class: between("combine", "barrier2") ? "b-focus" : "b" });
    const own = tokensOf(r);
    setText(v.sharedSub, s.combine > 0.3 ? `shared_out[${own[0]}–${own[3]}]` : "runs during combine");
    set(v.out, { class: s.output > 0 ? "b-focus" : "b" });
  }

  // ---------------------------------------------------------------- timeline

  const [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13] = STEPS.map((x) => x[0]) as [
    string, string, string, string, string, string, string, string, string, string, string, string, string, string,
  ];
  tl.addLabel(l0)
    .to(s, { intro: 1, duration: 0.8 }, l0)
    .addLabel(l1, "+=0.6")
    .to(s, { tokens: 1, duration: 0.8 }, l1)
    .addLabel(l2, "+=0.6")
    .to(s, { focus0: 1, duration: 1.2 }, l2)
    .addLabel(l3, "+=0.4")
    .to(s, { gate: 4, duration: 4, ease: "none" }, l3)
    .addLabel(l4, "+=0.6")
    .to(s, { notify: 1, duration: 1.6 }, l4)
    .addLabel(l5, "+=0.8")
    .to(s, { fill: 1, duration: 3, ease: "none" }, l5)
    .addLabel(l6, "+=0.8")
    .to(s, { others: 1, duration: 2.6, ease: "none" }, l6)
    .addLabel(l7, "+=0.8")
    .to(s, { barrier1: 1, duration: 2, ease: "none" }, l7)
    .addLabel(l8, "+=0.8")
    .to(s, { focus1: 1, duration: 1.4 }, l8)
    .addLabel(l9, "+=0.6")
    .to(s, { copy: 1, duration: 3.2, ease: "none" }, l9)
    .addLabel(l10, "+=0.6")
    .to(s, { gemm: 1, duration: 1.6 }, l10)
    .addLabel(l11, "+=0.6")
    .to(s, { combine: 1, duration: 4.2, ease: "none" }, l11)
    .addLabel(l12, "+=0.6")
    .to(s, { barrier2: 1, duration: 2, ease: "none" }, l12)
    .addLabel(l13, "+=0.6")
    .to(s, { output: 1, duration: 1.8, ease: "none" }, l13)
    .to({}, { duration: 3 }); // hold on the finished figure

  if (fallback) fallback.style.display = "none";
  return () => {
    svg.remove();
    desc.remove();
    if (fallback) fallback.style.display = "";
  };
}
