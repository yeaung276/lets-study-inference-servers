// Zero-overhead scheduling (ZOS) figure on pages/concepts.html, in two switchable modes: with ZOS and without it.
// Top: a profile with a CPU lane and a GPU lane, slices drawn as a cursor sweeps through time.
// Bottom: what CPU memory and GPU memory hold at the cursor, with token ids moving between them (H2D, D2H, D2D),
// and between the two boxes the CUDA stream: the queue the CPU launches into and the GPU runs from, in order.
// Order of work follows DeepInfer's Worker.execute_model_step, no MTP. With ZOS, in each GPU step the CPU
// post-processes the previous step, queues this step's sample (plus its D2H copy and an event), schedules the next
// step, prepares and queues it (H2D copy plus kernels), then waits for the event (the only sync). Without ZOS
// (DEEPINFER_ZERO_OVERHEAD_SCHEDULING off) each step runs in order: schedule, prep + launch, launch sample, wait,
// post-process; the GPU idles while the CPU works, and input_ids come from the CPU by H2D instead of the D2D gather.
// Model/view: the timeline tweens `s.t` (time in schematic units); render() derives everything from it.

import { mount, type Scene } from "..";

const NS = "http://www.w3.org/2000/svg";
const W = 1040;
const H = 608;
const X0 = 100;
const X1 = 1020;
/** Seconds of animation per schematic time unit. One speed for the whole figure, so the cursor moves at a constant
 *  pace and the profile stays true to time; slow enough to follow the moving data. */
const U = 1.75;

type LaneId = "cpu" | "gpu";
const LANE: Record<LaneId, { y: number; h: number; label: string }> = {
  cpu: { y: 30, h: 36, label: "CPU" },
  gpu: { y: 108, h: 36, label: "GPU" },
};

/** Step names, indexed by j: 0 = N−1, 1 = N, 2 = N+1, 3 = N+2, 4 = N+3. */
const STEP = ["N−1", "N", "N+1", "N+2", "N+3"];

// ---------------------------------------------------------------- data

const REQS = ["r1", "r2", "r3"];
const PROMPT = [["The", "cat"], ["I", "like"], ["Hi"]];
/** SAMPLED[j][i]: token request i gets from sample(step j). r2 samples <eos> in step N and finishes. */
const SAMPLED = [
  ["sat", "green", "there"],
  ["on", "<eos>", "how"],
  ["the", "(unused)", "are"],
];
/** Token ids. Illustrative, not from a real vocabulary; the word is shown under the id only for reading. */
const ID: Record<string, number> = {
  The: 671, cat: 6830, sat: 7731, on: 389, the: 279,
  I: 40, like: 1093, green: 6176,
  Hi: 13347, there: 1070, how: 703, are: 527, "<eos>": 1, "(unused)": 4421,
};
const idOf = (w: string) => String(ID[w]);
const FINISH = { req: 1, step: 1 };
/** The input token of request i in step j: the token step j−1 produced (for step N−1, the prompt's last token). */
const inputWord = (j: number, i: number) => (j === 0 ? PROMPT[i]!.at(-1)! : SAMPLED[j - 1]![i]!);

interface Slice {
  lane: LaneId;
  t0: number;
  t1: number;
  op: string;
  step: string;
  /** Drawn dashed and never highlighted: the CPU waiting, or (without ZOS) the GPU idle. */
  wait?: boolean;
  /** GPU only: when the CPU queued it. */
  launch?: number;
}
/** CPU phase ends within each 8-unit ZOS period: launch sample, schedule, prep + launch. Wait runs to 8. */
const L_END = 3.25;
const S_END = 4.75;
const P_END = 6.5;
/** Without ZOS, one step's period: post-process 2, schedule 1.5, prep + launch 1.75 (the same CPU costs as with ZOS),
 *  then the GPU's inputs 1, forward 5, sample 2 while the CPU launches sample and waits. */
const NZ = { sched: 2, prep: 3.5, launch: 5.25, launchSample: 6.5, period: 13.25 };

/** Items in the CUDA stream: e = enqueued by the CPU, s/f = the GPU starts/finishes it. An event finishes the
 *  moment the GPU reaches it. "inputs" includes the H2D copy; "sample" includes the D2H copy. */
interface StreamItem {
  op: string;
  sub: string;
  e: number;
  s: number;
  f: number;
  event?: boolean;
  /** inputs/forward launched by prep: the block flies from the scheduled batch into the stream, landing at e. */
  fly?: [number, number];
}
/** Items fade in just before their launch ends and work items fade out just before they finish, so each step's
 *  end state shows the queue as it is at that moment. An event stays, highlighted, for a moment after it's reached. */
const ENTER = 0.3;
const LEAVE = 0.35;
const leaveWin = (it: StreamItem): [number, number] => (it.event ? [it.f, it.f + LEAVE] : [it.f - LEAVE, it.f]);

type Win = [number, number];
/** [label, title, detail, from t, to t]. */
type StepDef = [string, string, string, number, number];

/** Everything that depends on the mode: when each piece of work runs, which requests each step has, and the steps. */
function timing(zos: boolean) {
  const P = zos ? 8 : NZ.period;
  /** First step scheduled inside the figure. With ZOS, step N's batch is ready at t = 0. */
  const FIRST = zos ? 2 : 1;
  const T_END = zos ? 18 : 28.5;
  const xs = (t: number) => X0 + (Math.min(t, T_END) / T_END) * (X1 - X0);
  /** Requests in GPU step j, in row order. r2 samples <eos> in step N. With ZOS it is still in N+1 (the CPU scheduled
   *  it before reading <eos>) and leaves at N+2, so r3 moves up from row 2 to row 1: its remap in N+2 is 2. Without
   *  ZOS the CPU reads <eos> before scheduling N+1, so r2 leaves at N+1. */
  const batchOf = (j: number) => (j >= (zos ? 3 : 2) ? [0, 2] : [0, 1, 2]);
  /** GPU step j: inputs 1, forward 5, sample 2. With ZOS back to back; without, after the CPU's prep + launch(j). */
  const gpuStart = (j: number) => (zos ? 8 * (j - 1) : P * (j - 1) + NZ.launch);
  const smpEnd = (j: number) => gpuStart(j) + 8;
  const fwd = (j: number): Win => [gpuStart(j) + 1, gpuStart(j) + 6];
  const sampleWin = (j: number): Win => [gpuStart(j) + 6, smpEnd(j)];
  const schedWin = (j: number): Win =>
    zos ? [8 * (j - 2) + L_END, 8 * (j - 2) + S_END] : [P * (j - 1) + NZ.sched, P * (j - 1) + NZ.prep];
  const prep = (j: number): Win =>
    zos ? [8 * (j - 2) + S_END, 8 * (j - 2) + P_END] : [P * (j - 1) + NZ.prep, P * (j - 1) + NZ.launch];
  const launchSampleEnd = (j: number) => (zos ? 8 * (j - 1) + L_END : P * (j - 1) + NZ.launchSample);
  /** Under the GPU lane: the span of step j. */
  const stepSpan = (j: number): Win => (zos ? [8 * (j - 1), 8 * j] : [P * (j - 1), P * j]);

  // inputs(j). With ZOS: H2D brings the batch data in, the metadata kernel writes the graph's pos and slots
  // placeholders, then the gather (D2D) fills input_ids from the sampled token ids. Without ZOS: H2D brings the batch
  // data and the input token ids (straight into input_ids), then the metadata kernel; no gather.
  const g0 = gpuStart;
  const h2d = (j: number): Win => (zos ? [g0(j), g0(j) + 0.35] : [g0(j), g0(j) + 0.5]);
  const mdk = (j: number): Win => (zos ? [g0(j) + 0.35, g0(j) + 0.65] : [g0(j) + 0.5, g0(j) + 1]);
  const d2d = (j: number): Win => (zos ? [g0(j) + 0.65, g0(j) + 1] : [g0(j) + 1, g0(j) + 1]);
  /** The copy that writes input_ids: the gather with ZOS, H2D without. */
  const idsWin = zos ? d2d : h2d;
  const d2h = (j: number): Win => [smpEnd(j) - 0.8, smpEnd(j)]; // tail of sample(j)
  /** The metadata kernel writes row k's pos (from its length) and slots: both fly from the batch data into the graph inputs. */
  const mdkFly = (j: number, k: number): Win => {
    const [a, b] = mdk(j);
    const len = b - a;
    return [a + 0.23 * len * k, a + 0.53 * len + 0.23 * len * k];
  };
  /** sample(j) writes row k of the sampled token ids: the token id travels from the output's row k along the sample
   *  arrow, one row after another, before the D2H copy starts. */
  const sampleFly = (j: number, k: number): Win => [smpEnd(j) - 1.45 + 0.1 * k, smpEnd(j) - 1 + 0.1 * k];
  /** post-process(j) runs as soon as sample(j) is done; it moves request i's token into REQUESTS. */
  const postFlight = (j: number, i: number): Win => [smpEnd(j) + 0.3 + 0.4 * i, smpEnd(j) + 0.9 + 0.4 * i];
  const FINISHED_AT = smpEnd(FINISH.step) + 1.9;
  /** Time prep(j) writes the batch's r-th request's last entry (remap with ZOS, the input token without). */
  const metaAt = (j: number, r: number) => prep(j)[0] + 0.25 + 0.45 * r;
  /** Once the last entry is written, prep(j)'s batch is complete: it stays highlighted until the launch ends. */
  const packedAt = (j: number) => metaAt(j, batchOf(j).length - 1);
  /** Inside schedule(j): first every chosen row counts its tokens, then each request and its count fly into the batch. */
  const countWin = (j: number): Win => [schedWin(j)[0] + 0.05, schedWin(j)[0] + 0.75];
  const schedFly = (j: number, i: number): Win => {
    const a = schedWin(j)[0] + 0.8 + 0.08 * i;
    return [a, a + 0.35];
  };

  // Profile slices.
  const SLICES: Slice[] = [];
  for (let j = 1; j <= 3; j++) {
    const b = P * (j - 1);
    if (zos) {
      SLICES.push(
        { lane: "cpu", t0: b, t1: b + 2, op: "post-process", step: STEP[j - 1]! },
        { lane: "cpu", t0: b + 2, t1: b + L_END, op: "launch", step: `sample ${STEP[j]}` },
        { lane: "cpu", t0: b + L_END, t1: b + S_END, op: "schedule", step: STEP[j + 1]! },
        { lane: "cpu", t0: b + S_END, t1: b + P_END, op: "prep + launch", step: STEP[j + 1]! },
        { lane: "cpu", t0: b + P_END, t1: b + 8, op: "wait for", step: `sample ${STEP[j]}`, wait: true },
      );
    } else {
      SLICES.push(
        { lane: "cpu", t0: b, t1: b + NZ.sched, op: "post-process", step: STEP[j - 1]! },
        { lane: "cpu", t0: b + NZ.sched, t1: b + NZ.prep, op: "schedule", step: STEP[j]! },
        { lane: "cpu", t0: b + NZ.prep, t1: b + NZ.launch, op: "prep + launch", step: STEP[j]! },
        { lane: "cpu", t0: b + NZ.launch, t1: b + NZ.launchSample, op: "launch", step: `sample ${STEP[j]}` },
        { lane: "cpu", t0: b + NZ.launchSample, t1: b + P, op: "wait for", step: `sample ${STEP[j]}`, wait: true },
        { lane: "gpu", t0: b, t1: b + NZ.launch, op: "idle", step: "", wait: true },
      );
    }
    const g = gpuStart(j);
    SLICES.push(
      { lane: "gpu", t0: g, t1: g + 1, op: "inputs", step: STEP[j]!, launch: prep(j)[1] },
      { lane: "gpu", t0: g + 1, t1: g + 6, op: "forward", step: STEP[j]!, launch: prep(j)[1] },
      { lane: "gpu", t0: g + 6, t1: g + 8, op: "sample", step: STEP[j]!, launch: launchSampleEnd(j) },
    );
  }
  const VISIBLE = SLICES.filter((sl) => sl.t0 < T_END);
  /** Launch arrows: from the end of the CPU slice that queues the work to the start of the GPU slice. */
  const LAUNCHES: Win[] = [];
  for (let j = 1; j <= 3; j++) LAUNCHES.push([prep(j)[1], gpuStart(j)], [launchSampleEnd(j), fwd(j)[1]]);
  const launches = LAUNCHES.filter(([a]) => a >= 0 && a < T_END).sort((x, y) => x[0] - y[0]);

  const STREAM: StreamItem[] = [];
  for (let j = 1; j <= 3; j++) {
    const g = gpuStart(j);
    const pe = prep(j)[1];
    // Blocks launched inside the figure fly out of the batch at the end of prep, forward just behind inputs.
    const fly = (k: number): Win | undefined => (pe <= 0 ? undefined : [pe - 0.5 + 0.1 * k, pe - 0.1 + 0.1 * k]);
    STREAM.push(
      { op: "inputs", sub: STEP[j]!, e: fly(0)?.[1] ?? pe - ENTER, s: g, f: g + 1, fly: fly(0) },
      { op: "forward", sub: STEP[j]!, e: fly(1)?.[1] ?? pe - ENTER, s: g + 1, f: g + 6, fly: fly(1) },
      { op: "sample", sub: STEP[j]!, e: launchSampleEnd(j) - ENTER, s: g + 6, f: g + 8 },
      { op: "event", sub: `sample ${STEP[j]} done`, e: launchSampleEnd(j) - ENTER, s: g + 8, f: g + 8, event: true },
    );
  }

  /** When the GPU learns request i owns its k-th slot: the first slot was there before t = 0; a later one arrives with
   *  the batch data (H2D) of the first step whose length needs it. r2's slots are freed once it has left: with ZOS
   *  after step N+1 is done, without ZOS when step N+1's batch data arrives without it. */
  const freeAt = zos ? smpEnd(2) : h2d(2)[1];
  const slotKnown = (i: number, k: number, t: number) => {
    if (i === FINISH.req && t >= freeAt) return false;
    if (k === 0) return true;
    const j = [1, 2, 3].find((j) => batchOf(j).includes(i) && PROMPT[i]!.length + j > k * SLOT_TOKENS);
    return j !== undefined && t >= h2d(j)[1];
  };
  /** Cell state of request i's token position p: "full" once a forward has stored it, "hot" while one is storing it. */
  const cellState = (i: number, p: number, t: number): "full" | "hot" | "empty" => {
    if (p < PROMPT[i]!.length) return "full"; // stored before step N
    for (let j = 1; j <= 3; j++) {
      if (!batchOf(j).includes(i) || p !== PROMPT[i]!.length + j - 1) continue;
      if (t >= fwd(j)[1]) return "full";
      if (within(t, fwd(j))) return "hot";
    }
    return "empty";
  };

  const STEPS: StepDef[] = zos ? [
    ["post-process(N−1)", "Step N is on the GPU; the CPU post-processes step N−1",
      "The CPU reads step N−1's token ids from the pinned buffer and appends them to the requests. The dashed boxes at the end of each request are tokens the CPU doesn't have yet, each labelled with the step that produces it: N−1 is on its way in, N is being made by step N on the GPU. Meanwhile the GPU runs step N: inputs fills the CUDA graph's input placeholders, then forward replays the graph.", 0, 2],
    ["launch sample(N)", "The CPU launches sample(N) into the CUDA stream",
      "The CUDA stream is a queue: the CPU appends work at the tail and returns at once; the GPU runs items from the head, in order. Behind sample(N) the CPU also adds an event, a marker it can wait on later. Dashed boxes are queued work the GPU hasn't started.", 2, L_END],
    ["schedule(N+1)", "schedule(N+1): the gap",
      "The CPU picks the requests for step N+1 and counts each one's length: the tokens step N+1 will read. That is the tokens the CPU has, plus step N's token (N), which sample N hasn't produced yet. The highlighted N+1 is the token step N+1 is scheduled to produce; it is not counted, because step N+1 doesn't read it. Each request moves into the scheduled batch with its length, GPU memory slots and sampling settings. The CPU also can't know r2 is about to finish, so r2 stays in the batch.", L_END, S_END],
    ["prep + launch(N+1)", "The CPU prepares step N+1 and queues it",
      "The CPU adds remap to the scheduled batch, one request at a time: the request's row in step N's batch, found by comparing the two batches. Remap is what lets the GPU find token N by itself later. Once remap is in, the batch is complete and the CPU launches step N+1 from it: two blocks go into the stream. inputs: the H2D copy of the batch (lengths, slots, remap), then two GPU kernels that fill the CUDA graph's input placeholders from it. forward: the replay of the CUDA graph. The blocks are commands that point at the batch; its data crosses to the GPU when the GPU reaches inputs.", S_END, P_END],
    ["sample(N)", "sample(N) on the GPU; the CPU waits",
      "sample reads each request's hidden state from the graph's output, turns it into a score for every possible token, and picks one: on, <eos> and how. It writes them to the sampled token ids, which stay in GPU memory, and copies them to the pinned buffer, CPU memory the GPU can copy into directly (D2H). The CPU blocks until the GPU reaches the event after sample(N): the only sync in the step.", P_END, 8],
    ["GPU fills the gap", "Step N+1 starts at once: the GPU fills in the missing tokens",
      "Step N+1's work is already next in the stream. inputs runs three copies. H2D brings the batch data in from the scheduled batch. The metadata kernel writes each request's pos (where its input token sits, counting from 0) and slots into the CUDA graph's placeholders. The gather (D2D) copies step N's token ids into input_ids, each row reading the sampled row its remap names. The CPU starts post-processing step N.", 8, 9],
    ["post-process(N)", "post-process(N): the CPU catches up",
      "It appends on, <eos> and how and sends the outputs to clients. r2 sampled <eos>, so it is marked finished. But r2 is already running in step N+1: that row is wasted work, the cost of scheduling one step ahead. Meanwhile the GPU replays the CUDA graph (forward): it reads the input placeholders and the earlier keys and values in the GPU memory slots, stores the input token's keys and values there, and writes one hidden state per request to the output.", 9, 10],
    ["schedule(N+2)", "The CPU launches sample(N+1), then schedules step N+2",
      "It now knows r2 has finished, so step N+2 has only r1 and r3: r1 in row 0, r3 moves up to row 1. Each reads the tokens the CPU has plus step N+1's token (N+1), which isn't sampled yet: r1 reads 5, r3 reads 4. The highlighted N+2 is the token step N+2 will produce. r1's 5 tokens no longer fit in one GPU memory slot, so r1 gets a second slot.", 10, 8 + S_END],
    ["The cycle repeats", "The cycle repeats",
      "The CPU prepares and queues step N+2. remap is now [0, 2]: r3 is in row 1 of step N+2's batch but was in row 2 of step N+1's, so on the GPU r3's input reads sampled row 2. Then the CPU waits for the event. The GPU samples the, a token for r2 that nobody will use, and are, and copies them back.", 8 + S_END, 16],
    ["post-process(N+1)", "post-process(N+1)",
      "The CPU appends the and are; r2's token is dropped because r2 has finished. The GPU is already running step N+2 with r1 and r3. Across the whole profile the GPU lane has no idle gaps: the stream never runs empty.", 16, 18],
  ] : [
    ["post-process(N−1)", "The CPU post-processes step N−1; the GPU is idle",
      "Without ZOS the CPU works on one step at a time, in order. It reads step N−1's token ids from the pinned buffer and appends them to the requests. Nothing is queued in the CUDA stream, so the GPU waits.", 0, NZ.sched],
    ["schedule(N)", "schedule(N)",
      "The CPU picks the requests for step N and counts each one's length: the tokens step N will read. All of them are already on the CPU, including the input token (N−1), so nothing is missing. The highlighted N is the token step N is scheduled to produce; it is not counted, because step N doesn't read it. The GPU is still idle.", NZ.sched, NZ.prep],
    ["prep + launch(N)", "The CPU prepares step N and queues it",
      "The CPU adds each request's input token to the scheduled batch: the last token it has. There is no remap: nothing needs finding on the GPU. Once the batch is complete, the CPU launches step N from it: inputs and forward go into the stream.", NZ.prep, NZ.launch],
    ["launch sample(N)", "The CPU launches sample(N); the GPU runs inputs(N)",
      "The CPU appends sample(N) and an event to the stream. On the GPU, inputs runs two copies. H2D brings the batch data in and copies each input token id straight from the scheduled batch into input_ids. The metadata kernel writes each request's pos (where its input token sits, counting from 0) and slots into the CUDA graph's placeholders.", NZ.launch, NZ.launchSample],
    ["forward(N)", "forward(N) on the GPU; the CPU waits",
      "The GPU replays the CUDA graph: it reads the input placeholders and the earlier keys and values in the GPU memory slots, stores the input token's keys and values there, and writes one hidden state per request to the output. The CPU has nothing to do until sample(N) is done.", NZ.launchSample, gpuStart(1) + 6],
    ["sample(N)", "sample(N) on the GPU; the CPU still waits",
      "sample reads each request's hidden state, turns it into a score for every possible token, and picks one: on, <eos> and how. It writes them to the sampled token ids and copies them to the pinned buffer (D2H). The CPU blocks until the GPU reaches the event after sample(N).", gpuStart(1) + 6, P],
    ["post-process(N)", "post-process(N): the GPU is idle again",
      "It appends on, <eos> and how and sends the outputs to clients. r2 sampled <eos>, so it is marked finished. The stream is empty: the GPU waits while the CPU post-processes, schedules and prepares the next step.", P, P + NZ.sched],
    ["schedule(N+1)", "schedule(N+1): the CPU already knows r2 finished",
      "The CPU has read <eos>, so step N+1 has only r1 and r3, with r3 in row 1. No row is wasted. r1 reads 4 tokens and r3 reads 3, all already on the CPU. The highlighted N+1 is the token step N+1 will produce.", P + NZ.sched, P + NZ.prep],
    ["prep + launch(N+1)", "The CPU prepares step N+1 and queues it",
      "Each request's input token goes into the batch: on for r1, how for r3. Then inputs and forward go into the stream.", P + NZ.prep, P + NZ.launch],
    ["launch sample(N+1)", "The CPU launches sample(N+1); the GPU runs inputs(N+1)",
      "H2D brings the batch data in and copies on and how straight into input_ids; the metadata kernel writes pos and slots.", P + NZ.launch, P + NZ.launchSample],
    ["forward(N+1)", "forward(N+1) on the GPU; the CPU waits",
      "The GPU replays the CUDA graph for r1 and r3, storing the keys and values of on and how in their GPU memory slots.", P + NZ.launchSample, gpuStart(2) + 6],
    ["sample(N+1)", "sample(N+1) on the GPU; the CPU waits",
      "sample picks the and are and copies them to the pinned buffer (D2H).", gpuStart(2) + 6, 2 * P],
    ["post-process(N+1)", "post-process(N+1): the GPU is idle",
      "The CPU appends the and are. In every step the GPU sits idle while the CPU post-processes, schedules and prepares: here 5.25 of every 13.25 time units. Steps N and N+1 finish at 26.5; with ZOS the same two steps finish at 16.", 2 * P, T_END],
  ];

  return {
    FIRST, T_END, xs, batchOf, gpuStart, smpEnd, fwd, sampleWin, schedWin, prep, stepSpan, h2d, mdk, d2d, idsWin,
    d2h, mdkFly, sampleFly, postFlight, FINISHED_AT, metaAt, packedAt, countWin, schedFly, VISIBLE, launches, STREAM,
    slotKnown, cellState, STEPS,
  };
}

// ---------------------------------------------------------------- layout

const BOX_Y = 196;
const CPU_BOX = { x: 20, y: BOX_Y, w: 400, h: 400 };
const GPU_BOX = { x: 560, y: BOX_Y, w: 460, h: 400 };
/** The CUDA stream sits in the gap between the two boxes: head at the top (the GPU takes from it, towards the GPU
 *  box), tail at the bottom (CPU launches append, coming from the CPU box). */
const GAP_X = (CPU_BOX.x + CPU_BOX.w + GPU_BOX.x) / 2;
const STRM = { x: CPU_BOX.x + CPU_BOX.w + 8, y: BOX_Y + 86, w: GPU_BOX.x - CPU_BOX.x - CPU_BOX.w - 16, h: 210 };
const ITEM = { w: STRM.w - 12, h: 32, gap: 6, top: STRM.y + 40 };
const CHIP = { w: 58, h: 22 };
const SUB_DY = 21; // word under a token chip
const PIN_X = [230, 300, 370];
const ROW_Y = BOX_Y + 70; // pinned buffer row; the D2H arrow joins it to the GPU box
// GPU side: one row per request, in batch order (row 0, 1, 2). Left to right: sampled token ids → (D2D) → CUDA graph
// inputs → CUDA graph (forward) → output → (sample) back to sampled token ids. Below: batch data and GPU memory slots.
const GR_Y = [314, 352, 390];
const GCHIP = 50; // token chip width on the GPU side
const SMP_CX = 610;
/** CUDA graph input placeholders: fixed GPU buffers the recorded forward reads on every replay. Column centres. */
const GIN = { x: 676, w: 172, ids: 712, pos: 764, slots: 804, name: 838 };
const GRAPH = { x: 862, w: 64 };
const OUT = { x: 940, w: 74, cells: 6 };
const BAND = { y: 280, h: 140 };
/** Batch data from H2D, next to the GPU memory slots. Column centres. */
const BATCH = { x: 574, tagY: 467, headY: 484, rowY: 501, dy: 16, req: 586, len: 620, slots: 664, remap: 720 };
/** Bordered boxes around the batch data and the GPU memory slots, side by side under the band. */
const LOW = { y: 452, h: 138 };
const BATCH_BOX = { x: 566, w: 206 };
const SLOT_BOX = { x: 782, w: 230 };
const REQ = { x: 92, dx: 56, w: 50, y: [BOX_Y + 150, BOX_Y + 190, BOX_Y + 230] };
const SCHED_Y = BOX_Y + 320;
const SCHED_DY = 28; // rows of the scheduled batch
/** Column centres of the scheduled batch: request, length, GPU memory slots, sampling settings, remap (added in prep). */
const SCHED_COL = { name: 52, len: 106, slots: 188, samp: 272, remap: 352 };
const META_FROM: P = [385, SCHED_Y + SCHED_DY];
/** Where launched blocks leave the scheduled batch (centre). */
const BATCH_OUT: P = [230, SCHED_Y + 14];
/** Per-request batch data. Slots and sampling settings are illustrative. */
/** GPU memory slots each request owns, in order. A slot holds SLOT_TOKENS tokens, so a request uses ceil(length / 4). */
const SLOT_IDS = [[3, 9], [4, 11], [7, 12]];
const SLOT_TOKENS = 4;
const slotsFor = (i: number, length: number) => `[${SLOT_IDS[i]!.slice(0, Math.ceil(length / SLOT_TOKENS)).join(", ")}]`;
const SAMPLING = ["temp 0.7", "temp 1.0", "greedy"];
const META_TO: P = [650, BATCH.rowY + 8];
/** GPU memory slots grid: slot ids 0–12, each with SLOT_TOKENS cells (one per token whose results it keeps). */
/** Wrapped: PER_ROW slots per row, rows DY apart. */
const GRID = { x: 792, y: 488, dx: 31, w: 24, h: 25, n: 14, perRow: 7, dy: 56 };
type P = [number, number];
const pinAt = (i: number): P => [PIN_X[i]!, ROW_Y];
/** Row k of the sampled token ids / of the graph's input_ids. */
const smpAt = (k: number): P => [SMP_CX, GR_Y[k]!];
const inAt = (k: number): P => [GIN.ids, GR_Y[k]!];
const reqAt = (i: number, k: number): P => [REQ.x + k * REQ.dx, REQ.y[i]!];

// ---------------------------------------------------------------- helpers

const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const ease = (p: number) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);
/** Strict at the start: a step's end state (t == a of the next window) must not show the next window as active. */
const within = (t: number, [a, b]: [number, number]) => t > a && t < b;
const prog = (t: number, [a, b]: [number, number]) => clamp((t - a) / (b - a));

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

function setText(node: Element, s: string): void {
  if (node.textContent !== s) node.textContent = s;
}

type ChipStyle = "plain" | "hot" | "unknown" | "next";
const CHIP_CLASS: Record<ChipStyle, [string, string]> = {
  plain: ["b", "t-chip"],
  hot: ["b-focus", "t-chip t-hot"],
  unknown: ["b-out", "t-chip t-muted"],
  // The token the step being scheduled will produce: highlighted, dashed because it doesn't exist yet.
  next: ["b-next", "t-chip t-hot"],
};

/** A labelled rounded rect positioned by its centre, with an optional small word under it. */
class Chip {
  readonly g: SVGGElement;
  readonly rect: SVGRectElement;
  readonly label: SVGTextElement;
  readonly sub: SVGTextElement;
  constructor(parent: Element) {
    this.g = el("g", {}, parent);
    this.rect = el("rect", { rx: 4 }, this.g);
    this.label = text(this.g, { "text-anchor": "middle" });
    this.sub = text(this.g, { "text-anchor": "middle", class: "t-mono t-muted" });
  }
  hide(): void {
    set(this.g, { display: "none" });
  }
  place([cx, cy]: P, content: string, style: ChipStyle, w = CHIP.w, opacity = 1, sub = "", subHot = false): void {
    set(this.g, { display: opacity <= 0.001 ? "none" : "inline", opacity: clamp(opacity) });
    if (opacity <= 0.001) return;
    const [rc, tc] = CHIP_CLASS[style];
    set(this.rect, { x: cx - w / 2, y: cy - CHIP.h / 2, width: w, height: CHIP.h, class: rc });
    set(this.label, { x: cx, y: cy + 3.5, class: tc });
    setText(this.label, content);
    set(this.sub, { x: cx, y: cy + SUB_DY, display: sub ? "inline" : "none", class: subHot ? "t-mono t-hot" : "t-mono t-muted" });
    setText(this.sub, sub);
  }
  /** A token: the id in the box, the word under it. */
  token(p: P, word: string, style: ChipStyle, w = CHIP.w, opacity = 1): void {
    this.place(p, idOf(word), style, w, opacity, word);
  }
}

// ---------------------------------------------------------------- scene

/** The figure in one mode. */
export const zosScene = (zos: boolean) => (scene: Scene) => build(scene, zos);

function build(scene: Scene, zos: boolean) {
  const { stage, tl } = scene;
  const {
    FIRST, T_END, xs, batchOf, gpuStart, smpEnd, fwd, sampleWin, schedWin, prep, stepSpan, h2d, mdk, d2d, idsWin,
    d2h, mdkFly, sampleFly, postFlight, FINISHED_AT, metaAt, packedAt, countWin, schedFly, VISIBLE, launches, STREAM,
    slotKnown, cellState, STEPS,
  } = timing(zos);

  // Model: the only thing the timeline animates.
  const s = { t: 0 };

  // The static figure in the page stays as the no-JavaScript fallback; the scene draws its own SVG.
  const fallback = stage.querySelector<SVGSVGElement>(":scope > svg");
  const svg = el("svg", {
    class: "diagram",
    viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label":
      (zos ? "With zero-overhead scheduling. " : "Without zero-overhead scheduling: the GPU idles while the CPU works. ") +
      "Animated profile of the scheduling loop. A CPU lane and a GPU lane show each operation over three decode " +
      "steps; below, CPU and GPU memory show token ids moving between them, with the CUDA stream queue in between. " +
      "On the GPU, token ids flow from the sampled token ids into the CUDA graph's input placeholders, through the " +
      "graph to its output, and back through sample.",
  }, stage);
  const defs = el("defs", {}, svg);
  for (const [id, cls] of [["zos-ah", "ah"], ["zos-ah-hot", "ah-hot"], ["zos-ah-gpu", "ah-gpu"]] as const) {
    const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    el("path", { class: cls, d: "M0,0 L10,5 L0,10 z" }, m);
  }
  const clipRect = el("rect", { x: X0, y: 0, width: 0, height: 160 }, el("clipPath", { id: "zos-clip" }, defs));
  el("rect", { x: X0, y: 0, width: X1 - X0, height: 160 }, el("clipPath", { id: "zos-win" }, defs));

  // ---- profile
  for (const id of ["cpu", "gpu"] as const) {
    const ln = LANE[id];
    el("rect", { class: "b-gpu", x: X0, y: ln.y, width: X1 - X0, height: ln.h, rx: 4 }, svg);
    text(svg, { class: "t-title t-gpu", x: 24, y: ln.y + ln.h / 2 + 4 }, ln.label);
  }
  // Step brackets under the GPU lane.
  for (let j = 1; j <= 3; j++) {
    const [s0, s1] = stepSpan(j);
    if (s0 >= T_END) continue;
    const a = xs(s0) + 3;
    const b = xs(s1) - 3;
    const y = 156;
    el("path", { class: "a", d: `M${a},${y - 5} V${y} H${b} V${y - 5}`, "stroke-width": 1 }, svg);
    text(svg, { class: "t-sub", x: (a + b) / 2, y: y + 14, "text-anchor": "middle" }, `step ${STEP[j]}`);
  }

  const launchLines = launches.map(([a, b]) =>
    el("line", { class: "a a-out", x1: xs(a), y1: LANE.cpu.y + LANE.cpu.h, x2: xs(b), y2: LANE.gpu.y, "marker-end": "url(#zos-ah)" }, svg));

  // Queued GPU work: dashed until it runs.
  const ghostLayer = el("g", { "clip-path": "url(#zos-win)" }, svg);
  const ghosts = VISIBLE.filter((sl) => sl.lane === "gpu" && !sl.wait).map((sl) => {
    const ln = LANE.gpu;
    const g = el("g", {}, ghostLayer);
    el("rect", { class: "b-out", x: xs(sl.t0) + 1, y: ln.y + 3, width: xs(sl.t1) - xs(sl.t0) - 2, height: ln.h - 6, rx: 3 }, g);
    const sub = text(g, { class: "t-mono t-muted", x: (xs(sl.t0) + xs(sl.t1)) / 2, y: ln.y + ln.h / 2 + 4, "text-anchor": "middle" }, "queued");
    return { sl, g, sub };
  });

  // Work that has run: revealed by the cursor.
  const sliceLayer = el("g", { "clip-path": "url(#zos-clip)" }, svg);
  const slices = VISIBLE.map((sl) => {
    const ln = LANE[sl.lane];
    const g = el("g", sl.lane === "gpu" ? { class: "hl-gpu" } : {}, sliceLayer);
    const x0 = xs(sl.t0) + 1;
    const x1 = xs(sl.t1) - 1;
    const rect = el("rect", { class: sl.wait ? "b-out" : "b", x: x0, y: ln.y + 3, width: x1 - x0, height: ln.h - 6, rx: 3 }, g);
    const cx = (x0 + x1) / 2;
    const op = text(g, { class: "t-chip", x: cx, y: ln.y + 16, "text-anchor": "middle" }, sl.op);
    const sub = text(g, { class: "t-mono t-muted", x: cx, y: ln.y + 28, "text-anchor": "middle" }, sl.step);
    // Narrow slices (without ZOS the profile spans more time) squeeze their labels to fit.
    for (const n of [op, sub]) {
      const room = x1 - x0 - 4;
      if (n.textContent && n.getComputedTextLength() > room) set(n, { textLength: room, lengthAdjust: "spacingAndGlyphs" });
    }
    return { sl, rect, op };
  });

  const cursor = el("line", { class: "a a-hot", x1: X0, y1: 22, x2: X0, y2: 150 }, svg);

  // ---- CUDA stream: the queue between the CPU and the GPU
  el("rect", { class: "b-ext", x: STRM.x, y: STRM.y, width: STRM.w, height: STRM.h, rx: 6 }, svg);
  text(svg, { class: "t-title", x: GAP_X, y: STRM.y + 16, "text-anchor": "middle" }, "CUDA stream");
  text(svg, { class: "t-mono t-muted", x: GAP_X, y: STRM.y + 31, "text-anchor": "middle" }, "head: GPU runs it →");
  text(svg, { class: "t-mono t-muted", x: GAP_X, y: STRM.y + STRM.h - 8, "text-anchor": "middle" }, "→ tail: CPU appends");
  const streamLayer = el("g", {}, svg);
  const streamItems = STREAM.map((it) => {
    const g = el("g", {}, streamLayer);
    const rect = el("rect", { width: ITEM.w, height: ITEM.h, rx: 3 }, g);
    const op = text(g, { "text-anchor": "middle" }, it.op);
    const sub = text(g, { "text-anchor": "middle", class: "t-mono t-muted" }, it.sub);
    return { it, g, rect, op, sub };
  });

  // ---- lower panel: memory on each side
  for (const [b, name] of [[CPU_BOX, "CPU · host memory"], [GPU_BOX, "GPU · device memory"]] as const) {
    el("rect", { class: "b-gpu", x: b.x, y: b.y, width: b.w, height: b.h, rx: 8 }, svg);
    text(svg, { class: "t-title t-gpu", x: b.x + 14, y: b.y + 22 }, name);
  }
  text(svg, { class: "t-tag", x: 34, y: ROW_Y - 28 }, "PINNED BUFFER");
  text(svg, { class: "t-tag", x: 34, y: REQ.y[0]! - 24 }, "REQUESTS (WHAT THE CPU KNOWS)");
  const schedTag = text(svg, { class: "t-tag", x: 34, y: SCHED_Y - 42 });
  // The last column is filled in prep: remap with ZOS, the input token without.
  for (const [x, label] of [[SCHED_COL.name, "request"], [SCHED_COL.len, "length"], [SCHED_COL.slots, "GPU memory slots"], [SCHED_COL.samp, "sampling"], [SCHED_COL.remap, zos ? "remap" : "input"]] as const) {
    text(svg, { class: "t-mono t-muted", x, y: SCHED_Y - 20, "text-anchor": "middle" }, label);
  }
  // ---- GPU side. Everything here is GPU work, highlighted in the GPU colour (.hl-gpu); CPU work keeps the accent.
  const gsvg = el("g", { class: "hl-gpu" }, svg);
  const mono = (x: number, y: number, cls = "t-mono", anchor = "middle") => text(gsvg, { class: cls, x, y, "text-anchor": anchor });
  // Sampled token ids: what sample wrote, one row per request of its batch. Kept for the next step's gather.
  const smpTag = text(gsvg, { class: "t-tag", x: 566, y: BAND.y + BAND.h + 12 });
  const smpName = GR_Y.map((y) => mono(566, y + 4, "t-mono t-muted", "start"));
  // CUDA graph input placeholders.
  const ginRect = el("rect", { class: "b", x: GIN.x, y: BAND.y, width: GIN.w, height: BAND.h, rx: 6 }, gsvg);
  const ginTag = text(gsvg, { class: "t-tag", x: GIN.x, y: BAND.y - 8 });
  for (const [x, label] of [[GIN.ids, "input_ids"], [GIN.pos, "pos"], [GIN.slots, "slots"]] as const) {
    mono(x, BAND.y + 16, "t-mono t-muted").textContent = label;
  }
  for (const y of GR_Y) el("rect", { class: "b-slot", x: GIN.ids - GCHIP / 2, y: y - CHIP.h / 2, width: GCHIP, height: CHIP.h, rx: 4 }, gsvg);
  const ginPos = GR_Y.map((y) => mono(GIN.pos, y + 4));
  const ginSlots = GR_Y.map((y) => mono(GIN.slots, y + 4));
  const ginName = GR_Y.map((y) => mono(GIN.name, y + 4, "t-mono t-muted"));
  // The CUDA graph: forward's kernels, recorded once and replayed every step on the same buffers.
  const graphRect = el("rect", { class: "b", x: GRAPH.x, y: BAND.y, width: GRAPH.w, height: BAND.h, rx: 6 }, gsvg);
  const graphLbl = ["CUDA graph", "forward", "(replay)"].map((label, k) => {
    const n = mono(GRAPH.x + GRAPH.w / 2, BAND.y + 60 + k * 14, k === 0 ? "t-chip" : "t-mono t-muted");
    n.textContent = label;
    return n;
  });
  // Output placeholder: one hidden state per row, drawn as a short vector.
  el("rect", { class: "b", x: OUT.x, y: BAND.y, width: OUT.w, height: BAND.h, rx: 6 }, gsvg);
  text(gsvg, { class: "t-tag", x: OUT.x, y: BAND.y - 8 }, "OUTPUT");
  mono(OUT.x + OUT.w / 2, BAND.y + 16, "t-mono t-muted").textContent = "hidden state";
  const outCells = GR_Y.map((y) => Array.from({ length: OUT.cells }, (_, c) =>
    el("rect", { x: OUT.x + 10 + c * 9.5, y: y - 7, width: 7, height: 14, rx: 1 }, gsvg)));
  // Batch data from H2D.
  el("rect", { class: "b", x: BATCH_BOX.x, y: LOW.y, width: BATCH_BOX.w, height: LOW.h, rx: 6 }, gsvg);
  const batchTag = text(gsvg, { class: "t-tag", x: BATCH.x, y: BATCH.tagY });
  for (const [x, label] of [[BATCH.req, "req"], [BATCH.len, "length"], [BATCH.slots, "slots"], [BATCH.remap, "remap"]] as const) {
    if (label === "remap" && !zos) continue; // without ZOS there is no remap
    mono(x, BATCH.headY, "t-mono t-muted").textContent = label;
  }
  const gbatch = REQS.map((_, k) => {
    const y = BATCH.rowY + k * BATCH.dy;
    return { req: mono(BATCH.req, y), len: mono(BATCH.len, y), slots: mono(BATCH.slots, y), remap: mono(BATCH.remap, y) };
  });
  // GPU memory slots: what forward stores, per slot, and which request owns it (from the batch data's slots).
  el("rect", { class: "b", x: SLOT_BOX.x, y: LOW.y, width: SLOT_BOX.w, height: LOW.h, rx: 6 }, gsvg);
  text(gsvg, { class: "t-tag", x: GRID.x, y: BATCH.tagY }, "GPU MEMORY SLOTS (4 TOKENS EACH)");
  const grid = Array.from({ length: GRID.n }, (_, sId) => {
    const x = GRID.x + (sId % GRID.perRow) * GRID.dx;
    const y = GRID.y + Math.floor(sId / GRID.perRow) * GRID.dy;
    text(gsvg, { class: "t-mono t-muted", x: x + GRID.w / 2, y: y - 6, "text-anchor": "middle" }, String(sId));
    el("rect", { class: "b-slot", x, y, width: GRID.w, height: GRID.h, rx: 3 }, gsvg);
    const cells = Array.from({ length: SLOT_TOKENS }, (_, c) =>
      el("rect", { x: x + 3, y: y + 3 + c * 5.25, width: GRID.w - 6, height: 4, rx: 1 }, gsvg));
    const owner = text(gsvg, { class: "t-mono", x: x + GRID.w / 2, y: y + GRID.h + 12, "text-anchor": "middle" });
    return { cells, owner };
  });
  for (let i = 0; i < 3; i++) {
    text(gsvg, { class: "t-mono t-muted", x: PIN_X[i]!, y: ROW_Y - 16, "text-anchor": "middle" }, REQS[i]!);
    text(gsvg, { class: "t-mono t-muted", x: 40, y: REQ.y[i]! + 4 }, REQS[i]!);
  }

  // Transfers: D2H (sampled token ids → pinned), H2D (scheduled batch → batch data), D2D (sampled → input_ids).
  const d2hLine = el("line", { class: "a", x1: GPU_BOX.x - 4, y1: ROW_Y, x2: CPU_BOX.x + CPU_BOX.w + 4, y2: ROW_Y }, gsvg);
  const d2hLbl = text(gsvg, { class: "t-mono t-muted", x: GAP_X, y: ROW_Y - 8, "text-anchor": "middle" }, "D2H");
  const h2dLine = el("line", { class: "a", x1: CPU_BOX.x + CPU_BOX.w + 4, y1: SCHED_Y, x2: GPU_BOX.x - 4, y2: SCHED_Y }, gsvg);
  const h2dLbl = text(gsvg, { class: "t-mono t-muted", x: GAP_X, y: SCHED_Y - 8, "text-anchor": "middle" }, "H2D");
  const D2D_X: [number, number] = [SMP_CX + GCHIP / 2 + 2, GIN.ids - GCHIP / 2 - 2];
  const d2dLines = GR_Y.map(() => el("line", { class: "a" }, gsvg));
  const d2dRow = GR_Y.map(() => mono(0, 0, "t-mono t-muted"));
  const d2dLbl = mono((D2D_X[0] + D2D_X[1]) / 2, BAND.y + 16, "t-mono t-muted");
  d2dLbl.textContent = "D2D";
  if (!zos) set(d2dLbl, { display: "none" }); // without ZOS input_ids come by H2D: no gather
  // metadata kernel: batch data → pos and slots placeholders.
  // metadata kernel: no arrow; its pos and slots values fly from the batch data into the graph inputs.
  const mdkLbl = mono(760, BAND.y + BAND.h + 16, "t-mono t-muted", "start");
  mdkLbl.textContent = "metadata kernel";
  // forward: placeholders → graph → output, and the graph reads and writes the GPU memory slots.
  const midY = BAND.y + BAND.h / 2;
  const fwdLines = [
    el("line", { class: "a", x1: GIN.x + GIN.w + 1, y1: midY, x2: GRAPH.x - 2, y2: midY }, gsvg),
    el("line", { class: "a", x1: GRAPH.x + GRAPH.w + 1, y1: midY, x2: OUT.x - 2, y2: midY }, gsvg),
  ];
  const kvX = GRAPH.x + GRAPH.w / 2;
  const kvLine = el("line", { class: "a", x1: kvX, y1: BAND.y + BAND.h + 3, x2: kvX, y2: LOW.y - 3 }, gsvg);
  const kvLbl = mono(kvX + 10, BAND.y + BAND.h + 16, "t-mono t-muted", "start");
  kvLbl.textContent = "read + write KV";
  // sample: output → sampled token ids, looping over the top.
  const LOOP_Y = 254;
  const smpLine = el("path", { class: "a", d: `M${OUT.x + OUT.w / 2},${BAND.y - 2} V${LOOP_Y} H${SMP_CX} V${GR_Y[0]! - CHIP.h / 2 - 3}` }, gsvg);
  const smpLbl = mono((SMP_CX + OUT.x) / 2, LOOP_Y - 5, "t-mono t-muted");
  smpLbl.textContent = "sample: hidden state → scores → token id";


  const finishedLbl = text(svg, { class: "t-mono t-hot", x: 0, y: 0 }, "finished");

  // Chips. Static ones are re-labelled every frame; flying ones carry a token id between places.
  const chipLayer = el("g", {}, svg);
  const pin = REQS.map(() => new Chip(chipLayer));
  const gpuChipLayer = el("g", { class: "hl-gpu" }, svg);
  const smp = GR_Y.map(() => new Chip(gpuChipLayer));
  const inp = GR_Y.map(() => new Chip(gpuChipLayer));
  const reqChips = REQS.map(() => Array.from({ length: 6 }, () => new Chip(chipLayer)));
  // Scheduled batch: one row per request: its name, length, GPU memory slots and sampling settings.
  const sched = REQS.map((_, i) => {
    const y = SCHED_Y + i * SCHED_DY + 4;
    const cell = (x: number) => text(svg, { class: "t-mono", x, y, "text-anchor": "middle" });
    // Without ZOS the last column is the input token id, with its word under it only for reading.
    const inWord = text(svg, { class: "t-mono t-muted", x: SCHED_COL.remap, y: y + 11, "text-anchor": "middle", "font-size": 8 });
    return { name: new Chip(chipLayer), len: cell(SCHED_COL.len), slots: cell(SCHED_COL.slots), samp: cell(SCHED_COL.samp), remap: cell(SCHED_COL.remap), inWord };
  });

  const flyLayer = el("g", {}, svg);
  const gpuFlyLayer = el("g", { class: "hl-gpu" }, svg);
  const flyD2H = REQS.map(() => new Chip(gpuFlyLayer));
  const flyD2D = REQS.map(() => new Chip(gpuFlyLayer)); // input token ids: D2D with ZOS, H2D without
  const flyPost = REQS.map(() => new Chip(flyLayer));
  const flyH2D = new Chip(gpuFlyLayer);
  const flySmp = GR_Y.map(() => new Chip(gpuFlyLayer));
  const flyMeta = GR_Y.map(() => [new Chip(gpuFlyLayer), new Chip(gpuFlyLayer)] as const);
  const flySched = REQS.map(() => new Chip(flyLayer));
  const flyCount = REQS.map(() => new Chip(flyLayer));

  // What the current step does, in words, under the diagram.
  const desc = document.createElement("div");
  desc.className = "viz-desc";
  desc.setAttribute("aria-live", "polite");
  const title = desc.appendChild(document.createElement("strong"));
  const detail = desc.appendChild(document.createElement("span"));
  stage.append(desc);

  // ---- render

  const fly = (c: Chip, from: P, to: P, u: number, content: string, w = CHIP.w) => {
    const e = ease(u);
    c.place([lerp(from[0], to[0], e), lerp(from[1], to[1], e)], content, "hot", w);
  };
  /** Latest step j in [lo, hi] whose event time f(j) has passed. */
  const latest = (t: number, lo: number, hi: number, f: (j: number) => number) => {
    let r = lo;
    for (let j = lo; j <= hi; j++) if (t >= f(j)) r = j;
    return r;
  };

  scene.onRender(() => {
    const t = s.t;
    const step = STEPS[Math.max(0, scene.stepIndex)]!;
    setText(title, step[1]);
    setText(detail, step[2]);

    // Profile
    const cx = xs(t);
    set(clipRect, { width: Math.max(0, cx - X0) });
    set(cursor, { x1: cx, x2: cx, opacity: t >= T_END ? 0 : 1 });
    for (const { sl, rect, op } of slices) {
      const hot = t > sl.t0 && t < sl.t1 && !sl.wait;
      set(rect, { class: sl.wait ? "b-out" : hot ? "b-focus" : "b" });
      set(op, { class: hot ? "t-chip t-hot" : sl.wait ? "t-chip t-muted" : "t-chip" });
    }
    for (const { sl, g, sub } of ghosts) {
      set(g, { display: t >= sl.launch! && t < sl.t1 ? "inline" : "none" });
      set(sub, { display: t < sl.t0 ? "inline" : "none" });
    }
    launches.forEach(([a], k) => set(launchLines[k]!, { opacity: t >= a ? 1 : 0 }));

    // CUDA stream: items in order from the head; each moves up as the ones ahead of it leave.
    // New items slide in from the CPU side (left); finished ones slide out towards the GPU side (right).
    let y = ITEM.top;
    for (const { it, g, rect, op, sub } of streamItems) {
      const lw = leaveWin(it);
      if (it.fly && within(t, it.fly)) {
        // Launched from the batch: flies (above the memory boxes) to its place at the tail.
        const u = ease(prog(t, it.fly));
        const ix = lerp(BATCH_OUT[0], GAP_X, u) - ITEM.w / 2;
        const iy = lerp(BATCH_OUT[1] - ITEM.h / 2, y, u);
        if (g.parentNode !== flyLayer) flyLayer.append(g);
        set(g, { display: "inline", opacity: 1, class: "" });
        set(rect, { x: ix, y: iy, class: "b-focus" });
        set(op, { x: ix + ITEM.w / 2, y: iy + 14, class: "t-chip t-hot" });
        set(sub, { x: ix + ITEM.w / 2, y: iy + 26 });
        y += ITEM.h + ITEM.gap;
        continue;
      }
      if (g.parentNode !== streamLayer) streamLayer.append(g);
      set(g, { class: "hl-gpu" });
      if (t < it.e || t >= lw[1]) {
        set(g, { display: "none" });
        continue;
      }
      const enter = it.fly ? 1 : clamp((t - it.e) / ENTER);
      const leave = prog(t, lw);
      const running = !it.event && t > it.s && t < it.f;
      const reached = it.event && t >= it.f;
      set(g, { display: "inline", opacity: Math.min(enter, 1 - leave) });
      const ix = GAP_X - ITEM.w / 2 - (1 - enter) * 24 + leave * 24;
      set(rect, { x: ix, y, class: running || reached ? "b-focus" : it.event ? "b" : "b-out" });
      set(op, { x: ix + ITEM.w / 2, y: y + 14, class: running || reached ? "t-chip t-hot" : "t-chip" });
      set(sub, { x: ix + ITEM.w / 2, y: y + 26 });
      y += (ITEM.h + ITEM.gap) * (1 - leave);
    }

    // Transfers
    // GPU step whose inputs ran last (step N until step N+1's inputs start).
    let jNext = 1;
    for (let j = 2; j <= 3; j++) if (t > gpuStart(j)) jNext = j;
    const d2hOn = [1, 2].some((j) => within(t, d2h(j)));
    const h2dOn = [1, 2, 3].some((j) => within(t, h2d(j)));
    const mdkOn = [1, 2, 3].some((j) => within(t, mdk(j)));
    const d2dOn = [1, 2, 3].some((j) => within(t, d2d(j)));
    const fwdOn = [1, 2, 3].some((j) => within(t, fwd(j)));
    const sampling = [1, 2].some((j) => within(t, sampleWin(j)));
    const lineHot = (node: Element, on: boolean, both = false) => {
      const m = on ? "url(#zos-ah-gpu)" : "url(#zos-ah)"; // every animated arrow is a GPU copy or kernel
      set(node, { class: on ? "a a-hot" : "a", "marker-end": m, ...(both ? { "marker-start": m } : {}) });
    };
    const textHot = (node: Element, on: boolean) => set(node, { class: on ? "t-mono t-hot" : "t-mono t-muted" });
    for (const [line, lbl, on] of [[d2hLine, d2hLbl, d2hOn], [h2dLine, h2dLbl, h2dOn], [smpLine, smpLbl, sampling]] as const) {
      lineHot(line, on);
      textHot(lbl, on);
    }
    textHot(mdkLbl, mdkOn);
    for (const line of fwdLines) lineHot(line, fwdOn);
    lineHot(kvLine, fwdOn, true);
    textHot(kvLbl, fwdOn);
    set(graphRect, { class: fwdOn ? "b-focus" : "b" });
    set(ginRect, { class: fwdOn ? "b-focus" : "b" }); // forward reads the graph inputs
    set(graphLbl[0]!, { class: fwdOn ? "t-chip t-hot" : "t-chip" });
    // D2D: input row k (request i in the running step's batch) reads the sampled row remap gives: request i's row
    // in the previous step's batch.
    d2dLines.forEach((line, k) => {
      const i = batchOf(jNext)[k];
      const used = zos && i !== undefined;
      const src = used ? batchOf(jNext - 1).indexOf(i) : k;
      lineHot(line, d2dOn);
      set(line, { display: used ? "inline" : "none", x1: D2D_X[0], y1: GR_Y[src]!, x2: D2D_X[1], y2: GR_Y[k]! });
      textHot(d2dRow[k]!, d2dOn);
      set(d2dRow[k]!, { display: used ? "inline" : "none", x: (D2D_X[0] + D2D_X[1]) / 2, y: Math.min(GR_Y[src]!, GR_Y[k]!) - 5 });
      setText(d2dRow[k]!, `row ${src}`);
    });
    textHot(d2dLbl, d2dOn);

    // GPU memory slots: owner from the metadata; cells fill as forward stores each token's results.
    grid.forEach(({ cells, owner }, sId) => {
      let who = -1;
      let k = -1;
      SLOT_IDS.forEach((ids, i) => {
        const at = ids.indexOf(sId);
        if (at >= 0 && slotKnown(i, at, t)) [who, k] = [i, at];
      });
      setText(owner, who >= 0 ? REQS[who]! : "");
      let writing = false;
      cells.forEach((cell, c) => {
        const st = who >= 0 ? cellState(who, k * SLOT_TOKENS + c, t) : "empty";
        writing ||= st === "hot";
        set(cell, { class: st === "full" ? "c-full" : st === "hot" ? "c-hot" : "c-empty" });
      });
      set(owner, { class: writing ? "t-mono t-hot" : "t-mono" });
    });

    // GPU: sampled token ids (step js), one row per request of its batch, shown once sample(js) has produced them.
    // Each row switches to sample(j)'s token when that row's token id lands; it flies from the output's row k, up,
    // along the sample arrow and down into row k.
    const along = (pts: P[], u: number): P => {
      const segs = pts.slice(1).map((q, n) => Math.hypot(q[0] - pts[n]![0], q[1] - pts[n]![1]));
      let d = ease(u) * segs.reduce((a, b) => a + b, 0);
      for (let n = 0; n < segs.length; n++) {
        if (d <= segs[n]! || n === segs.length - 1) {
          const v = segs[n]! ? Math.min(1, d / segs[n]!) : 1;
          return [lerp(pts[n]![0], pts[n + 1]![0], v), lerp(pts[n]![1], pts[n + 1]![1], v)];
        }
        d -= segs[n]!;
      }
      return pts[pts.length - 1]!;
    };
    const outCx = OUT.x + OUT.w / 2;
    setText(smpTag, `SAMPLED TOKEN IDS · STEP ${STEP[latest(t, 0, 2, (j) => sampleFly(j, 0)[1])]}`);
    smp.forEach((c, k) => {
      const js = latest(t, 0, 2, (j) => sampleFly(j, k)[1]);
      const i = batchOf(js)[k];
      const jf = [1, 2].find((j) => within(t, sampleFly(j, k)));
      const fi = jf === undefined ? undefined : batchOf(jf)[k];
      if (jf !== undefined && fi !== undefined) {
        const pts: P[] = [[outCx, GR_Y[k]!], [outCx, LOOP_Y], [SMP_CX, LOOP_Y], smpAt(k)];
        flySmp[k]!.place(along(pts, prog(t, sampleFly(jf, k))), idOf(SAMPLED[jf]![fi]!), "hot", GCHIP);
      } else flySmp[k]!.hide();
      set(smpName[k]!, { display: i === undefined ? "none" : "inline" });
      if (i === undefined) return c.hide();
      setText(smpName[k]!, REQS[i]!);
      const landed = [1, 2].some((j) => within(t, [sampleFly(j, k)[1], smpEnd(j)]));
      c.token(smpAt(k), SAMPLED[js]![i]!, landed ? "hot" : "plain", GCHIP);
    });

    // GPU: CUDA graph input placeholders for the step whose inputs ran last. The buffers are fixed: a field keeps the
    // previous step's value until this step's copy overwrites it (pos and slots by the metadata kernel, input_ids by
    // the gather). A row the new batch doesn't use is shown empty.
    setText(ginTag, `CUDA GRAPH INPUTS · STEP ${STEP[jNext]}`);
    const jIds = t >= idsWin(jNext)[1] ? jNext : jNext - 1;
    const mHot = within(t, mdk(jNext));
    inp.forEach((c, k) => {
      const i = batchOf(jNext)[k];
      const iw = idsWin(jNext);
      if (i !== undefined && within(t, iw)) {
        // With ZOS the gather reads the sampled row remap names; without, H2D brings it from the scheduled batch.
        const from: P = zos ? smpAt(batchOf(jNext - 1).indexOf(i)) : [SCHED_COL.remap, SCHED_Y + k * SCHED_DY];
        fly(flyD2D[k]!, from, inAt(k), prog(t, iw), idOf(inputWord(jNext, i)), GCHIP);
      } else flyD2D[k]!.hide();
      const iv = jIds >= 1 ? batchOf(jIds)[k] : undefined;
      if (iv === undefined || within(t, iw)) c.hide();
      else c.token(inAt(k), inputWord(jIds, iv), fwdOn ? "hot" : "plain", GCHIP);
      // pos and slots: row k switches to this step's values when the metadata kernel's copy lands.
      const mf = mdkFly(jNext, k);
      const nk = batchOf(jNext)[k];
      if (nk !== undefined && within(t, mf)) {
        const nlen = PROMPT[nk]!.length + jNext;
        const from = BATCH.rowY + k * BATCH.dy - 4;
        const u = prog(t, mf);
        fly(flyMeta[k]![0], [BATCH.len, from], [GIN.pos, GR_Y[k]!], u, String(nlen - 1), 24);
        fly(flyMeta[k]![1], [BATCH.slots, from], [GIN.slots, GR_Y[k]!], u, slotsFor(nk, nlen), 40);
      } else flyMeta[k]!.forEach((f) => f.hide());
      const jMeta = t >= mf[1] ? jNext : jNext - 1;
      const im = jMeta >= 1 ? batchOf(jMeta)[k] : undefined;
      const len = im === undefined ? 0 : PROMPT[im]!.length + jMeta;
      for (const [node, val] of [[ginPos[k]!, String(len - 1)], [ginSlots[k]!, im === undefined ? "" : slotsFor(im, len)]] as const) {
        set(node, { display: im === undefined || within(t, mf) ? "none" : "inline", class: (mHot && jMeta === jNext) || fwdOn ? "t-mono t-hot" : "t-mono" });
        setText(node, val);
      }
      set(ginName[k]!, { display: im === undefined ? "none" : "inline" });
      setText(ginName[k]!, im === undefined ? "" : REQS[im]!);
    });

    // GPU: output placeholder, written by the last forward that started (filling while it runs).
    const jo = latest(t, 0, 3, (j) => fwd(j)[0]);
    const fo = jo >= 1 ? prog(t, fwd(jo)) : 1;
    outCells.forEach((cells, k) => {
      const on = batchOf(jo)[k] !== undefined;
      const read = [1, 2].some((j) => within(t, sampleFly(j, k))); // sample is reading this row's hidden state
      cells.forEach((cell, c) => {
        const filled = fo >= 1 || (c + 1) / OUT.cells <= fo;
        set(cell, { display: on ? "inline" : "none", class: !filled ? "b-slot" : fo < 1 || read ? "c-hot" : "c-full" });
      });
    });

    // CPU: pinned buffer holds the last D2H; D2H flights carry sample(j)'s token ids across.
    const jp = latest(t, 0, 2, smpEnd);
    const jd = [1, 2].find((j) => within(t, d2h(j)));
    pin.forEach((c, i) => {
      if (jd !== undefined && batchOf(jd).includes(i)) fly(flyD2H[i]!, smpAt(batchOf(jd).indexOf(i)), pinAt(i), prog(t, d2h(jd)), idOf(SAMPLED[jd]![i]!));
      else flyD2H[i]!.hide();
      if (!batchOf(jp).includes(i)) return c.hide(); // not in the step: nothing copied for it
      const dropped = jp > FINISH.step && i === FINISH.req; // with ZOS: r2's token from N+1, which nobody uses
      c.token(pinAt(i), SAMPLED[jp]![i]!, "plain", CHIP.w, dropped ? lerp(1, 0.35, prog(t, postFlight(jp, i))) : 1);
    });

    // CPU: requests. post-process(j) moves each token from the pinned buffer to the end of its request.
    // Simplification: the requests show text; the CPU actually stores token ids there too.
    const finished = t >= FINISHED_AT;
    // schedule(j) runs while t is inside schedWin(j); jsch is the step the batch is for.
    let jsch = FIRST - 1;
    for (let j = FIRST; j <= 3; j++) if (t > schedWin(j)[0]) jsch = j;
    const scheduling = jsch >= FIRST && within(t, schedWin(jsch));
    // prep(j) builds step j's metadata from its batch, one request at a time; jPrep is the latest one started.
    let jPrep = FIRST - 1;
    for (let j = FIRST; j <= 3; j++) if (t > prep(j)[0]) jPrep = j;
    const building = jPrep >= FIRST && within(t, prep(jPrep));
    const writing = (r: number) => building && t > metaAt(jPrep, r) - 0.45 && t <= metaAt(jPrep, r);
    REQS.forEach((_, i) => {
      const toks = [...PROMPT[i]!];
      let flying: { j: number; u: number } | undefined;
      for (let j = 0; j <= 2; j++) {
        if (j > FINISH.step && i === FINISH.req) continue; // dropped: r2 already finished
        const w = postFlight(j, i);
        if (t >= w[1]) toks.push(SAMPLED[j]![i]!);
        else if (t >= w[0]) flying = { j, u: prog(t, w) };
      }
      const row = reqChips[i]!;
      const done = i === FINISH.req && finished;
      // Dashed boxes after the known tokens: every token already scheduled to be produced but not on the CPU yet,
      // labelled with the step that produces it. schedule(j) adds token j's box (highlighted while scheduling: step j
      // is scheduled to produce it; not counted: step j doesn't read it). The box stays until post-process brings the
      // token in. A count runs along the row up to the input token j−1: with ZOS that token isn't on the CPU yet (its
      // dashed box is counted); without ZOS it is the last token the CPU has.
      const counting = scheduling && batchOf(jsch).includes(i) && !done && !flying;
      const firstPending = toks.length - PROMPT[i]!.length; // step that produces the first token not on the CPU
      const reads = toks.length + (zos ? 1 : 0);
      const fw = jsch >= FIRST ? schedFly(jsch, i) : ([Infinity, Infinity] as [number, number]);
      const upTo = counting && t < fw[0] ? prog(t, countWin(jsch)) * reads : 0;
      const num = (k: number) => (k < upTo ? String(k + 1) : "");
      row.forEach((c, k) => {
        const at = reqAt(i, k);
        if (k < toks.length) return c.place(at, toks[k]!, k < upTo ? "hot" : "plain", REQ.w, 1, num(k), true);
        const j = firstPending + (k - toks.length);
        if (done || j > jsch || (j === firstPending && flying)) return c.hide();
        c.place(at, STEP[j]!, counting && j === jsch ? "next" : "unknown", REQ.w, 1, num(k), true);
      });
      // The count moves into the batch as the request's length.
      const [nx, ny] = reqAt(i, reads - 1);
      if (counting && within(t, fw)) fly(flyCount[i]!, [nx, ny + SUB_DY - 4], [SCHED_COL.len, SCHED_Y + batchOf(jsch).indexOf(i) * SCHED_DY], prog(t, fw), String(reads), 28);
      else flyCount[i]!.hide();
      if (flying) fly(flyPost[i]!, pinAt(i), reqAt(i, toks.length), flying.u, idOf(SAMPLED[flying.j]![i]!), REQ.w);
      else flyPost[i]!.hide();
      if (i === FINISH.req) {
        const [fx, fy] = reqAt(i, toks.length);
        set(finishedLbl, { x: fx - REQ.w / 2, y: fy + 4, display: done ? "inline" : "none" });
      }
    });

    // CPU: the batch most recently scheduled, one row per request in row order (a finished request's row is gone, so
    // the ones after it move up). schedule(j) clears the old batch, then each chosen request flies in with its count.
    setText(schedTag, `SCHEDULED BATCH · STEP ${STEP[jsch]}`);
    // After the last remap entry the whole batch lights up, until its blocks have flown into the stream.
    const packed = building && jPrep === jsch && t > packedAt(jsch);
    sched.forEach(({ name, len, slots, samp, remap, inWord }, i) => {
      const on = batchOf(jsch).includes(i);
      const rowY = SCHED_Y + Math.max(0, batchOf(jsch).indexOf(i)) * SCHED_DY;
      for (const cell of [len, slots, samp, remap]) set(cell, { y: rowY + 4 });
      set(inWord, { y: rowY + 15 });
      const fw: [number, number] = jsch >= FIRST ? schedFly(jsch, i) : [-Infinity, -Infinity];
      const landed = t >= fw[1];
      if (on && within(t, fw)) fly(flySched[i]!, [52, REQ.y[i]!], [SCHED_COL.name, rowY], prog(t, fw), REQS[i]!, 36);
      else flySched[i]!.hide();
      if (on && landed) name.place([SCHED_COL.name, rowY], REQS[i]!, scheduling || packed || writing(batchOf(jsch).indexOf(i)) ? "hot" : "plain", 36);
      else name.hide();
      const shown = !on || !landed ? 0 : jsch < FIRST ? 1 : prog(t, [fw[1], fw[1] + 0.15]); // scheduled before t = 0
      const cls = scheduling || packed ? "t-mono t-hot" : "t-mono";
      setText(len, String(PROMPT[i]!.length + jsch));
      setText(slots, slotsFor(i, PROMPT[i]!.length + jsch));
      setText(samp, SAMPLING[i]!);
      set(len, { display: on && landed ? "inline" : "none", class: cls });
      for (const cell of [slots, samp]) set(cell, { display: shown > 0 ? "inline" : "none", opacity: shown, class: cls });
      // Added in prep. With ZOS: remap, the request's row in the previous step's batch. Without: the input token,
      // which the CPU already has. A batch scheduled before t = 0 already has it.
      const r = batchOf(jsch).indexOf(i);
      const hasRemap = on && (jsch < FIRST || t > metaAt(jsch, r));
      setText(remap, zos ? String(batchOf(jsch - 1).indexOf(i)) : idOf(inputWord(jsch, i)));
      setText(inWord, zos ? "" : inputWord(jsch, i));
      set(inWord, { display: !zos && hasRemap ? "inline" : "none" });
      set(remap, { display: hasRemap ? "inline" : "none", class: writing(r) || (building && jPrep === jsch) ? "t-mono t-hot" : "t-mono" });
    });

    // GPU: batch data of the step most recently copied in; H2D carries it across from the scheduled batch.
    const jm = latest(t, 1, 3, (j) => h2d(j)[1]);
    const hasBatch = t >= h2d(1)[1];
    const fresh = t - h2d(jm)[1] < 0.8 || mHot;
    setText(batchTag, `BATCH DATA (H2D) · STEP ${STEP[jm]}`);
    gbatch.forEach((row, k) => {
      const i = batchOf(jm)[k];
      const len = i === undefined ? 0 : PROMPT[i]!.length + jm;
      const vals = i === undefined ? ["", "", "", ""] : [REQS[i]!, String(len), slotsFor(i, len), String(batchOf(jm - 1).indexOf(i))];
      [row.req, row.len, row.slots, row.remap].forEach((node, c) => {
        set(node, { display: hasBatch && i !== undefined && (zos || c < 3) ? "inline" : "none", class: fresh || (c === 3 && d2dOn) ? "t-mono t-hot" : "t-mono" });
        setText(node, vals[c]!);
      });
    });
    const jh = [1, 2, 3].find((j) => within(t, h2d(j)));
    if (jh !== undefined) fly(flyH2D, META_FROM, META_TO, prog(t, h2d(jh)), "batch data", 72);
    else flyH2D.hide();
  });

  // ---------------------------------------------------------------- timeline

  // Steps run back to back: no pause between them or at the end.
  STEPS.forEach(([name, , , t0, t1], k) => {
    tl.addLabel(name, k === 0 ? 0 : "+=0");
    tl.to(s, { t: t1, duration: (t1 - t0) * U, ease: "none" }, name);
  });

  if (fallback) fallback.style.display = "none";
  return () => {
    svg.remove();
    desc.remove();
    if (fallback) fallback.style.display = "";
  };
}

/** Mount the figure with a mode switch above it. Switching rebuilds the scene in the other mode and plays it from the
 *  start. Without JavaScript the page shows the static figure (with ZOS) and no switch. */
export function mountZos(target: string): void {
  const figure = document.querySelector<HTMLElement>(target);
  if (!figure) return;
  let scene = mount(figure, zosScene(true));
  const bar = document.createElement("div");
  bar.className = "viz-mode";
  bar.setAttribute("role", "group");
  bar.setAttribute("aria-label", "Scheduling mode");
  const buttons = ([["With ZOS", true], ["Without ZOS", false]] as const).map(([label, zos]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.setAttribute("aria-pressed", String(zos));
    b.addEventListener("click", () => {
      if (b.getAttribute("aria-pressed") === "true") return;
      scene?.destroy();
      scene = mount(figure, zosScene(zos));
      for (const o of buttons) o.setAttribute("aria-pressed", String(o === b));
    });
    return b;
  });
  bar.append(...buttons);
  figure.prepend(bar);
}
