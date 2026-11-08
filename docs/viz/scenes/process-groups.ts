// Process groups figure on pages/concepts.html. Same nodes, GPUs and processes as the Ranks figure.
// Step 1: TP groups (one per node). Step 2: DP groups (same position on each node). Step 3: one EP group.
// Hovering (or focusing) a process shows its global rank and local rank, as in the Ranks figure.

import type { Scene } from "..";
import { rankHover } from "./ranks";

export default function processGroups(scene: Scene) {
  const { stage, tl } = scene;

  tl.addLabel("TP groups")
    .to(".tp", { opacity: 1, duration: 0.5 }, "TP groups")
    .addLabel("DP groups", "+=1.5")
    .to(".tp", { opacity: 0, duration: 0.3 }, "DP groups")
    .to(".dp", { opacity: 1, duration: 0.5 }, "DP groups+=0.2")
    .addLabel("EP group", "+=1.5")
    .to(".dp", { opacity: 0, duration: 0.3 }, "EP group")
    .from(".ep", { opacity: 0, duration: 0.5 }, "EP group+=0.2");

  return rankHover(stage);
}
