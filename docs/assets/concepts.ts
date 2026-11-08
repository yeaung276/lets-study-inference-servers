// Page script for pages/concepts.html.
import { mount } from "../viz";
import ranks from "../viz/scenes/ranks";
import processGroups from "../viz/scenes/process-groups";
import moeAllToAll from "../viz/scenes/moe-all-to-all";
import { mountZos } from "../viz/scenes/zos";
import { mountToc } from "./toc";

mount("#ranks-viz", ranks);
mount("#groups-viz", processGroups);
mount("#a2a-viz", moeAllToAll);
mountZos("#zos-viz");
mountToc();
