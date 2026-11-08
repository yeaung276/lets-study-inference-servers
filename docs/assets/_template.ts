// Page script for pages/_template.html. Copy to assets/<page>.ts alongside the page.
import { mount } from "../viz";
import exampleKvFill from "../viz/scenes/example-kv-fill";

mount("#example-viz", exampleKvFill);
