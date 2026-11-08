# Docs: how to add a page

Run the site from the repo root with `bun run docs`, then open http://localhost:3000.

## Layout

```
docs/
├── server.ts            Bun server (do not edit to add pages)
├── index.html           home page, always first in the sidebar
├── pages/
│   ├── _template.html   copy this to start a new page
│   └── <name>.html      served at /<name>
├── assets/
│   ├── docs.css         shared styles
│   ├── docs.js          injects the left sidebar
│   └── <name>.ts        page script for pages/<name>.html, bundled by the server on each request
└── viz/                 animation runtime; imported by page scripts, never served directly
    ├── scene.ts         mount(): master timeline, player bar, autoplay, reduced motion
    ├── canvas.ts        HiDPI canvas layer that draws in the SVG's viewBox units
    ├── theme.ts         theme tokens for canvas code
    └── scenes/          one file per animated figure
```

## Add a page

1. Copy `pages/_template.html` to `pages/<name>.html`. Use a lowercase, hyphenated file name, for example `kv-cache.html`. The page is served at `/<name>`.
2. Set `<title>`. It becomes the page's label in the sidebar.
3. Set `<meta name="nav-order" content="N">`. Pages are sorted by this number, lowest first.
4. Write the content as `<section id="...">` elements that are **direct children of `<main>`**, each starting with an `<h2>`. Each section becomes a sub-item under the page in the sidebar, labelled with its `<h2>` text.

```html
<section id="paged-attention">
  <h2>Paged attention</h2>
  <p>...</p>
</section>
```

## Rules

- The sidebar is built automatically, so never edit it by hand and never register pages anywhere.
- Every page must link `/assets/docs.css` and `/assets/docs.js` in `<head>`, as the template does.
- Section `id`s must be unique within a page.
- Headings inside a section (`<h3>` and below) do not appear in the sidebar.
- Files in `pages/` that start with `_` are hidden from the sidebar.
- Put shared CSS and JS in `assets/`. For a page-specific script, add `assets/<name>.ts` and include it in that page only with `<script type="module" src="/assets/<name>.ts"></script>`.
- Run `bun run typecheck` after changing any `.ts` file. Build errors also print in the server terminal and the browser console.

## Animated figures

The stack:

- **GSAP** handles timing. Each figure has one paused master timeline.
- **d3** is for math only: scales, layouts and interpolators. Don't use it to edit the DOM.
- **The renderer** depends on how many marks are drawn:
  - SVG for up to about 2k marks.
  - A canvas layer under the SVG for up to about 50k.
  - PixiJS (2D) or Three.js (3D) beyond that. Neither is installed yet; `bun add` one when a scene needs it.
- **Packages come from npm,** never from a CDN.

Working example: the "Animated figures" section of `pages/_template.html`, with `viz/scenes/example-kv-fill.ts`.

### Add one

1. **Write the figure in the page.** Put the SVG inside `<div class="viz-stage">` in a `<figure id="...">`. Draw the final state, so the figure still makes sense without JavaScript, and reuse the `.diagram` classes.
2. **Write the scene** in `viz/scenes/<scene>.ts`. Its default export is `(scene: Scene) => void`.
3. **Mount it** in `assets/<page>.ts` with `mount("#figure-id", scene, options)`. Options:
   - `autoplay` (default on)
   - `loop`
   - `controls` (default on)
   - `aspect`, for canvas-only stages
   - `defaults`, for tween defaults
4. **Load the script** in the page's `<head>` with `<script type="module" src="/assets/<page>.ts"></script>`.

### Scene rules

- **Separate model and view.** The timeline tweens a plain state object, and `scene.onRender(() => ...)` draws from that state. SVG-only scenes may tween SVG elements directly.
- **Labels are steps.** The player's previous and next buttons move between them. Anchor each step's tweens at its label: `tl.addLabel("decode", "+=0.5").to(state, {...}, "decode")`. A label on its own does not move the next tween, so the gap disappears.
- **Canvas uses the SVG's coordinates.** `canvasLayer(scene, { viewBox: [w, h] })` draws in viewBox units, so canvas marks line up with SVG labels.
- **Colors come from the theme.** Canvas colors come from `scene.theme` and are never hard-coded. SVG colors come from the CSS classes.
- **Selectors stay inside the figure.** Selector strings in tweens only match inside the stage.
- **Leave `scene.tl`'s callbacks alone.** Don't set `onUpdate` or `onComplete` on it; use `tl.call()` or `onRender`.
- **Heavy simulations go in a Worker.** Run them there and feed the results into the state object.
- **These are handled for you:**
  - autoplay once the figure scrolls into view, and pausing while it is off screen
  - `prefers-reduced-motion`: the figure shows its final state and steps jump instead of animating
  - HiDPI canvas, resizing, and the theme toggle

## Constraints (learned from past mistakes)

**Scope**
- Change only what was asked. Never remove, rename, shorten or "improve" anything else on the way, including terms, sections and other figures.
- If a request is ambiguous, ask one short question. Don't guess and edit.
- "Revert" means put back exactly what was there. No tweaks.

**Writing**
- The reader has only the page. Each concept starts with why it exists, then what it is, then the example in the figure.
- Don't cut context to make text shorter, and don't pile up bare definitions.

**Figures**
- Don't repeat the same long label in every box. Show it once (header, legend or hover).
- Keep figures centred and balanced, with no empty gutter on one side.
- Hardware and software get different colours. Use only the site palette: GPUs (hardware) are neutral (`--muted` / `--bg-soft`), processes (software) are accent (`--accent` / `--note-bg`). No new hues such as green, purple or pink.
- Hover and focus must never cover, hide or move other content. Show the extra info inside the hovered element itself, not in a floating tooltip.
- Interactive and animated figures use the viz runtime (see "Animated figures" above).
