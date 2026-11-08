// Docs server: `bun run docs` → http://localhost:3000
//   /             → index.html
//   /<name>       → pages/<name>.html
//   /assets/*.ts  → assets/*.ts bundled for the browser (imports from viz/ and node_modules resolved)
//   /assets/*     → assets/*
//   /nav.json     → list of pages and their sections (used by assets/docs.js)
import path from "node:path";

const ROOT = import.meta.dir;
const PORT = Number(process.env.PORT ?? 3000);

type Page = { url: string; title: string; order: number; sections: { id: string; title: string }[] };

async function readPage(file: string, url: string): Promise<Page> {
  const page: Page = { url, title: "", order: 999, sections: [] };
  let inH2 = false;
  await new HTMLRewriter()
    .on("title", { text: (t) => void (page.title += t.text) })
    .on('meta[name="nav-order"]', { element: (el) => void (page.order = Number(el.getAttribute("content"))) })
    .on("main > section[id]", { element: (el) => void page.sections.push({ id: el.getAttribute("id")!, title: "" }) })
    .on("main > section[id] > h2", {
      element: (el) => {
        inH2 = true;
        el.onEndTag(() => void (inH2 = false));
      },
      text: (t) => {
        if (inH2) page.sections.at(-1)!.title += t.text;
      },
    })
    .transform(new Response(Bun.file(file)))
    .text();
  page.title = page.title.trim();
  page.sections.forEach((s) => (s.title = s.title.trim()));
  return page;
}

async function nav() {
  const pages = [await readPage(path.join(ROOT, "index.html"), "/")];
  for await (const f of new Bun.Glob("*.html").scan(path.join(ROOT, "pages"))) {
    if (f.startsWith("_")) continue; // _template.html etc. are hidden
    pages.push(await readPage(path.join(ROOT, "pages", f), "/" + f.replace(/\.html$/, "")));
  }
  const [home, ...rest] = pages;
  return [home, ...rest.sort((a, b) => a.order - b.order)];
}

// Page scripts are TypeScript entry points. They are bundled fresh on every request,
// so edits show up on reload without restarting the server.
async function bundle(entry: string): Promise<Response> {
  const headers = { "Content-Type": "text/javascript; charset=utf-8", "Cache-Control": "no-store" };
  try {
    const out = await Bun.build({ entrypoints: [entry], target: "browser", format: "esm", sourcemap: "inline" });
    if (!out.success) throw new AggregateError(out.logs, "Build failed");
    return new Response(out.outputs[0], { headers });
  } catch (err) {
    const logs = err instanceof AggregateError ? err.errors : [err];
    const message = `[docs] ${path.relative(ROOT, entry)} failed to build:\n${logs.map(String).join("\n")}`;
    console.error(message);
    // Served as JS so the error shows up in the browser console too.
    return new Response(`console.error(${JSON.stringify(message)});`, { headers });
  }
}

Bun.serve({
  port: PORT,
  async fetch(req) {
    const { pathname } = new URL(req.url);
    if (pathname === "/nav.json") return Response.json(await nav());

    const file =
      pathname === "/"
        ? path.join(ROOT, "index.html")
        : pathname.startsWith("/assets/")
          ? path.join(ROOT, pathname)
          : path.join(ROOT, "pages", pathname + ".html");

    if (!file.startsWith(ROOT) || !(await Bun.file(file).exists())) {
      return new Response("Not found", { status: 404 });
    }
    if (file.endsWith(".ts")) return bundle(file);
    return new Response(Bun.file(file));
  },
});

console.log(`Docs → http://localhost:${PORT}`);
