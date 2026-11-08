// "On this page" index, fixed at the top right like most docs sites.
// Lists each section's <h2> (title) and the <h3 id> inside it (subtitles). Clicking jumps there;
// the entry for the part you are reading is highlighted.

export function mountToc(): void {
  const main = document.querySelector("main");
  if (!main) return;

  const nav = document.createElement("nav");
  nav.className = "page-toc";
  nav.setAttribute("aria-label", "On this page");
  nav.innerHTML = `<div class="page-toc-title">On this page</div>`;
  const list = document.createElement("ul");
  nav.append(list);

  const targets: { el: HTMLElement; link: HTMLAnchorElement }[] = [];
  const add = (el: HTMLElement, id: string, text: string, sub: boolean) => {
    const li = document.createElement("li");
    if (sub) li.className = "sub";
    const a = document.createElement("a");
    a.href = `#${id}`;
    a.textContent = text;
    li.append(a);
    list.append(li);
    targets.push({ el, link: a });
  };

  for (const section of main.querySelectorAll<HTMLElement>(":scope > section[id]")) {
    const h2 = section.querySelector("h2");
    if (!h2) continue;
    add(section, section.id, h2.textContent ?? "", false);
    for (const h3 of section.querySelectorAll<HTMLElement>("h3[id]")) add(h3, h3.id, h3.textContent ?? "", true);
  }
  if (!targets.length) return;

  document.body.append(nav);
  document.body.classList.add("has-toc");

  // Highlight the last heading that has scrolled past the top of the window.
  let queued = false;
  const update = () => {
    queued = false;
    let current = targets[0];
    for (const t of targets) if (t.el.getBoundingClientRect().top <= 120) current = t;
    for (const t of targets) t.link.classList.toggle("active", t === current);
  };
  const queue = () => {
    if (!queued) (queued = true), requestAnimationFrame(update);
  };
  addEventListener("scroll", queue, { passive: true });
  addEventListener("resize", queue);
  update();
}
