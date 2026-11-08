const savedTheme = localStorage.getItem("theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;

document.addEventListener("DOMContentLoaded", async () => {
  const pages = await fetch("/nav.json").then((r) => r.json());
  const here = location.pathname.replace(/\/$/, "") || "/";

  const sidebar = document.createElement("aside");
  sidebar.className = "sidebar";
  sidebar.innerHTML = `
    <div class="top">
      <a class="brand" href="/">Inference Engine</a>
      <button class="theme-toggle" type="button">Theme</button>
    </div>
    <ul>${pages
      .map(
        (p) => `
      <li class="${p.url === here ? "active" : ""}">
        <a href="${p.url}">${p.title}</a>
        <ul>${p.sections.map((s) => `<li><a href="${p.url}#${s.id}" data-id="${s.id}">${s.title}</a></li>`).join("")}</ul>
      </li>`,
      )
      .join("")}
    </ul>`;
  document.body.prepend(sidebar);

  sidebar.querySelector(".theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    localStorage.setItem("theme", root.dataset.theme);
  });

  const sections = [...document.querySelectorAll("main > section[id]")];
  const links = sidebar.querySelectorAll("li.active [data-id]");
  const highlight = () => {
    let current = sections[0]?.id;
    for (const s of sections) if (s.getBoundingClientRect().top < 120) current = s.id;
    links.forEach((a) => a.classList.toggle("current", a.dataset.id === current));
  };
  addEventListener("scroll", highlight, { passive: true });
  highlight();
});
