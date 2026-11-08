// Theme tokens for code that can't use CSS variables directly (canvas, WebGL).
// SVG layers should keep using the .diagram classes in docs.css instead; they follow the theme on their own.

const TOKENS = {
  text: "--text",
  muted: "--muted",
  border: "--border",
  bgSoft: "--bg-soft",
  accent: "--accent",
  bg: "--bg",
  hover: "--hover",
  noteBg: "--note-bg",
} as const;

export type Theme = { readonly [K in keyof typeof TOKENS]: string } & { readonly dark: boolean };

export function readTheme(): Theme {
  const style = getComputedStyle(document.documentElement);
  const theme = { dark: style.colorScheme.includes("dark") } as Record<string, string | boolean>;
  for (const [key, cssVar] of Object.entries(TOKENS)) theme[key] = style.getPropertyValue(cssVar).trim();
  return theme as Theme;
}

/** Calls `cb` with fresh tokens whenever the theme toggle or the OS color scheme changes. Returns an unsubscribe. */
export function onThemeChange(cb: (theme: Theme) => void): () => void {
  const fire = () => cb(readTheme());
  const observer = new MutationObserver(fire);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const media = matchMedia("(prefers-color-scheme: dark)");
  media.addEventListener("change", fire);
  return () => {
    observer.disconnect();
    media.removeEventListener("change", fire);
  };
}
