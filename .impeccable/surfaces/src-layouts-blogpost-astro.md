---
version: 1
slug: "src-layouts-blogpost-astro"
primary_target: "src/layouts/BlogPost.astro"
related_targets: ["src/pages/writing/index.astro","src/pages/about.astro","src/layouts/Page.astro","src/styles/article.css","src/styles/site.css"]
---

# Reading and index surfaces

## Mode
`/writing/<slug>`, `/writing`, `/reading`, `/places`, `/about` are Read surfaces in the bench's world.

## Essay page direction contract
THESIS: Plain and fast. An essay page is text set for long reading with an "In short" list at the top, so a reader with little time can skim before reading. It refuses the sidebar table of contents, the giant display title and the banner image.
OWN-WORLD: The bench's paper and ink without its tiles: near-white page, near-black text, one orange accent for hover and the current link only. Geist 18px, line height ~1.62, measure ~38rem, text justified with `hyphens: auto` (left-aligned fallback where hyphenation is unavailable). Geist Mono for the meta line, section numbers, "In short" label and captions.
STORY: The reader sees a small title and one mono meta line (name · date · read time), reads the "In short" list (numbered section titles, each with its first sentence verbatim), jumps to a section or reads straight through, and finds a quiet "↑ In short" at the end of each section.
FIRST VIEWPORT: Mono meta line at the top; title at ~1.75rem semibold; the "In short" block directly under it, ending with a hairline; the first section begins below. No banner image, no sidebar, nothing sticky.
FORM: Essay "In short", user-pinned (round 3, 5/5), with justified text from the Plain variant's note.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Rules
- Use real authored content. Never paraphrase: hooks are the first sentence of each section, extracted at build time. A post may override one hook in front matter later.
- Existing banner images are not shown on the essay page.
- Keep `/writing` and `/about` routes and legacy redirects. The site is English only; there are no translations.
- Usable without JavaScript: the "In short" block is generated at build time.
- Code uses site colours on a transparent ground.
