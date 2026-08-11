---
version: 1
slug: "src-layouts-blogpost-astro"
primary_target: "src/layouts/BlogPost.astro"
related_targets: ["src/pages/writing/index.astro","src/pages/bookshelf/index.astro","src/pages/about.astro","src/layouts/Page.astro","src/styles/article.css","src/styles/site.css"]
---

# Reading and index surfaces

## Scope and visitor mode

The four non-home routes: the article reading page (`/writing/<slug>`), the writing archive (`/writing`), the bookshelf (`/bookshelf`), and About (`/about`). All four are **Read**: the visitor is here to understand something, so comprehension and wayfinding outrank expression.

The homepage keeps its own brief and its own seed. These surfaces inherit its visual world rather than restating it.

## Audience and job

A founder, executive, or CTO who arrived from the homepage index, a search result, or a shared link, plus technically curious peers. The job is to read one piece properly and then find a second reason to stay — another essay, a book, or a way to make contact.

Success is that a visitor landing directly on an article, with no memory of the homepage, recognises the same authorship and can navigate the whole site from where they landed.

## Proof and content

Real content only: the `blog` collection (four pieces, one with a French translation), the reading archive in `src/booksData.ts` (twelve books), and Mathieu's existing About prose in `src/content/about.md`. Article prose, titles, descriptions, and dates are authored; nothing on these surfaces is generated.

## Constraints

- **Routes renamed with redirects.** `/blog` → `/writing`, `/reads` → `/bookshelf`, `/about` added; 301s preserve inbound links. The nav vocabulary and the addresses must keep matching.
- **About prose is verbatim.** Layout and typography only. The user explicitly chose to keep the existing role sentence unflagged, against the option of marking it as an unresolved placeholder — so its divergence from PRODUCT.md's "role deliberately unresolved" is a recorded user decision, not an oversight.
- **No JavaScript dependency.** The bookshelf and archive render at build time. The article's margin apparatus is enhancement: every heading link works as a plain anchor without the marking script.
- **The legacy world is gone.** `src/styles/global.css` (Tailwind, `font-size: 85%`, Roboto Mono body) and the POC pages that were its only importers have been deleted, along with Tailwind itself. `BaseHead.astro` still imports no stylesheet: a layout owns its surface's world, and adding a stylesheet import there would apply it to all five surfaces at once.
- One entry per piece in the archive: translations are reached from the article's own apparatus, never listed as separate rows.

## Chosen direction and memorable moment

**The Marginalia Sheet** — candidate 4 of the grounded structural list, seed key `61ce0048` (`--scope surface --mode read`).

One measured reading column with a wide outer margin carrying the editorial apparatus. The memorable moment is the margin itself: the dated witness line and a contents list whose active entry takes the same orange star the homepage index uses, tracking a reading line just below the top of the viewport, plus a numbered section reference (`§` and its ordinal) hung in the outer margin level with each `h2` and linking to it. The contents list repeats those ordinals, so the margin references and the index are one apparatus keyed by one set of numbers. Headings that are themselves apparatus (an inline "Table of Contents") are skipped by `src/utils/sections.mjs`, so the index never lists itself as a section. Apparatus is keyed both globally and to a place in the text; the prose stands complete without either.

Code is set in the site's own five colours via `src/styles/code-theme.json` rather than an off-the-shelf highlighter theme, so a code block reads as part of the page instead of a pasted terminal.

## Cited adaptations

- **Bookshelf drops cover images.** They were hotlinked to `covers.openlibrary.org` and could fail; typography carries the metadata instead, and ratings render as a numeral rather than star glyphs (the craft floor refuses glyph icon systems).
- **About hides `about.md`'s own `# About Me`** with `display: none`, removing it from the accessibility tree too, because the page head already names the surface. The heading stays in the user's file.
- **Measure is set in `ch` at 52, not 68.** `ch` is the advance of "0", which is 1.42× Commissioner's average character width here, so `68ch` rendered 79–91 characters per line. `52ch` lands the real count at 67–73.

## Unresolved

- Section copy on the archive and bookshelf is marked placeholder; Mathieu writes it.
- Contact destinations are still unresolved sitewide, so the article's close offers adjacent essays and the archive but no conversation path. The footer links Writing, Bookshelf, About, and RSS only. When X and Telegram destinations exist, PRODUCT.md's stated success condition needs a real route from a reading surface, not only from the homepage close.
- No comp was rendered for these surfaces; no image generation was available in the build session. Composition was judged against the recorded OWN-WORLD.
- The artwork that carries the homepage has no role on any of these four surfaces. The bookshelf in particular now ships with no image material at all.
