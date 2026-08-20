---
version: 1
slug: "src-layouts-blogpost-astro"
primary_target: "src/layouts/BlogPost.astro"
related_targets: ["src/pages/writing/index.astro","src/pages/bookshelf/index.astro","src/pages/about.astro","src/layouts/Page.astro","src/styles/article.css","src/styles/site.css"]
---

# Reading and index surfaces

## Mode

`/writing/<slug>`, `/writing`, `/bookshelf`, and `/about` are Read surfaces. They inherit the Living Index world but prioritize comprehension and wayfinding.

## Rules

- Use real authored content from the blog collection, reading archive, and About page.
- Keep `/writing`, `/bookshelf`, and `/about` as the visible route vocabulary; preserve legacy redirects.
- Keep About prose verbatim; layout and typography may change.
- Use one archive row per piece. Reach translations from the article, never as duplicate rows.
- Keep the site usable without JavaScript. The margin is enhancement, not navigation.
- Use the Marginalia Sheet: page head, measured `52ch` prose, bounded margin, one contents list, and onward exits.
- Use site-colour code on a transparent ground; do not import a terminal theme.
- Bookshelf uses typography and numeric ratings, not remote cover images or star glyphs.
- Homepage artwork does not appear on these surfaces.

## Open decisions

- Archive and bookshelf standfirsts remain placeholders until Mathieu writes them.
- Contact destinations remain unresolved.
