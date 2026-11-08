# Move to Next.js, React, and Tailwind

Supersedes [ADR 0001](./0001-keep-astro-as-the-experience-shell.md).

The site moves from Astro to Next.js (App Router), React, and Tailwind CSS. The reusable
components are shadcn components on Base UI. They are restyled onto one design system in
`src/ds/`.

## Why

- **One design system, read the same way everywhere.** Each design value is now stated once,
  in `src/ds/tokens/`. Before, values lived in DESIGN.md frontmatter, `theme.css`, and dozens
  of scoped `<style>` blocks. Tailwind utilities read the tokens, so a class can use only a
  value the system defines.
- **A component base to build on.** shadcn on Base UI supplies the accessible behaviour
  (focus, dismiss, keyboard, scroll lock), so the site does not hand-roll it. The look stays
  the site's own.
- **One component model.** Pages, atoms, and the interactive objects are all React. The
  framework-free controllers (deck, glide, pointer, room) are unchanged, and a small client
  component mounts each one.

## ADR 0001's rules still hold

- **Reading is static HTML.** `next build` runs as a static export (`output: 'export'`) into
  `dist/`. Pages are Server Components and render at build time. Only the interactive objects
  ship client JS.
- **Nothing essential needs JS.** Essays, In short, the book list, and navigation work without
  WebGL, animation, or JavaScript. The room falls back to its poster with real links. `three`
  loads only when the room starts (`scripts/check-bundle.mjs`).

## Consequences

- Redirects move to `vercel.json`, because a static export has no server to answer them.
- Markdown renders through unified and Shiki in `src/lib/markdown.ts`. It replaces Astro's
  content layer, and `src/lib/content.ts` validates frontmatter with zod.
- Media is published by `scripts/sync-assets.mjs` under content-hashed names, and the
  manifest is written to `src/generated/media.ts`.
- DESIGN.md no longer generates tokens. `scripts/check-design-tokens.mjs` fails the build
  when DESIGN.md and `src/ds/tokens/` disagree.
