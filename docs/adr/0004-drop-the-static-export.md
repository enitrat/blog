# Drop the static export; render markdown as React

Amends [ADR 0003](./0003-move-to-nextjs.md). Its rule "Reading is static HTML" still holds;
the way the build delivers it changes.

The site no longer builds with `output: 'export'`. It runs as a standard Next.js app on Vercel.
Every page still prerenders at build time: the static routes, plus `generateStaticParams` for
posts and work. Markdown renders to React elements, no longer to an HTML string.

## Why

- **The export forced a hand-built image pipeline.** `next/image` cannot optimise images in an
  export, so `scripts/sync-assets.mjs` hashed files, wrote WebP srcsets and generated a
  manifest. `next/image` does all of that itself now.
- **Redirects belong in `next.config.ts`.** They were in `vercel.json` only because an export
  has no server to answer them.
- **Markdown as an HTML string needed DOM patching.** After hydration, client effects cloned
  the Copy buttons into code blocks and the "In short" back links before headings. Rendering
  the hast tree with `hast-util-to-jsx-runtime` and a components map puts both in the markup.

## Consequences

- `src/lib/media.ts` loads an image under `src/assets/` or `src/content/` with a dynamic
  `import()`. The import returns `StaticImageData` for `next/image`. Each file extension needs
  its own import template there, because Turbopack fails on a template that matches no file.
- `src/lib/markdown.ts` returns a hast tree. `Prose` renders it, mapping `pre` to `CodeBlock`
  and `img` to `next/image`. The "In short" back links are now plain anchors in the HTML, so
  they work without JS.
- The browser checks run `next start` on the build (`serveBuild()` in `scripts/dev-server.mjs`),
  not a static file server. `scripts/check-bundle.mjs` reads the prerendered pages from
  `.next/server/app`.
- The room's `.glb` files are not published yet; nothing in the Next app loads them. When the
  room is ported, it needs a way to serve them, such as `public/` or an asset import.
