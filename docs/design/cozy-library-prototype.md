# Cozy library prototype

Status: working experiment, awaiting Mathieu's visual and interaction review.
Branch: `prototype/cozy-library`.

## Question

Does a substantial oak bookcase, with books that pull out and open into turning pages,
make the bookshelf feel like a cozy personal library?

The agreed direction is a nearly full-screen bookcase, several levels, warm light,
Pléiade spines, and a slightly angled view. The room is suggested around the case.
A hover partly extracts a book. Selection clears the shelf, presents the cover in the
foreground, and opens it. The library remains behind the reader. Closing returns the
book to its place. Notes belong on the pages, including page turns for longer notes.

## Try it

```sh
bun install
bun run prototype
```

Open `/bookshelf/prototype/`. The existing `/bookshelf/` remains the reading archive.
Every book uses existing catalog metadata and clearly labeled sample pages. No personal
reflections have been invented. The prototype routes have `noindex` metadata.

With that server running, check the interaction with:

```sh
bun run prototype:check
```

Use `PROTOTYPE_URL=http://127.0.0.1:PORT bun run prototype:check` if the dev server
selected another port. Playwright requires its Chromium and WebKit browsers:
`bunx playwright install chromium webkit`.

## What this tests

- Plain Three.js for a stationary case and book extraction; no React integration.
- Procedural wood and spine textures; no copied demo assets or models.
- A separate foreground canvas keeps the book visible outside the shelf's bounds.
- StPageFlip for hard covers, HTML pages, touch gestures, and page turning.
- An iframe owns the reader's lifetime. StPageFlip 2.0.7 keeps scheduling frames after
  its documented destroy method, so removing the iframe stops its loop completely.
- The shelf renders when its geometry changes. Foreground animation redraws the
  selected book. Large cover textures exist only while that book is selected.
- Keyboard book selection, dialog focus, Escape, reduced motion, and an ordinary
  archive link when JavaScript is unavailable.

## Still to decide

- Whether the wood, lighting, viewing angle, and extraction feel right. These are
  provisional, not a new site-wide design system.
- How authored Markdown is divided into pages. The prototype uses short fixed sample
  pages. Automatic pagination and real book notes are not implemented.
- Mobile layout: this experiment arranges the case into a tall, scrollable single bay.
- Resizing the shelf while reading returns the book immediately and reframes the case.
- Actual phone GPU performance still needs a physical-device check. Browser viewport
  emulation does not establish mobile frame rate.

When the direction is accepted, update PRODUCT.md, CONTEXT.md, and DESIGN.md to reflect
notes inside Bookshelf and the library's motion. Do not promote placeholder prose.

## References

- [The Complete Shelf](https://mengto.github.io/complete-shelf/) inspired the selection
  sequence. Its code and assets were not copied; no explicit reuse license was found.
- [StPageFlip](https://github.com/Nodlik/StPageFlip), MIT, supplies page mechanics.
- [Three.js rendering on demand](https://threejs.org/manual/en/rendering-on-demand.html).
