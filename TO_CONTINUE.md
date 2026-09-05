# Continue the cozy library prototype

## Where we stopped

Mathieu approved the direction and asked us to build the first working prototype.
It is implemented on branch `prototype/cozy-library`, with uncommitted changes.
The original `/bookshelf/` page is unchanged. Nothing has been deployed.

The latest dev server was serving:

http://127.0.0.1:4322/bookshelf/prototype/

To restart:

```sh
bun install
bun run prototype
```

Use the port Astro prints. `bun run prototype` opens `/bookshelf/prototype/`.

## The agreed experience

A substantial oak bookcase nearly fills the page. It has several shelves of Pléiade
books, warm light, visible depth, and a slight viewing angle. Only a little surrounding
wall is needed to suggest a cozy room. A freely moving camera is not required.

1. Hover a book: it slides partly out along its depth.
2. Select it: it clears its neighbors, rotates to reveal its cover, and moves forward.
3. The cover opens into readable pages, with the library still behind the book.
4. Longer personal notes belong on actual turning pages. Mathieu explicitly chose this
   over a scrollable reading panel framed by a book.
5. Closing the book returns it to its place on the shelf.

Notes belong inside Bookshelf, not in Writing. PRODUCT.md, CONTEXT.md, and DESIGN.md
are open to revision. Their current contrary language has not yet been changed because
this is still a prototype.

Mathieu found The Complete Shelf somewhat laggy but liked its direction. He wants a
cozier, larger bookcase with multiple levels. Do not copy that demo wholesale or assume
its performance problems have been diagnosed.

## What is implemented

- Astro remains the site framework. Added `three`, `page-flip`, and `@types/three`.
- All 55 existing catalog entries appear, using the existing author/color mapping.
- Each book can demonstrate extraction, opening, page turning, and returning.
- Wood, spines, and covers are generated with canvas textures. No demo code, models,
  or artwork were copied.
- Desktop has a multi-level bookcase with two bays. Mobile uses one tall, scrollable bay.
- The selected book moves through a separate transparent foreground canvas so it is
  not clipped by the original shelf canvas.
- StPageFlip supplies the HTML page-turning and hard-cover mechanics.
- Sample pages are clearly labeled. No personal book reflections were invented.
- Native dialog, keyboard book buttons, Escape, focus return, and reduced motion are
  implemented. The ordinary archive remains available without JavaScript.

### Performance choices

The stationary Three.js scene stops scheduling frames when idle. During extraction,
only the foreground book needs repeated drawing once shelf hover motion has settled.
Full-size cover textures are created only for the selected book and released on return.
Pixel ratio is capped at 1.5. Lighting uses simple lights and painted shading rather than
live shadow maps. The foreground WebGL context is explicitly released on close.

StPageFlip 2.0.7 schedules animation frames perpetually, including after its documented
`destroy()` call. The reader therefore lives in an iframe. Removing that iframe when
the book closes ends the reader's animation loop and event lifecycle.

## Files to read

- `src/pages/bookshelf/prototype.astro`: prototype host, dialog, controls, archive link.
- `src/components/bookshelf/prototype/library.ts`: scene, textures, layout, extraction,
  return, resource ownership, and communication with the reader.
- `src/components/bookshelf/prototype/library.css`: provisional visual treatment.
- `src/pages/bookshelf/prototype-reader.astro`: iframe, sample HTML pages, StPageFlip.
- `src/components/bookshelf/prototype/page-flip.d.ts`: narrow types for its untyped ESM build.
- `scripts/check-library-prototype.mjs`: runnable browser verification.
- `docs/design/cozy-library-prototype.md`: experiment brief and references.
- `src/booksData.ts`, `src/utils/pleiade.ts`: existing catalog and author styles.

## Verification at handoff

Luna-high reported:

- Chromium passed at 1440×900, 979×900, 760×900, 390×844, and 320×568.
- Opening, turning pages, returning, interrupted extraction, and reduced motion passed.
- All 55 books rendered; no horizontal overflow at the smallest viewport.
- The longest catalog title fit the sample pages at 320px. Its toolbar label overlapped
  the reader slightly; a subsequent CSS change truncates the toolbar label with an ellipsis.
- Build, token check, lint, focused TypeScript check, and `git diff --check` passed before
  the last small reader/CSS changes. Do not present those as checks of the final file state.

WebKit initially failed the Next-page control. Instrumentation established that valid
parent-to-iframe messages had the correct origin but `event.source === window` inside
the child, rather than `event.source === parent`. Chromium reported the parent normally.
The button fired, waiting longer did not help, and ArrowRight inside the iframe worked.

The latest patch retains the origin check and accepts a source equal to either the
parent or the child window. The targeted WebKit recheck at 390×844 passed: open, Next to Page 2, Previous, and
Escape close. Formatting of the final reader/CSS changes also passed. The full matrix
and build were not rerun after these last changes.

Checks, with the dev server running:

```sh
PROTOTYPE_URL=http://127.0.0.1:4322 bun run prototype:check
bun run build
bun run tokens:check
bun run lint
bunx tsc --noEmit --skipLibCheck --moduleResolution bundler --module esnext --target es2022 --strict src/components/bookshelf/prototype/library.ts src/components/bookshelf/prototype/page-flip.d.ts
git diff --check
```

Install Playwright browsers if missing: `bunx playwright install chromium webkit`.
Screenshots from verification are in `/tmp/cozy-library-check/`; these are temporary.
Browser emulation does not establish performance or gesture quality on an actual phone.

## What to do next

1. Finish the full matrix and build checks for the final file state, then capture the
   experiment on the prototype branch. The targeted WebKit control recheck already passed.
2. Let Mathieu judge the live experience before expanding the implementation. Ask what
   feels wrong about the wood, scale, lighting, viewing angle, or book movement.
3. Inspect the cover-to-HTML handover and reverse motion slowly. This remains a visual
   approximation between two renderers, not one continuously deforming book mesh.
4. Test on a real iPhone and desktop. Measure frame timing and memory across repeated
   opens. Improve only the bottlenecks actually observed.
5. Obtain one real note from Mathieu and test it across the page layouts. The current
   fixed sample page breaks do not solve automatic pagination for arbitrary Markdown.

## Prewalk: production-quality Three.js

Before promoting the prototype, investigate these leads and record the tradeoffs:

- Profile a real phone first: frame time during extraction, draw calls, texture memory,
  and repeated open/close cycles. Use measurements to decide whether shared geometry,
  texture atlases, or instancing would help; preserve rendering on demand.
- Give scene setup, animation state, and the HTML reader clear ownership. Make opening,
  cancellation, returning, resize, and teardown explicit transitions; verify resource
  cleanup, WebGL context loss/recovery, and browser back/forward restoration.
- Reassess the two-renderer handover and iframe workaround before treating them as
  permanent architecture. Compare visual continuity, lifecycle reliability, and cost
  against a simpler shared renderer; check the current Three.js and StPageFlip sources.
- Establish an asset and lighting approach with one finished oak panel and one finished
  book. Investigate physically based materials and baked contact lighting; compare the
  visual gain and GPU cost before adding real-time shadows or post-processing.
- Promote only after a real authored note works across page sizes, keyboard/touch input,
  reduced motion, and the plain HTML reading path. Retain repeatable browser checks and
  a physical-device performance baseline alongside the production implementation.

## Future improvements, after the prototype earns them

- More convincing oak grain, leather, paper edges, and contact shading. The current
  materials are procedural placeholders for judging the scene, not finished art assets.
- Refine how much bookcase fills the desktop viewport and whether mobile should use a
  tall bay or a different browsing arrangement.
- Preserve the open reader across resizing. Currently resizing the shelf closes it and
  reframes the case immediately.
- Build the authored-note model, likely Markdown in an Astro content collection, with
  stable book URLs and an ordinary HTML reading path independent of JavaScript.
- Decide how notes without commentary behave and how visitors recognize books with notes.
- Design pagination for real prose, including font loading, narrow screens, links,
  headings, long paragraphs, and text resizing. StPageFlip expects already divided pages.
- Audit the final reader for accessibility with real content and assistive technology.
- Update the product/design vocabulary once this direction is accepted. Keep experimental
  routes and sample copy out of the shipped site; `noindex` is not a deployment gate.

## Sources and constraints

- https://mengto.github.io/complete-shelf/ — interaction inspiration. No explicit reuse
  license was found; its source/assets were not adopted.
- https://github.com/Nodlik/StPageFlip — MIT page-turn implementation.
- https://threejs.org/manual/en/rendering-on-demand.html — rendering only when needed.

Mathieu's latest workflow instruction: delegate verification and other low-complexity,
high-token work to **Luna with high reasoning**. Keep engineering decisions and fixes
with the primary agent. Do not spend the next session repeating the research or widening
scope before he has reviewed the prototype.
