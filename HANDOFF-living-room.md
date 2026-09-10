# Living room handoff

Updated: 2026-09-10

## Where this pass stopped

The desk pass is complete. The leather reading chair on the right of the room
is gone; in its place, along the open side, stands a fumed-oak pedestal writing
desk and a slatted side chair, both authored in `scripts/room.py` and baked
into `furniture.glb`. The desk carries a brass banker's lamp with an emerald
shade, a manuscript with a loose sheet, a fountain pen and an inkwell, all in
`objects.glb`. The lamp is a fourth practical light, so the whole room was
rebaked; the poster was regenerated and `bun run build` and
`bun run room:check chromium` pass. The user has seen the preview render and the
poster.

The scene now has a home for each strand: readings on the shelf, SoundCloud on
the record player, and the desk for the writings. Nothing in the runtime
targets the desk yet: no hotspot, no camera frame, no manuscript link. That is
the next step, and `scene.ts`/`bookshelf.ts` are where the shelf does the same.

The 55 isolated book bakes take about 21 of a full run's 25 minutes and cannot
see the rest of the room, so `bun run room:bake --room-only` now skips books,
covers and bookmark. Use it for any change that does not move a light or a
book.

## Previous pass

The book-opening pass is complete. The front covers bake from one temporary
joined mesh with the corrected hinge, while the exported covers keep their
individual hinges. The book bodies are baked in isolation under the room
lights and now carry that atlas: the previous `books.glb` had been exported
before its bake and shipped 55 untextured bodies. Both assets are in place and
consistent with each other.

The front covers are baked only for annotated volumes. All 55 shared one
4096px sheet, so the one cover a reader can actually open was lettered at eight
pixels per centimetre and looked pixelated on screen; it is now forty-four, and
`covers.glb` fell from 1.8MB to 439KB. The cost is that adding a note needs a
`--books-only` rebake to give that volume a cover. `scene.ts` treats a missing
cover as normal for an unannotated book and only falls back when an annotated
one has none.

The final working tree includes three runtime assets beyond the room:
`books.glb` with 55 book-body nodes, `covers.glb` with 55 hinged front covers,
and `bookmark.glb` with one reusable bookmark node. The poster was regenerated
after those assets were published. The user has not yet reviewed the result
visually.

The inspiration remains Henry Heffernan's portfolio, particularly its baked,
unlit rendering and coordinated camera position/focus transitions. Reference
source is available at `/tmp/henry-reference.Le1cOF/`; it is not a dependency.
No reference-site geometry or textures were copied.

The open book was a modal dialog, which made the rest of the document inert:
the page could not scroll behind it and a click on the shelf never reached the
canvas. Only the conventional record — reduced motion, or no renderer — is
modal now. The consequence to weigh is that a non-modal dialog does not contain
the tab order, so a reader can tab out of an open book; that follows from the
page behind it being live, and Escape now reaches the book from anywhere on the
page rather than only from inside the room.

## Current bookshelf implementation

- The hierarchy remains room → cabinet → row → book, in the existing Three.js
  scene. The legacy `/bookshelf/` archive and surrounding Astro site remain.
- Only books with authored notes can open. Other spines expose metadata on
  hover, focus, or tap. Old book URLs without notes normalize to their row.
- Annotated books have a warm tint and persistent ivory dot. Mouse hover has a
  100ms dwell; keyboard focus responds immediately. Hover tilts six degrees and
  moves the book forward 8mm. The animation clock resets after idle so the first
  frame does not consume a large part of the movement.
- Opening has a separate pose: the book straightens and slides 6cm forward.
  The native notes dialog appears after 280ms. Closing returns the book to its
  slot. Escape during extraction cancels the opening; browser history can also
  interrupt it. Reduced motion skips extraction and opens the dialog immediately.
- Reading status is independent of notes. A red textile bookmark attaches to
  each currently reading volume and moves with it. It is a short folded tab
  emerging between the top pages, with thickness, softened edges, a shaped end,
  woven bump detail, and Cycles lighting. No runtime strip geometry remains.
- The note for *East of Eden*, `src/content/notes/9780143129486.md`, retains the
  user's exact text, “A gift from Aurelien”. Preserve user authorship.
- Notes are discovered on the shelf. Do not restore the removed random-pick,
  “Find notes”, or “Read note” controls.

## Asset ownership

Read `src/assets/room/README.md` for commands and the full artifact contract.

| Concern | Owner |
|---|---|
| Geometry, materials, bake, bookmark | `scripts/room.py` |
| Book layout and cabinet manifests | `scripts/bake-room.mjs` |
| Printed spine artwork | `scripts/spine-art.mjs` |
| Rendering, camera, book movement | `src/components/home/room/scene.ts` |
| Navigation, focus, picking, notes dialog | `src/components/home/room/bookshelf.ts` |
| Records and HTML fallback | `src/components/home/room/Bookshelf.astro` |
| Reader layout and notes dots | `src/components/home/room/bookshelf.css` |
| Stage and controls | `src/styles/home.css` |

Seven GLBs load together. The new `books.glb` contains 55 `Body_<isbn>` nodes,
one per volume. Each origin is at the bottom front of the spine. Covers, leather
spine, and paper are joined within each volume. Book bodies are no longer sliced
out of `objects.glb` at runtime. Each body is baked independently under the same
room lights, with other meshes hidden, preserving its own cover/page shadows
without permanently darkening the surfaces behind neighbouring books.

The printed jacket atlas remains separate. Runtime slices each jacket using
`book-slots.json` and attaches its body to the same pivot. `bookmark.glb`
contains one reusable `Bookmark` node with its origin at the book's head.
The runtime places copies based on reading status, so changing reading status
or adding notes needs no rebake. Book dimensions, layout, and artwork changes
do require regeneration.

Default quality is 256 samples. Shell, furniture, and book bodies have 2048px
atlases. Objects and printed jackets have 4096px atlases. Turntable parts and
bookmark have 1024px atlases. Isolated book baking adds time because Blender
performs a separate bake for every volume.

Preserve source artwork UVs separately from bake UVs. Color is converted once
offline; runtime materials remain unlit. Moving books do not have dynamic
shadows. The new extraction is a translation into the native notes dialog,
not articulated 3D page turning.

Row views retain a 40-degree lens and raised camera to clear the lampshade.
Their 0.32m vertical frame leaves space for the bookmark during extraction.
The back control now sits between the arrows below the canvas, keeping the
cloth and book unobstructed.
Room and cabinet retain the authored 23.83-degree lens. Camera position,
focus, and lens interpolate together using quintic ease-in-out, following
Henry's camera approach. Durations remain 500ms entering a row, 280ms between
row frames, and 750ms approaching the cabinet.

## Interaction constraints

Preserve history, Escape, visible back controls, sibling-row history replacement,
and inactive near-miss gaps. Room picking is a raycast; row and spine picking
use projected HTML targets. Targets stay fixed during book movement. Controls
and captions occupy the strip below the canvas. Preserve the HTML fallback,
native dialog, and focus restoration. Notes content nodes move into the dialog
rather than being cloned, preserving their enhanced elements.

## Workspace and recovery

This work builds on the existing uncommitted bookshelf pass. Inspect `git status`
before editing. Leave `_lh.mjs`, `_sc.mjs`, and `bounce-nationale.m4a` alone.

Book-body source and intermediate textures are in
`/tmp/blog-bookshelf-refinement/`; the bake log is
`/tmp/blog-bookshelf-refinement.log`. Corrected static atlases are in
`/tmp/blog-bookshelf-static/`. The latest bookmark source, textures, and export are in
`/tmp/blog-bookshelf-review/`, with the bake log at
`/tmp/blog-bookshelf-review.log`. Its `bake-bookmark.py` is a temporary copy of
`scripts/room.py` that filters both bake/export loops to the bookmark group.
It uses the same 256-sample bake and 1024px atlas as the full asset pipeline.
The exported bookmark is copied to `src/assets/room/bookmark.glb`.
The previous darker export remains in `/tmp/blog-bookshelf-ribbon/`.
The bookmark template is hidden during static bakes so it cannot leave a shadow
on an unrelated book. Blender is available at
`/Volumes/Blender/Blender.app/Contents/MacOS/Blender`.
The previous pass remains under `/tmp/blog-bookshelf-polish/`.

Run Playwright with `node` from the repo root. Browser checks are authorized.
The prior no-tests instruction remains; this pass uses compilation and browser
inspection rather than adding a repository test suite.

## Verification

- `bun run build` passes, including the lazy Three.js bundle check.
- Targeted strict TypeScript compilation and Biome checks pass for `scene.ts`
  and `bookshelf.ts`. `home.css` has two existing `!important` warnings on the
  mix-credit positioning rules. This is not a whole-repository type-clean claim.
- `/tmp/check-living-room-review.mjs` checks Chromium at 1280×1000, 768×1024,
  and 390×844. It asserts delayed opening, authored note content, closing,
  focus restoration, Escape/history cancellation during extraction, immediate
  reduced-motion opening, dialog width, navigation placement without overlap,
  and no page errors. It also checks the no-JavaScript authored note.
- Deterministic animation captures use Playwright's clock. The 240ms capture
  precedes the 280ms dialog delay, so extraction can actually be inspected.
  Captures are `/tmp/room-review-{desktop,tablet,phone}-{rest,hover,extract,notes}.png`.
- The earlier clipping is visible in `/tmp/room-extract-400.png`.
  Henry's live portfolio overview is captured in `/tmp/henry-live.png`.
- The poster was regenerated from the final bookmark asset with controls hidden.
- Browser observations do not certify physical devices or other browser engines.

- `/tmp/check-modal.mjs` covers what a non-modal open book has to keep: the
  page scrolling behind it, the note still scrolling under the pointer, a
  background click closing it, Escape with focus restored, and reduced motion
  staying a conventional modal. `/tmp/check-escape-out.mjs` checks Escape after
  the tab order has left the book.
- `/tmp/check-book-opening.mjs` passes at 1280x1000, 768x1024, and 390x844
  against the production textures. It covers keyboard opening, the cover and
  page sequence, closing, three interruption points during extraction, history
  navigation, focus restoration, resizing while a book is open, long-note
  scrolling, and switching to reduced motion mid-open. It also checks the
  direct note link and the no-JavaScript authored note. It also guards the
  reach-and-release regression below. Captures are
  `/tmp/book-final-{desktop,tablet,phone}-{cover,hinge,note}.png`.
- A frame's rAF timestamp is the moment the frame began, which can fall a few
  milliseconds behind a `performance.now()` read while scheduling it. `tick`
  used that difference unclamped, so a cold start from `bookshelf.pulls` could
  produce a negative `dt`. Divided by the negative closing rate in `pull`, that
  opened a book nobody had reached for: `data-book-moving` went on, `home.css`
  hid the targets, controls, and caption, and the on-demand loop then stopped
  with nothing left to animate, so the attribute never cleared. Hovering a
  spine and moving the pointer away emptied the whole row view until the next
  real animation. `dt` is now clamped at zero.

The user has not approved the latest appearance. The next step is their visual
review of the opened spread, the cloth, and navigation placement. The right
page of the spread is a flat panel next to the warm baked left page; decide
whether that contrast is wanted. No next room area has been selected.
