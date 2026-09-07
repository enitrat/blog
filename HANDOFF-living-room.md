# Living room handoff

Updated: 2026-09-07
Next focus: refine the bookshelf experience inside the approved Blender room.

## Start here

1. Inspect `git status` and the current source before editing. HEAD is `3a06628`; the bookshelf work described here is uncommitted. Checkpoint `04c0305` owns the approved room composition and materials. Later commits and the working tree contain interaction changes, not a new visual baseline.
2. Read [the asset README](src/assets/room/README.md) for the pipeline, manifest, current navigation mechanics, and regeneration commands. Use the ownership map below for the concern being changed.
3. Preserve the settled interaction hierarchy and latest cue choices below. The user called the first implementation a "good start" but "not excellent quality." Agreement on a direction is not certification of the rendered result.
4. Continue under the verification limits below. This handoff does not authorize a browser inspection, poster capture, tests, or a redesign of the room.

This document replaces the previous handoff and its contradictory Opus appendices. It records decisions and lessons; source files, commits, and the asset README own implementation details. The user explicitly requested this repository file, overriding the handoff skill's default temporary-directory destination.

## Binding instructions and visual authority

- The user said "Dont run tests, dont visually verify at first: just implement the scene properly." Earlier instructions also prohibited writing tests, and no test writing or execution is authorized. **Browser inspection is now authorized**: the user said "you can do browser checks if you need". Playwright's chromium is installed and `scripts/shoot-poster.mjs` shows the pattern — a scratch script that starts `astro dev`, drives the page, and screenshots `.living-room__canvas`. Run it with `node` from the repo root, not `bun`: bun resolves a cached playwright that has no browser. Poster regeneration is still not requested.
- The room remains an authored, baked Blender scene with an eighties library and vinyl-player setting. Preserve its approved composition, materials, and surrounding site's identity. There is no walking character.
- Older design documents contain a prohibition on homepage artwork. The later room request and approval supersede that prohibition for this section. Consult `PRODUCT.md`, `CONTEXT.md`, and `DESIGN.md` for the surrounding site, not as grounds to remove the room.
- Keep user-authored prose and the real reading archive. Notes must come from actual content, not invented examples presented as the user's writing.
- The user warned that Opus's implementation was likely poor. Review its reasoning and actual behavior independently. This is permission to correct defects, not a mandate to discard useful assets or rebuild working infrastructure.
- Ponytail mode remains active: understand the complete flow, then make the smallest sound change. It does not justify lowering visual quality or omitting requested behavior. Its test-writing guidance is overridden by the user's no-tests instruction.
- Historical permission for a Luna research task does not authorize a new agent team.

## Settled experience and latest corrections

The user approved **room → whole cabinet → row → book**.

The first bookshelf click is an approach and orientation step. Visitors see the cabinet's arrangement before choosing a row. A row click reaches reading distance; individual books then become inspectable. If a row cannot fit legibly, horizontal movement stays within that row. Vertical row changes have separate controls.

The user then refined the cues twice:

- Replace rectangular highlights with the same unfilled ivory dots used elsewhere in the room.
- Center the cues and show a **pointer** cursor on hover. This supersedes the earlier recommendation to use a zoom-in cursor over rows.

The latest implementation centers each row dot horizontally and vertically within its row target. The entire invisible row target remains clickable. The cabinet body also uses a pointer cursor. Individual books reveal a small unfilled dot on hover or keyboard focus; note ribbons remain a separate content indicator. The original room hotspot anchor was not repositioned in the last small patch.

Clear background uses the zoom-out cursor and returns one level. Narrow gaps between spines are deliberately inactive: missing a thin book must not eject the visitor. A visible back control, Escape, and native history complement the cursor for keyboard and touch users. Moving along a row or between sibling rows replaces the current history entry, rather than making Back retrace every movement.

Full mechanics and URL forms are documented in the asset README. Do not revive numbered sections as the visitor's organizing model. Sections are an internal framing calculation; rows are the visible structure.

## What we learned and corrected

### Preserve the room while adding a browsing mode

Opus previously mounted the old bookshelf as a fullscreen layer over the room. The user rejected that because it was effectively another screen. Reusing a large existing implementation is not a saving if it reproduces the wrong experience.

The current bookshelf stays in the same Three.js scene and uses the actual Blender cabinet. A second scene or higher-detail asset remains possible if later evidence warrants it, but is not a prerequisite for camera movement. Reading an individual book in a DOM dialog is distinct from replacing the entire browsing experience with an overlay.

Our first implementation also made a design mistake: it went directly to part of one row. The user's correction restored the whole-cabinet orientation step. This is not a return to Opus's repeated clicks on the same ring; the second click now chooses a specific row.

### Texture resolution and display size are separate constraints

The claim that baked text cannot be zoomed was false. Printed planes lost detail when they shared atlas space with much larger physical surfaces. A cabinet-wide atlas would still waste most pixels on bindings and page edges. Separating only the printed spines was a useful Opus change worth retaining.

The next correction was at source rasterization. A larger output PNG is not proof that an SVG was rasterized at that resolution. The generator now supplies rasterization density rather than enlarging a low-resolution raster afterward.

Automatic UV packing also left substantial empty space around the narrow spine islands. The current bake assigns regular print cells. See the README for dimensions; those are allocation calculations, not a browser legibility measurement.

Higher-resolution text still needs enough screen space. The runtime sizes reading sections from book bounds and viewport width. The resulting framing, smallest text, and touch-target behavior have not been visually verified in these passes. When that phase is requested, judge actual CSS display size, not a magnified crop.

### Asset identity does not require a runtime mesh per book

The old runtime had no reliable ISBN-to-position mapping and the generator silently stopped after a fixed number of books. The bake now produces a manifest from the positions Blender consumes. The page checks ISBN coverage, and a full cabinet fails explicitly instead of dropping books.

An initial bake with separate printed meshes repeated expensive scene synchronization. That attempt was stopped; the successful bake joins the printed planes after assigning their UV cells. The manifest preserves identity for projected HTML links while the atlas is baked and drawn as one mesh.

Keep the two kinds of picking distinct: the cabinet body uses a raycast against the room; rows and individual books use projected HTML targets. The invisible rectangles are hit areas, not the requested visual cues.

### Native behavior must survive renderer limitations

The inherited coarse-pointer exclusion prevented touch devices from loading the room. Opus's change from a bookshelf link to a button then left it without an action when WebGL was unavailable. The current DOM controller owns navigation and records independently of Three.js. The load path admits touch devices, supports explicit activation, and retains an HTML list.

Focus, input during camera travel, and history need their own ownership. The current controller gates scene targets until the camera settles, distinguishes cabinet and row state, restores focus, and keeps near misses inactive. These are implemented behaviors, not claims that all browser edge cases were exercised.

## Where the code lives

| Concern | Read |
|---|---|
| Room loading, scene lifetime, camera movement, cabinet raycast, projection, turntable transforms | `src/components/home/room/scene.ts` |
| Cabinet/row/book state, history, row and spine targets, keyboard, swipe, focus, native record dialog | `src/components/home/room/bookshelf.ts` |
| Server-rendered records, actual note content, fallback list, navigation controls | `src/components/home/room/Bookshelf.astro` |
| Dots, note ribbons, reader styling, bookshelf controls | `src/components/home/room/bookshelf.css` |
| Poster, renderer load gate, SoundCloud integration, stop control during browsing | `src/components/home/LivingRoom.astro` |
| Room stage dimensions, overview hotspots, background cursor, touch layout | `src/styles/home.css` |
| Book layout manifest, source spine artwork, bake invocation | `scripts/bake-room.mjs` |
| Geometry, lights, UV assignment, baking and GLB export | `scripts/room.py` |
| Bake modes, artifact contract and operational details | `src/assets/room/README.md` |
| Actual library, metadata, note identity and binding palette | `src/booksData.ts`, `src/utils/bookRecord.ts`, `src/utils/notes.ts`, `src/utils/pleiade.ts` |
| Legacy standalone bookshelf and page-flip reader | `src/pages/bookshelf/`, `src/components/bookshelf/library/` |

Read `docs/adr/0001-keep-astro-as-the-experience-shell.md` before proposing a framework change. The room is part of an Astro site.

## This session: the spines and the gesture

The user called the zoomed-in books bad: inconsistent heights, thickness that ignored the page count, lettering that did not fit. All three came from one place. `bake-room.mjs` drew every jacket at a fixed 128x768 and stretched it onto a plane whose aspect ranges 1:5.5 to 1:12, jittered the heights by `index % 4`, and fitted the author line with `textLength` — which the renderer behind `sharp` ignores, so lettering that did not fit ran off the leather instead of being condensed.

`scripts/spine-art.mjs` now owns the drawing: gold ribbing, a label plate at the same height on every volume, author and title sized to fit the plate over as few lines as that size allows. Each jacket is authored at its own spine's aspect. Every volume is 237mm tall; only thickness varies, 19mm to 45mm with the page count. Atlas cells were reshaped from 13x5 to 26x3, about 158x1365, because cap height runs along the tall axis. See the asset README for the numbers.

Then the hover gesture, twice. First a 20mm translate of the jacket alone, which was wrong: the user wanted a book tipped out by a finger on its top. That needs the leather body too, but not a new bake — both halves cut out at load, `TILT` at 12 degrees. See the two entries below.

## Remaining work and limits

- **Visual refinement is partly done.** The spine artwork, the shelf at reading distance, and the hover pull were checked in a real browser at 1280x900. Camera composition, transitions, dot placement, phone framing, and reduced motion have still never been seen. The overview hotspot has not moved from where the bake put it.
- **A hovered volume leans out of the row.** `bookshelf.pulls(listener)` reports which spine is being reached for; the scene tips that whole book `TILT` (12 degrees) about its bottom front edge and back, snapping instead under reduced motion. The user asked for about twenty; twenty swings the top out of the framed row and reads as toppling, so it was dialled back. Turn `TILT` alone to change it.

  Both halves of a book move, so both are cut out of the bake at load: the printed jacket from `spines.glb` and the leather body from the joined `objects.glb`, matched to their slot in `book-slots.json` by triangle centroid. Verified against the shipped GLBs: 2 jacket triangles and 84 to 88 body triangles per book, all 55 books, nothing else claimed. It is all of them or none — a half-loose shelf is worse than a row that cannot lean.

  The one trap here, already paid for: the sliced geometry has the baked node's transform baked into it, so what the room keeps has to be drawn as its own mesh with the original node hidden. Handing it back to the node applies that transform twice and flings half the room across the picture.
- **The old archive still exists.** `/bookshelf/`, its separate runtime, and existing navigation links remain. The longer-term intent is one Blender-authored bookshelf, but migration and removal were deliberately deferred until the room experience is ready. Do not mistake the retained archive for the desired final architecture.
- **No authored notes exist yet.** `src/content/notes/` contains only `.gitkeep`. No real book currently displays a notes ribbon. The new room reader renders actual note content into a native dialog when available. It does not reuse the legacy iframe/page-flip reader, physically extract a 3D volume, or simulate turning pages. Those effects were not implemented in this pass.
- **SoundCloud truth is still unresolved.** `LivingRoom.astro` still calls `state(true)` on playback intent before a confirmed PLAY event. The platter can therefore follow intent rather than audible playback. A stop button was added during bookshelf browsing; that did not fix the earlier crackle-without-music report, asynchronous loading races, or audio cleanup concerns. Diagnose these when playback work is requested.
- **High-resolution assets are not loaded separately on approach.** The live-room loader currently fetches all five GLBs together. The new atlas is a detail improvement, not an implemented streaming or level-of-detail system.
- **Library changes require appropriate asset regeneration.** ISBN coverage checks do not certify that changed titles, page counts, or other metadata match old artwork and geometry. Treat typography refinements as source-artwork work in `scripts/spine-art.mjs`, not something a camera adjustment can repair.
- **A leaning book shows a black top and black flanks.** Reported by the user against the shipped tilt, and diagnosed but not fixed. Two causes, one bake. The top is leather because the `Bible paper edges` block is inset 4mm under the solid top face of the `Pleiade leather binding` box, so no page edge is ever visible; letting the paper block finish flush with or a hair proud of the leather would show a cream top with a leather rim, which is what a closed book looks like from above. The flanks are black because they were baked wedged in a row with 1.5mm between spines and no light reaching them — the objects atlas shows the cream page-block islands lit and the leather side islands genuinely near-black, so this is baked lighting, not a missing UV. A leaning book is a state the room was never baked for. Fixing it means a fill inside the carcass, or more air between volumes, and then a full rebake: judge the resting row after any such change, not only the leaning book.

- **The poster is stale relative to the new print atlas and the new spine shapes.** Books are now one height with page-proportional thickness, so the poster shows a shelf that no longer exists. `bun run room:poster` would fix it and has not been asked for. Keep the opaque renderer and hide all overlaid controls in a future poster capture.
- **Preserve bake invariants.** Source artwork UVs and bake UVs serve different purposes. Color conversion happens once, offline. Materials remain unlit in the browser. Baked contact shadows limit physical book movement; do not introduce extraction animations without accounting for exposed surfaces and painted shadows. Model pivots must respect Blender's Z-up to Three.js's Y-up conversion.

## Verification and working-tree status

The spine-only Blender bake completed and exported one primitive. Plain `astro build` completed with 19 pages after the hierarchy and unfilled-dot changes. A targeted strict TypeScript compile passed for `bookshelf.ts` and `scene.ts`. The final center/pointer-only patch was applied afterward without another compile.

No tests were written or run. No browser or screenshot verification was performed in the implementation passes. A static style scan reported palette/type-ramp advisories against older design documentation; that is not visual verification. An earlier whole-project TypeScript invocation encountered existing `astro.config.mjs` plugin-type conflicts and missing `bun:test` declarations. Do not describe the entire repository as type-clean.

Use plain compilation if continuing under the current restrictions. Read `package.json` before invoking wrappers: the normal build script includes additional checks. Legacy room/library check scripts are not evidence that this new interaction is validated; do not run or repair them under the no-tests instruction.

The working tree combines earlier Opus changes and this session's changes. None of this session's implementation was committed or pushed. In particular, preserve the untracked `book-slots.json`, `spines.glb`, and new bookshelf modules. Inspect the diff rather than resetting to HEAD, which lacks this work. Leave unrelated scratch files `_lh.mjs`, `_sc.mjs`, and `bounce-nationale.m4a` alone.

Temporary recovery aids, not dependencies:

- `/tmp/blog-bookshelf-bake/`: generated source artwork and Blender source from the recent bake. The saved `.blend` precedes the final bake/export steps; the scripts remain authoritative.
- `/tmp/blog-bookshelf-bake.log`: successful spine export log.
- `/tmp/blog-bookshelf-build.log`: latest plain build log.
- `/Volumes/Blender/Blender.app/Contents/MacOS/Blender`: executable used for baking, if the disk image is still mounted.

## Suggested skills

Call the Skill tool when available, or read the skill through the available skill mechanism. Apply only those relevant to the next requested task; user verification restrictions override skill defaults.

| Skill | When to use it |
|---|---|
| `impeccable` | Refine the agreed bookshelf interaction or conduct a requested visual pass within the approved room. |
| `animate` or `emil-design-eng` | Refine camera transitions, interruption, focus feedback, or reduced motion. |
| `coding-standards` | Change the TypeScript controller or renderer. |
| `diagnosing-bugs` | Investigate a concrete interaction, rendering, or playback failure, respecting the no-tests instruction. |
| `comments-best-practices` | Explain a non-obvious UV, ownership, or lifecycle constraint. |
| `agent-conv-cli` | Recover a decision only when this handoff and referenced artifacts are insufficient. |
| `unslop` | Write labels, explanations, or documentation. |
| `handoff` and `writing-for-agents` | Replace this document with the next consolidated state. |

Image generation is not needed merely to refine these interactions. Blender owns the physical assets; a browser capture owns the eventual runtime poster.
