# Living room assets

Original geometry, materials and print artwork are authored in `scripts/room.py`
and `scripts/bake-room.mjs`. The spines use Mathieu's `src/booksData.ts` library.
No models or textures from the reference websites are included.

The rendering approach follows Henry Heffernan's
[baked models](https://github.com/henryjeff/portfolio-website/blob/master/src/Application/Utils/BakedModel.ts):
glTF geometry with baked color and light, displayed using unlit materials.
Camera position and focus move together with quintic ease-in-out, following his
[camera transitions](https://github.com/henryjeff/portfolio-website/blob/master/src/Application/Camera/Camera.ts).

With Blender 4.5 available, run:

```sh
BLENDER=/path/to/blender bun run room:bake
bun run room:poster
```

Set `ROOM_WORK` to keep the generated `.blend` and source textures in a chosen
directory. `--preview` renders a Cycles still without exporting assets.
`moving.glb` holds the two parts the runtime turns — the platter and the tonearm
— as separate nodes whose origin sits on the axis they turn about. They share one
atlas because a group's parts unwrap together, and they are baked in place, so
their own contact shadows read correctly at rest.

`spines.glb` holds the printed jackets on a separate 4096px atlas. `spine-art.mjs`
draws each one: gold ribbing over the leather, a label plate at the same height on
every volume, author and title in gold capitals fitted to the plate rather than
condensed into it — the renderer behind `sharp` ignores `textLength`, so lettering
that does not fit runs off the leather instead of being squeezed. Each drawing is
authored at its own spine's aspect and rasterized from that, never at a fixed size
and stretched onto the plane.

The bake assigns each jacket a regular atlas cell with a 10px gutter, shaped like
a spine at about one to twelve: cap height runs along the tall axis. For 55 books
this is a 26×3 grid, about 158×1365 pixels per jacket. The planes are joined after
UV packing for one bake and one runtime draw call.

A Pléiade volume is one height whatever it holds, so every slot is 237mm tall;
only thickness varies, with the page count, from 19mm to 45mm.

`book-slots.json` records ISBN, shelf row, spine bounds, and position in Three.js
coordinates. `bake-room.mjs` generates these slots; Blender consumes them to
place the actual books and exports the manifest alongside the GLBs. Rebuild
the room after changing the library. The generator rejects a full cabinet
instead of silently omitting books. The page build rejects missing ISBNs.

`cabinet.json` is the bookcase itself: centre, width, depth, front planes, the
heights of the plinth and crown, and the shelves. The `CABINET` literal at the
top of `bake-room.mjs` is the only place any of it is written down. Blender
builds the furniture from it, the layout stands the books between its sides, and
`shelves.ts` derives the overview frame, the hotspot anchor, the pick volume and
the row names from it, so none of those can fall behind the geometry on screen.
Both files are written to this directory on every run, whether or not Blender
is invoked. They are generated, so the formatter leaves them alone.

To update only printed spine artwork without rebaking the room:

```sh
BLENDER=/path/to/blender bun run room:bake --spines-only
```

To rebuild articulated book bodies and front covers without rebaking the room:

```sh
BLENDER=/path/to/blender bun run room:bake --books-only
```

Only use these options when book positions and dimensions have not changed.
`--layout-only` writes both manifests without invoking Blender; it is for
recovering them for matching assets, not for relocating existing books.

The room bookshelf uses the same scene, with frontal camera sections derived
from these slots. Its HTML links, records, notes, history, and fallback list live
in `src/components/home/room/Bookshelf.astro` and `bookshelf.ts`. Notes are
rendered from the content collection. Adding a note needs `--books-only` to
bake that volume a front cover; nothing else about a note requires a rebake.
`/#bookshelf` approaches the whole cabinet. Unfilled ivory dots mark its row
targets; selecting a row reaches reading distance. Books with notes show a persistent warm dot and tip on hover or keyboard focus.
Only those books open. Other spines expose metadata without an opening action.
A red cloth ribbon marks books currently being read, independently of notes. Clear background steps back one level;
the band between book spines is inactive so a near miss does not zoom out.
Previous/next-row controls and Up/Down switch rows. Left/Right controls and
horizontal swipes stay within the current row. Movement within a row and
between sibling rows replaces the current history entry, so Back returns to
the cabinet. Closing a book restores its row and focused spine.

Book URLs use `/#bookshelf/<row-anchor-isbn>/<book-isbn>`. The old `/bookshelf/`
archive remains available during the transition.

The poster matches the latest full rebake. The capture hides scene controls.

`books.glb` holds each book body as `Body_<isbn>`, with its origin at the bottom
front of the spine. The back cover, spine, and paper form one movable node per volume.
Each body is baked in isolation under the room lights so neighbouring books do
not leave black shadows on exposed pages. Printed jackets keep their separate
atlas. The runtime attaches each body and jacket to the same pivot.

`covers.glb` holds a `Cover_<isbn>` front cover for each **annotated** volume
only, since only those open: each has a hinge at its spine-side edge, leather
thickness, an ivory endpaper, and original gold title and author artwork from
`coverSvg` in `spine-art.mjs`. A temporary joined copy bakes them together,
excluding secondary rays between these flat covers; the exported originals
retain independent hinges. Baking all 55 divided the atlas among 200-odd faces
the reader can never reach and left the openable cover lettered at eight pixels
per centimetre; the sheet is now sized from the number of covers actually in
it, and one note is read at forty-four.

`bookmark.glb` holds one reusable Blender-authored textile bookmark. It has a
curved fold, a twisted and tapered tail, solid thickness, bevelled edges, and baked woven bump detail. Its
origin is the top of the book; the runtime attaches a copy to each currently
reading volume. Reading-status changes do not require a rebake.

Annotated books tilt six degrees and slide forward slightly on hover. Opening
runs a reversible 900ms sequence: clear the shelf, turn the front cover toward
the viewer, then open the cover. The book travels 32cm in total. The camera
retreats and centers the volume before the cover opens into a spread. Closing
fades the HTML for 100ms and reverses the physical sequence over 650ms.
Escape or history navigation can interrupt extraction. Closing returns the book
to its slot. Reduced motion uses the conventional notes dialog immediately. With motion,
the native dialog starts focus containment at once, while its HTML stays inert
and transparent until the cover reveals the pages. The renderer projects the
page bounds into the dialog, keeping notes selectable and links functional.
Each page scrolls independently for long content, including on phones.
Only annotated jackets and bodies receive the warm material tint.

Row views use a 40-degree lens and a slightly raised camera, clearing the foreground
lampshade. Cabinet and room retain the authored lens. Captions and row navigation
sit below the canvas; projected hit targets stay still while cloth ribbons follow
the moving book. The native notes dialog preserves content nodes and returns focus
to the same spine when closed.

The default bake uses 256 samples, 2048px shell/furniture atlases, 4096px
object and jacket atlases, a 2048px book-body atlas, a front-cover atlas that
doubles from 2048px per four covers up to 4096px, and 1024px atlases for the
turntable parts and bookmark. Vite fingerprints the GLBs; Astro generates the poster formats.
