# Living room assets

Original geometry, materials and print artwork are authored in `scripts/room.py`
and `scripts/bake-room.mjs`. The spines use Mathieu's `src/booksData.ts` library.
No models or textures from the reference websites are included.

The rendering approach follows Henry Heffernan's
[baked models](https://github.com/henryjeff/portfolio-website/blob/master/src/Application/Utils/BakedModel.ts):
glTF geometry with baked color and light, displayed using unlit materials.

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

To update only printed artwork without rebaking the room:

```sh
BLENDER=/path/to/blender bun run room:bake --spines-only
```

Only use that option when book positions and dimensions have not changed.
`--layout-only` writes both manifests without invoking Blender; it is for
recovering them for matching assets, not for relocating existing books.

The room bookshelf uses the same scene, with frontal camera sections derived
from these slots. Its HTML links, records, notes, history, and fallback list live
in `src/components/home/room/Bookshelf.astro` and `bookshelf.ts`. Notes are
rendered from the content collection; adding a note does not require a rebake.
`/#bookshelf` approaches the whole cabinet. Unfilled ivory dots mark its row
targets; selecting a row reaches reading distance. Individual books show the
same dot on hover or keyboard focus. Clear background steps back one level;
the band between book spines is inactive so a near miss does not zoom out.
Previous/next-row controls and Up/Down switch rows. Left/Right controls and
horizontal swipes stay within the current row. Movement within a row and
between sibling rows replaces the current history entry, so Back returns to
the cabinet. Closing a book restores its row and focused spine.

Book URLs use `/#bookshelf/<row-anchor-isbn>/<book-isbn>`. The old `/bookshelf/`
archive remains available during the transition.

The current poster predates the new print atlas. Regenerate it in the next
visual pass; this implementation pass intentionally did not capture the browser.

The default bake uses 256 samples, 2048px shell/furniture atlases, 4096px
object and jacket atlases, and a 1024px atlas for the moving parts. Vite fingerprints the GLBs; Astro generates the poster formats.
