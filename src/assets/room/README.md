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

`spines.glb` holds the printed jackets on a separate 4096px atlas. Source SVGs
are rasterized at 512×3072, without first rasterizing at a smaller size and
upscaling. The bake assigns each jacket a regular atlas cell with a 20px gutter.
For 56 books this is a 13×5 grid, about 275×779 usable pixels per jacket.
The planes are joined after UV packing for one bake and one runtime draw call.

`book-slots.json` records ISBN, shelf row, spine bounds, and position in Three.js
coordinates. `bake-room.mjs` generates these slots; Blender consumes them to
place the actual books and exports the manifest alongside the GLBs. Rebuild
the room after changing the library. The generator rejects a full cabinet
instead of silently omitting books. The page build rejects missing ISBNs.

To update only printed artwork without rebaking the room:

```sh
BLENDER=/path/to/blender bun run room:bake --spines-only
```

Only use that option when book positions and dimensions have not changed.
`--layout-only` writes the manifest without invoking Blender; it is for
recovering a missing manifest for matching assets, not relocating existing books.

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
