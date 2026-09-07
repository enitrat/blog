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

The default bake uses 256 samples, 2048px shell/furniture atlases, a 4096px
object atlas and a 1024px atlas for the moving parts. Vite fingerprints the GLBs; Astro generates the poster formats.
