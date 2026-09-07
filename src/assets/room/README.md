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
The default bake uses 256 samples, 2048px shell/furniture atlases and a 4096px
object atlas. Vite fingerprints the GLBs; Astro generates the poster formats.
