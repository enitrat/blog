# Bake the living room from code

The homepage's living room has a bookcase, a record player, and a writing
desk. It must look rendered, load on an ordinary laptop, and be rebuildable by
anyone, or any agent, from the repository.

```mermaid
flowchart LR
	C["Books, notes, posts"] --> L["bake-room.mjs: layout + artwork"]
	L --> P["room.py in headless Blender"]
	P --> B["Cycles bake, one group at a time"]
	B --> V["Validate each GLB"]
	V --> R["src/assets/room"]
	R --> T["scene.ts in Three.js"]
```

## Light is computed once, in Blender

Cycles bakes each surface's color, shadows, and bounce light into textures.
The browser draws them unlit, with no lights, shadows, or tone mapping.
Real-time lighting would cost more and still look flat. The price is fixed
light. Moving a lamp means baking again.

## The scene is a script

`scripts/room.py` builds the room with Blender's Python API: geometry,
materials, lights, and camera. Headless Blender runs it, bakes, and exports.
A `.blend` file is only a disposable preview.

The script builds meshes with `bmesh`, not `bpy.ops` operators. Every operator
call refreshes the whole scene, and 2,232 of them took 32 seconds. With
`bmesh` the build takes about 2.

Bruno Simon's *My Room in 3D* and Henry Heffernan's portfolio were baked by
hand from `.blend` files that were never committed. Neither can rebuild its
own assets. Any machine with Blender 4.5.3 can rebuild ours.

## One bake group per file

The room ships as nine GLBs. Each is one bake group, meshes baked into one
shared atlas. Groups exist for baking, not loading, and differ in two ways:

- **What they see while baking.** Walls and furniture bake inside the full
  room and catch its shadows. Books, covers, the bookmark, and manuscripts
  move, so they bake with everything else hidden. Otherwise a pulled book
  would keep its shelf's shadow.
- **Atlas size.** Readable type, like spines and manuscripts, gets 4096
  pixels. The bookmark gets 1024.

A change rebakes only the groups it touches. A new note rebakes only the
covers, in about 40 seconds.

Cycles denoises the whole target image after every bake, so baking 55 books
straight into the 2048-pixel atlas denoised it 55 times. Now each body bakes
alone into a small image, copied into its own atlas cell. A 32-sample books bake fell from 457 to 29 seconds,
and each book got about twice the texture pixels.

The browser downloads all nine files at start. Deferring the covers, bookmark,
and manuscripts made the first frame only 60 to 100 ms sooner, so the simpler
eager load stayed.

## A refactor must show it changed nothing

`--no-publish` prints a `DIFF` line per staged atlas: its mean channel
difference from the checked-in one, and the share of channels off by more
than 16. At production samples, a refactor that changes no geometry or light
shows only Cycles noise.

## Only validated files reach the repository

Each bake writes into its own temporary work directory. `bake-room.mjs` runs
the glTF validator on every GLB and copies them into `src/assets/room/` only
after all pass. A failed bake leaves the checked-in assets alone, and git
undoes a bad publish.

`bun run build` runs `check-room-assets.mjs`. It fails if a book, note, or
post has no baked node, or if the GLBs exceed 11 MB. Without it, a note
written after the last bake would ship with no cover and no warning.

## What we chose not to do

| Option | Why not |
|---|---|
| Meshopt on every group | Meshopt cut `shell`, `furniture`, and `objects` from 3.24 to 1.83 MB gzipped. Its quantization moves each node's origin, and the browser turns, lifts, or places the other groups by their origins. glTF Transform's `optimize` and default `prune` are out too. Both delete nodes and names the browser looks up. |
| KTX2 GPU textures | Each of the three 4096-pixel atlases takes about 90 MB of GPU memory. A 2048-pixel atlas takes a quarter, so try smaller atlases first. Revisit KTX2 if a real phone runs short. |
| Linked `.blend` libraries | A second source of truth. One script rebuilds the scene in about 2 seconds. |
| Parallel group bakes | Cycles on one Apple GPU gains little. Each run has its own work directory, so separate terminals, agents, or worktrees still bake without clashing. |

## Evidence

| Measure | Result |
|---|---|
| Full bake at 256 samples | About 6 minutes, down from 25 before `bmesh` and per-book cells |
| Nine GLBs | 7.84 MB, down from 10.46 before Meshopt and per-book cells |
| Browser frames, new against old pipeline | Mean difference 0.6 to 0.9 of 255, at overview and shelf distance |
| Two bakes of one input | Byte-identical geometry, textures within a mean of 0.00005 of 255 |
