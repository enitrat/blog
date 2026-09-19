# Incremental Blender builds for the room bake

Research checked on 2026-09-19 against Blender 4.5 LTS documentation and
developer documentation. The repository facts below come from
[`scripts/bake-room.mjs`](../../scripts/bake-room.mjs) and
[`scripts/room.py`](../../scripts/room.py).

## Current shape of the pipeline

`bake-room.mjs` starts Blender as
`--background --factory-startup --python scripts/room.py`. `room.py` builds all
room groups and materials, saves `living-room.blend` **before** baking, then
bakes selected groups and exports their GLBs in the same process. The existing
`--room-only`, `--books-only`, and `--spines-only` options skip bake/export
groups, but they do not skip procedural scene construction or reuse a prior
bake. That explains why the current switches reduce the Cycles work but do not
make generation fully incremental.

## Facts from Blender 4.5

### `.blend` libraries and overrides

- Link keeps a reference to data in another `.blend`; changes in the source are
  reflected after the referencing file is reloaded. Linked data is initially
  read-only. Append copies data and keeps it local, so later source changes do
  not flow through. [Link & Append](https://docs.blender.org/manual/en/4.5/files/linked_libraries/link_append.html)
- Library Overrides allow edits to linked data while retaining synchronization,
  but they rely on a correct collection hierarchy and can require resyncing.
  Blender documents known limitations, and says resync can be costly and can
  delete overrides when the source hierarchy changes. [Library
  Overrides](https://docs.blender.org/manual/en/4.5/files/linked_libraries/library_overrides.html)
- Linking is not a bake cache. Scene-level settings are not copied when
  linking objects, and compressed libraries may be loaded in their entirety
  even when only a small part is linked. [Link & Append limitations](https://docs.blender.org/manual/en/4.5/files/linked_libraries/link_append.html#known-limitations)

**Recommendation:** do not introduce library overrides for this room yet. The
source is one deterministic Python generator and the desired boundaries are
already nine browser GLBs. A plain generated base `.blend` plus external bake
images is a smaller, inspectable seam. Revisit libraries only if independently
authored Blender files or multiple scenes actually appear.

### Collections and view layers

Blender lets each view layer change collection visibility. Excluded collections
are not rendered, and separate view layers can be rendered separately; Blender
explicitly notes that this can avoid rerendering an entire image after a layer
change. Cycles also has per-view-layer sample overrides. [Layers and
Passes](https://docs.blender.org/manual/en/4.5/render/layers/introduction.html)

This is useful organization and an explicit visibility mechanism, but it is not
an incremental procedural build system: it does not stop `room.py` from
constructing objects, and a bake still depends on which objects are visible to
the active scene/view layer. The current script’s hide-render isolation for
books, covers, bookmarks, and sheets is part of the visual contract and must be
reproduced and checked if collections/view layers replace it.

### Background jobs and separate stages

`--background` runs Blender without the UI. Blender’s command-line parser
preserves argument order, supports `--python`, and passes arguments after `--`
to the Python script. A saved `.blend` can therefore be an input to a later
headless bake or export job. [Command Line
Arguments](https://docs.blender.org/manual/en/4.5/advanced/command_line/arguments.html)

The Python API exposes opening and saving main files, and the bake operator
accepts selected objects, an active bake target, image dimensions/margins, and
`use_clear`. [Window operators](https://docs.blender.org/api/current/bpy.ops.wm.html)
and [object operators](https://docs.blender.org/api/current/bpy.ops.object.html#bpy-ops-object-bake)
are the relevant APIs.

Render baking requires a mesh UV map and an active Image Texture or Color
Attribute target. Cycles uses the scene’s render settings (including samples
and bounces) for baking. [Render Baking](https://docs.blender.org/manual/en/4.5/render/cycles/baking.html)

**Safe staged shape:**

1. Generate and save a base scene containing geometry, materials, lights,
   collections, UVs, and the exact bake settings.
2. Open that scene in a headless bake job, apply the same per-group visibility,
   selection, active-image, and `use_clear` rules, and save each baked image
   externally (or save a post-bake `.blend`).
3. Export GLBs in a later job from the post-bake scene/material state.

The current pre-bake `living-room.blend` cannot by itself support export-only:
it has no baked material assignment yet, and its image paths point into the
temporary work directory. If stages are added, make the work directory stable
for the run, save after bake/material replacement, and record Blender version,
scene/settings hash, input/art hash, and output names in a small manifest.

### Persistent data

Cycles’ **Persistent Data** keeps render data in memory after rendering so
rerenders and animation renders can be faster, at the cost of memory. With
multiple view layers, Blender preserves only one layer’s data (while preserving
objects shared between layers). [Cycles Performance](https://docs.blender.org/manual/en/4.5/render/cycles/render_settings/performance.html)

This is process-local and documented for renders, not a disk-backed bake cache.
It cannot help across the current `bake-room.mjs` process boundary, and it is
not evidence that a later process can reuse a Cycles bake. It may be worth
benchmarking in one long-lived Blender process that performs several bakes or
preview renders, but do not make output correctness depend on it.

### Shader compilation and device choice

Cycles optimizes shader nodes at compile time and run time. Blender’s developer
caching notes say compiled machine code is hardware-dependent and that compile
caches can have long initial fill times; they are not a portable project cache.
[Cycles shader nodes](https://docs.blender.org/manual/en/4.5/render/cycles/optimizations/nodes.html)
and [Blender caching analysis](https://developer.blender.org/docs/features/nodes/proposals/caching/)
are the relevant primary sources. The 4.5 GPU notes also say Metal kernels are
compiled at runtime, with no extra build step. [Cycles GPU
Binaries](https://developer.blender.org/docs/handbook/building_blender/cycles_gpu_binaries/#apple)

GPU rendering can be faster, but support is backend/platform-specific and GPU
memory limits can force a performance penalty. [GPU Rendering](https://docs.blender.org/manual/en/4.5/render/cycles/gpu_rendering.html)
The current script hard-codes Metal when available and otherwise falls back to
CPU; retain that fallback and benchmark on the actual bake machine rather than
assuming a device switch is an incremental-build feature.

### Adaptive sampling

Adaptive sampling stops sampling pixels that have crossed the noise threshold,
so it can reduce work in already-clean areas. Lower thresholds improve quality
at the cost of time, and Cycles uses the same render sampling settings for
baking. [Cycles Sampling](https://docs.blender.org/manual/en/4.5/render/cycles/render_settings/sampling.html)
and [Render Baking](https://docs.blender.org/manual/en/4.5/render/cycles/baking.html)

The room already enables adaptive sampling at `0.025`. It is a useful measured
quality/time knob, not a cache or dependency tracker; changing it invalidates
the baked pixels for every affected group.

## Recommendation, in order

1. Add content-addressed skip logic around the existing group outputs first:
   hash the relevant generator/art/layout/settings inputs per group, and reuse a
   GLB only when its manifest matches. This gives incremental builds without
   changing Blender’s scene semantics.
2. If scene construction is a meaningful share of the remaining time, save a
   stable generated base `.blend`, then run bake and export stages against it.
   Keep the current isolation rules and validate every changed GLB with the
   existing glTF/browser checks.
3. Keep one Blender process for a batch of group bakes or previews where
   practical; this is the only place Persistent Data and runtime compilation
   reuse could plausibly amortize work, and it needs timing evidence.
4. Use collections/view layers for explicit visibility and maintainability
   only after a visual comparison. Do not add library linking/overrides unless
   the project gains genuinely independent Blender-authored libraries.

There is no official Blender 4.5 guarantee of an incremental Cycles bake cache
that makes a changed object’s pixels reusable automatically. Treat bake images,
post-bake `.blend` files, and GLBs as ordinary pipeline artifacts with explicit
input hashes and invalidation rules.
