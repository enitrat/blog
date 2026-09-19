# Fast Blender-to-web GLB pipeline

Research checked 2026-09-19. Sources are Khronos, Blender, glTF Transform, and
Three.js documentation or source. The recommendations below are applied to this
repository's nine baked room GLBs, not general-purpose 3D production.

## What the current assets say

The repository already has a useful partition: `shell`, `furniture`, and
`objects` are static room structure; `moving` is pivoted; `spines`, `books`,
`covers`, `bookmark`, and `sheets` are content or interaction assets. The
[asset contract](../../src/assets/room/README.md) names those boundaries, and
the [room loader](../../src/components/home/room/scene.ts) currently fetches all
nine with `Promise.allSettled`.

`gltf-transform inspect` reports that all nine files use one baked JPEG
`baseColorTexture` and `KHR_materials_unlit`. The checked-in GLBs total about
9.7 MB, but the inspector's minimum GPU estimates are much larger: each 4096²
atlas is about 89.5 MB, each 2048² atlas about 22.4 MB, and each 1024² atlas
about 5.6 MB. This is an estimate, not a device measurement, but it identifies
texture memory, not mesh bytes, as the first optimization target.

## Partitioning: preserve the nine GLBs

glTF can describe a complete scene, but Khronos' asset-creation guidance says a
single `.glb` is generally preferred for simple delivery while separate files
can be more efficient when a configurator swaps or loads parts independently.
The glTF specification also permits external buffers and images, which enables
progressive or on-demand delivery when the runtime supports it.

For this room, keep the nine files because their boundaries match ownership,
interaction, and failure recovery. They also give the browser/CDN independent
cache keys. If first paint becomes slow, the smallest useful runtime change is
to load `shell`, `furniture`, and `objects` first, then defer shelf and desk
content until those views are entered. Do not merge everything merely to reduce
request count: the current payload is already small enough that the 4096²
textures dominate.

glTF Transform's [`partition()`](https://gltf-transform.dev/modules/functions/functions/partition)
splits binary payloads into multiple `.bin` buffers for engines that can lazy
load them. It does not create nine independently addressable GLBs, and it is
not useful unless the loader understands that loading model. Use it only if a
single logical asset needs lazy binary ranges; the existing nine-file contract
is the simpler chunking mechanism here.

## Atlas versus per-asset textures

Khronos' [`KHR_texture_transform`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_transform/README.md)
specification explicitly identifies atlases as a way to minimize the number of
textures the GPU loads, with each object's UV region represented by an offset,
rotation, and scale. That supports the current shared `spines` atlas: many tiny
printed jackets share one material and one texture, while the runtime slices
the mesh by ISBN.

An atlas is a good fit when objects are small, opaque, loaded together, and
read-only. It is a worse fit when one object changes often, needs independent
lazy loading, has a different resolution requirement, or would force a large
texture to stay resident. A per-asset texture makes those updates and loads
independent, but increases image/material bookkeeping and can increase draw
calls. For this room, retain the spine atlas and the baked group atlases; only
split a group after measuring a real interaction or memory problem. The
`covers` atlas is already sized from the number of note-bearing books, which is
the right direction for incremental content.

## JPEG, WebP, and KTX2

Blender's glTF manual describes JPEG as a reasonable smaller web output when
loss is acceptable, while PNG is lossless. JPEG has no alpha, so it is suitable
for these opaque, display-referred Cycles bake atlases but not transparent
artwork. Keep high-quality JPEG as the baseline and compare visual text edges
before lowering quality.

`EXT_texture_webp` is ratified and lets a glTF texture use WebP, optionally with
a PNG or JPEG fallback. Three.js lists `EXT_texture_webp` among the extensions
handled by `GLTFLoader`. WebP is a low-risk transfer-size experiment for these
atlases, but the Khronos KTX guide notes that WebP/JPEG are decompressed for GPU
use; they reduce download bytes, not texture memory.

KTX2 with `KHR_texture_basisu` is the memory-oriented option: Khronos says KTX
textures remain compressed in GPU memory and are transcoded to the hardware's
best format. Its rule of thumb is ETC1S for color data and UASTC for non-color
data; high-contrast baked lettering may justify testing UASTC even though the
atlas is color data. The extension requires 2D KTX2 images and recommends full
mip pyramids. For this repository, test KTX2 first on `objects`, `sheets`, and
`spines`, the three 4096² atlases, and compare text quality plus GPU memory.

The runtime cost is explicit: Three.js requires a configured `KTX2Loader` via
[`GLTFLoader.setKTX2Loader()`](https://threejs.org/docs/pages/GLTFLoader.html),
and the loader must be configured for the renderer's supported formats. Keep a
JPEG artifact until the browser check proves KTX2 is reliable across the
supported browsers.

## Meshopt versus Draco

`KHR_draco_mesh_compression` compresses triangle geometry. Meshopt's ratified
extension can compress point, line, and triangle geometry, morph targets, and
animation data. glTF Transform exposes both, but the runtime implications are
different:

| Choice | Three.js requirement | Fit for this room |
| --- | --- | --- |
| Meshopt | Register `MeshoptDecoder` with `GLTFLoader` | Preferable if geometry or animation later becomes large; current room has no animations and tiny mesh payloads relative to atlases. |
| Draco | Configure a `DRACOLoader` and decoder path | Useful for geometry-heavy assets, but adds another decoder path and does not address the 4096² texture cost. |

The current `spines` inspection is representative of the trade-off: its mesh
is roughly 5 KB while its JPEG is about 3.05 MB. Do not add either decoder to
the shipped path until an inspect report shows geometry is material. If testing,
produce a separate web artifact and load it through the corresponding Three.js
decoder before replacing the checked-in GLB.

## glTF Transform: inspect first, then select transforms

The official SDK describes itself as reproducible, low-level glTF control for
bundling, splitting, and optimizing existing models. The CLI provides the
smallest useful loop:

```sh
bunx @gltf-transform/cli inspect src/assets/room/spines.glb
bunx @gltf-transform/cli validate src/assets/room/spines.glb
```

`inspect` returns scene, mesh, material, texture, animation, and memory
statistics. `validate` checks the glTF file against the specification. The
CLI's `optimize` command combines many transforms, but its documentation warns
that defaults may not suit every scene; use individual commands for this
already-baked pipeline:

- `dedup` and `prune` for lossless cleanup;
- `webp`, `jpeg`, or `uastc`/`etc1s` for measured texture variants;
- `meshopt` or `draco` only when geometry warrants the decoder;
- `partition` only when the consuming loader can lazy-load its buffers.

The [SDK API](https://gltf-transform.dev/) also exposes `NodeIO`/`WebIO` and
transform functions, so a small script can inspect and write each GLB without
reimplementing glTF indexing or byte offsets. Every changed GLB should then be
inspected and loaded by the existing browser check.

## Incremental and content-addressed builds

There is no content-addressed cache built into Blender, glTF, Three.js, or
glTF Transform. The practical build-layer design is therefore an inference,
using their existing file-oriented tools:

1. Hash the source content inputs, bake settings, transform configuration, and
   Blender/glTF Transform versions.
2. Store the resulting nine GLBs and three manifests under that hash in a local
   cache; reuse unchanged outputs and copy them to the normal asset paths.
3. Keep separate variants (`jpeg`, `webp`, `ktx2`, and optionally `meshopt`) so
   changing delivery compression does not invalidate the Blender bake.
4. Run `inspect`, `validate`, and the existing Chromium room check only for
   outputs whose hash changed.

This preserves the current source-of-truth split: Blender performs the Cycles
bake, glTF Transform performs deterministic delivery transforms, and Vite
fingerprints the final files. A cache is worth adding when repeated bakes or
compression experiments become a measurable bottleneck; until then, the
existing scripts and nine-file contract are the smaller system.

## Primary sources

- [Khronos glTF 2.0 specification](https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc)
- [Khronos 3D asset-creation file-structure guidance](https://github.com/KhronosGroup/3DC-Asset-Creation/blob/main/asset-creation-guidelines/full-version/sec01_FileStructure/FileStructure.md)
- [Khronos `KHR_texture_transform`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_transform/README.md)
- [Khronos `KHR_texture_basisu`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_basisu/README.md)
- [Khronos KTX Artist Guide](https://github.com/KhronosGroup/3D-Formats-Guidelines/blob/main/KTXArtistGuide.md)
- [Khronos `EXT_texture_webp`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_texture_webp/README.md)
- [Khronos `KHR_draco_mesh_compression`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_draco_mesh_compression/README.md)
- [Blender glTF manual](https://docs.blender.org/manual/en/4.5/addons/import_export/scene_gltf2.html)
- [Blender glTF exporter API](https://docs.blender.org/api/current/bpy.ops.export_scene.html)
- [Three.js `GLTFLoader` docs](https://threejs.org/docs/pages/GLTFLoader.html)
- [glTF Transform SDK and CLI](https://gltf-transform.dev/)
- [glTF Transform `inspect`](https://gltf-transform.dev/modules/functions/functions/inspect), [`partition`](https://gltf-transform.dev/modules/functions/functions/partition), [`meshopt`](https://gltf-transform.dev/modules/functions/functions/meshopt), [`draco`](https://gltf-transform.dev/modules/functions/functions/draco), and [`textureCompress`](https://gltf-transform.dev/modules/functions/functions/textureCompress)

## Addendum: post-targeted-bake priorities (2026-09-19)

The targeted-bake and sheet-batching work is doing its job: the current
`sheets.glb` is 478,196 bytes versus 492,280 bytes in the previous checkout
(14,084 bytes, or 2.9%, smaller). It does not change the runtime bottleneck.
The nine current GLBs total about 9.6 MB; `inspect` still estimates roughly
268 MB of GPU allocation for the three 4096² atlases (`objects`, `sheets`,
`spines`), roughly 89 MB for four 2048² atlases, and roughly 11 MB for two
1024² atlases. The scene loader still starts all nine loads together and does
not create the renderer until every load settles.

Ranked next steps, with a measurable trigger:

1. **Measure cold first render and classify requests.** Add temporary
   `performance.mark()` points around the first successful render and the
   all-assets completion, then read each GLB's `PerformanceResourceTiming`
   `transferSize`, `encodedBodySize`, and `decodedBodySize`. The [Resource
   Timing standard](https://w3c.github.io/resource-timing/) defines those
   fields. Trigger: any slow connection where all-assets completion is later
   than the first room view, or where decoded bytes dominate. Benefit: gives a
   before/after number for every change below; no payload change by itself.

2. **Stage assets that are not needed for the first view.** Keep the current
   core (`shell`, `furniture`, `objects`, `moving`, `spines`, `books`, and
   `sheets`) together if the room view needs them, but load `bookmark` only
   when a reading bookmark can be shown and test loading `covers` on the first
   note-opening path. Trigger: browser tracing confirms the canvas waits on
   those conditional files. Immediate cold-start saving is about 0.58 MB
   transfer and 28 MB of inspector-estimated texture allocation; the exact
   first-render improvement depends on connection and decode time. Preserve
   the current fallback and per-phase cleanup behavior.

   **Experiment result:** deferring `covers` and `bookmark` improved the first
   completed frame by only about 60–100 ms locally while delaying those assets
   until roughly 2.1–2.3 seconds. The runtime change was rejected; the eager
   loader remains simpler and keeps those interactions immediately available.

3. **Run a KTX2 A/B artifact for the three 4096² atlases.** Khronos says KTX2
   Basis textures can remain compressed in GPU memory, and Three.js requires a
   configured `KTX2Loader` through `GLTFLoader.setKTX2Loader()`. Trigger: a
   device trace shows texture memory pressure, context loss, or high texture
   upload time. Expected benefit: the largest possible memory reduction in this
   asset set, plus a likely transfer reduction; measure the actual transcoded
   size, upload time, and text quality. Keep JPEG as the fallback artifact
   until the Chromium room check and supported-browser matrix pass.

4. **Test selective atlas downscaling before touching geometry.** Produce
   2048² variants of `objects` and `sheets` and compare a fixed browser
   screenshot at the closest reading views. A 4096² to 2048² change reduces
   the inspector's nominal allocation from 89.5 MB to 22.4 MB per atlas;
   whether that is acceptable depends on projected text size. Do not downscale
   `spines` without a close-shelf text comparison: its atlas is the deliberate
   readability exception.

5. **Try WebP only if transfer remains the measured bottleneck.** Three.js
   supports `EXT_texture_webp`, and glTF Transform can produce a WebP variant,
   but Khronos notes WebP/JPEG are decompressed for GPU use. Trigger: KTX2 is
   unsuitable for a target browser and Resource Timing shows network bytes,
   rather than decode or GPU memory, dominate. Expect download savings only.

Explicit skips for the current evidence: Draco, Meshopt, glTF `partition()`, a
new shared atlas, and a service-worker cache. The inspected geometry is tiny
relative to the baked textures; the current Three.js loader has no compression
decoders; `partition()` needs a loader that understands split buffers; and
fingerprinted URLs plus the browser HTTP cache already cover repeat requests.
Revisit those only when the measurements above show their specific cost.

The [Fetch Standard](https://fetch.spec.whatwg.org/) defines the browser cache
modes used by normal requests, including reuse of matching fresh HTTP-cache
responses; it does not remove the need to set appropriate CDN cache headers.
