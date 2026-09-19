# Blender process audit

Assessment date: 2026-09-19.

This audit combines the current repository, glTF inspection, and the local
Claude Code, Codex, and Cursor history for this project. The supporting web
research is in:

- [Incremental Blender builds](./blender-incremental-builds.md)
- [Efficient Blender MCP use](./blender-mcp-efficiency.md)
- [Web 3D asset pipeline](./web-3d-asset-pipeline.md)

## Verdict

The main direction is right. Live Blender and MCP are the inspection loop,
`room.py` is the durable source, headless Blender produces the checked-in
artifacts, and browser tests judge the delivered result.

The slow and fragile part was the build graph. Runtime assets had been split
into nine GLBs, but the generator still behaved like one large build. Partial
flags selected outputs late, after all content art and the whole Blender scene
had been constructed. Bake, material replacement, export, and publication also
happened in one process. A failure could leave source assets from different
generations in the repository.

The first refactor should make the existing boundaries real. It does not need
a new build framework, linked Blender libraries, or one Python file per prop.

## Implemented in this pass

The first build-side changes landed with the audit:

- `--only` accepts any comma-separated set of the nine existing output groups.
- Legacy partial flags still work through the same target parser.
- Artwork preparation skips unrelated images for the isolated books, covers,
  bookmark, and sheets targets. Cover art is generated only for noted books.
- The scripts report artwork, scene, per-group bake, export, validation, and
  total elapsed time.
- Manifests and GLBs build in the work directory. The command validates every
  selected GLB before replacing its checked-in counterpart.
- A partial bake fails before Blender starts if book slots or cabinet geometry
  differ from the checked-in manifests.

A low-sample `--only sheets` check completed in 29.75 seconds. It spent 16.4
seconds constructing the scene, 10.45 seconds baking the eight sheets, 0.32
seconds exporting, and 0.35 seconds validating. It published only `sheets.glb`
and `sheets.json` to its temporary output.

The production check took 257.1 seconds at 256 samples and a 4096px atlas.
Scene construction took 13.96 seconds and the eight serial sheet bakes took
241.17 seconds. The regenerated geometry and glTF JSON were byte-identical to
the checked-in asset. Decoded atlas channels had a mean absolute difference of
0.000052 on the 0 to 255 scale, with a maximum difference of 4 from Cycles and
JPEG output.

A production `--only covers` run took 37.73 seconds. The old `--books-only`
path used for cover changes took about 17 minutes because it baked all 55 book
bodies first. The regenerated cover geometry was byte-identical.

Caching and export-only resume remain deferred. Blender has no automatic
incremental Cycles invalidation, and the current static groups still have
implicit shadow and occlusion dependencies. Adding a cache before recording
those dependencies would trade elapsed time for stale assets.

## Current system

```text
books and writing content
        |
        v
bake-room.mjs
  layouts + manifests + selected source artwork
        |
        v
room.py
  whole scene -> selected Cycles bakes -> staged GLB exports
        |
        v
glTF validation -> publish selected GLBs and manifests
        |
        v
poster capture -> browser checks -> production build
```

What is working:

- Blender is pinned to 4.5.3 and the script fails on a different version.
- Random scene variation has a fixed seed.
- Movable books, covers, the bookmark, and manuscripts have separate runtime
  nodes and deliberate bake isolation.
- The browser has a poster fallback, real DOM controls, on-demand rendering,
  resource disposal, and Chromium/WebKit checks.
- MCP experiments do not replace the checked-in Python source.

What is not yet reproducible:

- Manuscript and spine artwork depend on fonts installed on the bake machine.
  `sheet-art.mjs` explicitly uses macOS fonts and falls back elsewhere.
- The MCP client is registered, but every live session still depends on opening
  Blender and starting the addon. It was not running during this audit.
- The configured global `blender-mcp` command now identifies itself as a
  compatibility wrapper. The repository setup documentation now uses
  `mcp-for-blender`; changing the global entry can wait until the next install.

## Evidence from the project history

The history shows the same failure pattern several times:

- A full bake took about 25 minutes. Fifty-five isolated book bodies consumed
  21 minutes.
- `--room-only` reduced a room change to about five minutes, but sheet-only
  fixes still paid that five-minute cost.
- A long session ended after the book texture finished but before export. The
  repository had a final cover GLB, a stale book GLB, and a completed atlas in
  `/tmp`.
- The recovery script skipped baking and exported the completed work in about
  one minute. Its cover output was byte-identical. This proves that bake and
  export are useful recovery boundaries, but the normal command does not
  expose them.
- One checked-in `books.glb` contained 55 nodes and no texture because it had
  been exported before the corresponding bake completed.
- The manuscript bug needed repeated room bakes. The first geometry change did
  not fix the actual atlas occlusion, and the browser check initially missed
  the user-visible state.
- Cover quality improved only after the build stopped spending one atlas on 55
  covers when one cover was reachable. The resulting GLB fell from 1.76 MB to
  439 KB and texel density rose from 8 to 44 pixels per centimetre.

These are build-system failures, not reasons to replace Blender.

## Findings in priority order

### 1. Failed builds can publish partial output

`bake-room.mjs` writes the three checked-in manifests before Blender starts.
`room.py` then exports each selected GLB directly into `src/assets/room`.
There is no staging validation or final publish step.

Build into the work directory first. Inspect every staged GLB, then replace the
selected checked-in outputs only after the Blender process exits successfully.
Keep the old outputs untouched on failure.

### 2. Partial bake flags are too coarse

The current selectors are:

| Selector | Outputs |
|---|---|
| full | all nine groups |
| `--room-only` | shell, furniture, objects, moving, spines, sheets |
| `--books-only` | books, covers |
| `--spines-only` | spines |
| `--layout-only` | manifests |

There is no direct target for sheets, covers, bookmark, furniture, or another
single group. Adding a note rebuilds all 55 book bodies even though the body
generator does not depend on note presence. Publishing one writing rebuilds
five unrelated GLBs.

Replace the special cases with one `--only` group list. Keep the friendly npm
commands if they remain useful, but make them aliases over the same selector.

### 3. Selection happens after most preparation work

`bake-room.mjs` rasterizes every spine, every cover, every manuscript, and both
prop images before it passes the partial flags to Blender. It even renders
cover artwork for books that cannot produce a cover node. `room.py` constructs
and converts the entire scene before it skips bake groups.

Filter source-art generation by target first. Then make the isolated groups
buildable without constructing unrelated room geometry. Do not split the
static room blindly because its baked shadows and occlusion couple several
groups on purpose.

### 4. Bake and export cannot resume cleanly

The script saves `living-room.blend` before baking, but does not save the
post-bake material state. A later process cannot open that file and export the
finished asset. Recovery required an ad hoc Python script.

Keep a stable work directory for a run and save after each expensive group
bake. A small run manifest should record the Blender version, settings, input
hash, finished groups, and output names. An export-only resume should refuse a
mismatched manifest.

### 5. Asset partitioning does not reduce initial browser cost

The nine GLBs total 9.7 MB. glTF Transform reports two 1024 atlases, four 2048
atlases, and three 4096 atlases. With mipmaps, those textures occupy about
352 MiB before browser and driver overhead. `scene.ts` loads all nine GLBs with
one `Promise.allSettled` call.

The existing split is useful for ownership and animation, but it is not a
loading strategy. Geometry compression will not solve a texture-dominated
budget. Test GPU-compressed KTX2 against the current JPEG output, and consider
loading the hidden cover and bookmark only when a reader opens a noted book.
Keep the current path until visual quality, Safari behavior, and browser checks
pass with the decoder.

### 6. Dependencies are implicit

The output groups look independent, but their bake inputs differ:

- Static groups can receive shadows and occlusion from the complete room.
- Books, bookmark, and sheets deliberately hide other meshes while baking.
- Covers use a temporary joined copy and retain separate hinges for export.
- Lights and color-management settings affect every baked group.
- Book layout affects manifests, bodies, covers, spines, and static room
  occlusion.

Write this mapping as data before adding a cache. Hash-based skipping is safe
only when the dependency list is explicit. Blender does not provide automatic
incremental Cycles bake invalidation.

### 7. Verification is strong but manually assembled

The browser check is valuable, and the asset reference says to inspect every
changed GLB. The bake command does not perform that inspection or report which
outputs changed. Poster capture and browser checks are separate manual steps.

The publish command should print the changed targets and run glTF inspection
for those targets. Keep poster capture and browser checks explicit because not
every isolated content bake changes the poster.

## Recommended refactor sequence

1. Add elapsed timing per preparation, scene generation, bake group, export,
   inspection, poster, and browser-check phase. Keep one machine as the
   comparison baseline.
2. Stage outputs outside `src/assets/room`, inspect them, and publish only on a
   successful run.
3. Add a general `--only` selector. The first useful targets are `sheets` and
   `covers`; they remove the repeated five-minute and 21-minute penalties seen
   in the history.
4. Split `room.py` into functions for scene setup, target construction, bake,
   and export. Keep them in one file until a second caller or independently
   authored asset makes another module useful.
5. Save resumable post-bake state and a small run manifest. This turns the
   proven one-minute recovery path into a supported command.
6. After the target dependencies are explicit, add content hashes to skip
   outputs whose inputs and settings have not changed.
7. Benchmark a joined or spatially separated batch bake for isolated books and
   sheets. Accept it only if the resulting atlases match the current isolation
   visually.
8. Treat GPU texture compression and lazy runtime loading as a separate web
   delivery pass. They solve memory and transfer cost, not authoring speed.

## What not to build

- No custom task runner yet. The Node standard library can hash files, stage a
  directory, spawn Blender, and rename validated outputs.
- No Blender library overrides for one generated scene. They add resync and
  hierarchy failure modes without shortening the known expensive bakes.
- No module per chair, lamp, or prop. Split at rebuild and ownership boundaries,
  not at object names.
- No MCP-driven production bake. MCP is valuable for short inspection and
  correction loops. Long, logged, resumable work belongs in the headless CLI.
