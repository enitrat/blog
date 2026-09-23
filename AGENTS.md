# Repository instructions

## Authored content

Preserve Mathieu's prose, claims, contact details, and artwork. Ask before
replacing authored content.

## Blender room

For changes to the room geometry, materials, lighting, camera, bake, or assets,
read `docs/room-workflow.md` and `src/assets/room/README.md` before editing.

Use Blender 4.5.3. Use Blender MCP and `bpy` for short live inspection,
experiments, screenshots, and Cycles comparisons. Run long and final bakes
through `bun run room:bake`. Put accepted logic in `scripts/room.py` or
`scripts/bake-room.mjs`.

Use the smallest valid bake target documented in `docs/room-workflow.md`.
Publish generated assets through `bun run room:bake`, which stages and
validates outputs before replacing repository assets.

For bake-pipeline experiments, use `--no-publish`. For room-loading or asset
changes, compare `bun run room:check chromium --metrics` before and after. Keep
an optimization only when it measurably improves the result and room checks
still pass.

A visual room change is complete after its production-quality targeted bake,
glTF Transform inspection of each changed GLB, and browser pass. Run both checks
with `bun run room:verify --only <comma-separated changed targets>`.
