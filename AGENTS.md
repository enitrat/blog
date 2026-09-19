# Repository instructions

## Authored content

Preserve Mathieu's prose, claims, contact details, and artwork. Ask before
replacing authored content.

## Blender room

For changes to the room geometry, materials, lighting, camera, bake, or assets,
read `docs/room-workflow.md` and `src/assets/room/README.md` before editing.

Use Blender 4.5.3. Change the live scene through Blender MCP and `bpy`. Use
screenshots and Cycles renders to judge the result. Put accepted logic in
`scripts/room.py` or `scripts/bake-room.mjs`.

A room change is complete after the relevant headless bake, glTF Transform
inspection of each changed GLB, and `bun run room:check chromium` pass.
