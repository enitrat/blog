# Why the room uses Blender MCP and a headless bake

Research checked on 2026-09-11 supports a two-part Blender workflow. Live
Blender gives the agent fast visual feedback. Headless Blender produces the
repeatable assets that enter the repository.

## Tool roles

| Tool | Use | Do not use it for |
|---|---|---|
| Blender MCP and `bpy` | Inspect and change a live scene | The final source of truth |
| Viewport screenshots | Judge shape, placement, and broad composition | Controlling Blender |
| Cycles renders | Judge lighting, materials, camera, and bake output | Replacing the scripted bake |
| `blender --background --python` | Rebuild GLBs from checked-in source | Fast visual exploration |
| glTF Transform | Inspect web artifacts and measure optimization options | Authoring the scene |
| Three.js browser checks | Verify the shipped files in their runtime | Authoring or baking assets |

Computer use is a fallback for addon controls with no API. It is too slow and
brittle for normal scene edits.

## Use real Blender

The MCP server does not replace Blender. It starts a local process that talks
to an addon inside the running Blender application. The addon exposes scene
inspection, Python execution, and viewport screenshots.

A standalone Python `bpy` package does not provide the complete application,
exporter, addon, render, and viewport environment used by this project. The
headless command still runs the real Blender binary.

## Keep one durable source

MCP edits change the open `.blend` file. Those edits are useful for exploration
but are hard to review and reproduce. The accepted logic belongs in
`scripts/room.py` or `scripts/bake-room.mjs`.

The manuscript bake bug shows why this boundary matters. A live session can
reveal the black occlusion. The durable fix still belongs in `room.py`, where
each sheet now bakes with the other sheets hidden.

## Inspect before compression

The current exporter writes unlit GLBs with baked JPEG atlases. glTF Transform
can report mesh, texture, extension, and memory details without changing the
file:

```sh
bunx @gltf-transform/cli inspect src/assets/room/shell.glb
```

Draco, Meshopt, WebP, and KTX2 can reduce some assets, but each choice adds a
runtime requirement. The current Three.js loader does not configure Draco or
Meshopt decoders. Measure first, add the required decoder, write a separate web
artifact, and run the browser checks before replacing a checked-in GLB.

## Sources

- [Blender MCP README](https://github.com/ahujasid/blender-mcp)
- [Blender MCP server](https://github.com/ahujasid/blender-mcp/blob/main/src/blender_mcp/server.py)
- [Blender MCP addon](https://github.com/ahujasid/blender-mcp/blob/main/addon.py)
- [Blender glTF manual](https://docs.blender.org/manual/en/4.4/addons/import_export/scene_gltf2.html)
- [Blender glTF exporter API](https://docs.blender.org/api/main/bpy.ops.export_scene.html)
- [glTF Transform documentation](https://gltf-transform.dev/)

Follow the [room workflow](../room-workflow.md) for commands. Use the
[asset reference](../../src/assets/room/README.md) for file contracts.
