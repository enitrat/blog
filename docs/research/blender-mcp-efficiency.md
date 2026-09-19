# Blender MCP efficiency notes

Checked 2026-09-19 against upstream `main` and the current Codex MCP docs.
This is a workflow note, not a recommendation to make the live `.blend` the
project source of truth.

## Current upstream shape

The upstream project now calls itself **MCP for Blender** (`mcp-for-blender`).
Its README says existing `blender-mcp` / `uvx blender-mcp` setups continue to
work; `pyproject.toml` on `main` reports version `2.0.0`. The GitHub repository
currently exposes no latest release or tag, so record the package/addon versions
and use the add-on handshake (`get_addon_status`) when diagnosing drift. Sources:
[README](https://github.com/ahujasid/mcp-for-blender/blob/main/README.md),
[pyproject.toml](https://github.com/ahujasid/mcp-for-blender/blob/main/pyproject.toml),
[releases](https://github.com/ahujasid/mcp-for-blender/releases),
[tags](https://github.com/ahujasid/mcp-for-blender/tags).

The supported local shape is a Codex stdio MCP process plus a Blender add-on
socket server. Upstream's Codex setup is `codex mcp add blender -- uvx
mcp-for-blender`; run one MCP server instance, then start the add-on from the
Blender 3D View sidebar. [Upstream quickstart](https://github.com/ahujasid/mcp-for-blender/blob/main/README.md#quickstart)

## What the live bridge actually provides

- `get_scene_info` returns scene name, object/material counts, and only the
  first ten objects with name, type, and location. Use `get_object_info` or the
  world snapshot for targeted detail. [addon.py](https://github.com/ahujasid/mcp-for-blender/blob/main/addon.py#L984-L1016)
- `execute_blender_code` runs arbitrary Python in Blender, captures stdout, and
  returns a traceback on failure. There is no dedicated Cycles-render MCP tool;
  rendering is done through `bpy.ops.render.*` inside this tool. [server.py](https://github.com/ahujasid/mcp-for-blender/blob/main/src/blender_mcp/server.py#L631-L670),
  [addon.py](https://github.com/ahujasid/mcp-for-blender/blob/main/addon.py#L1490-L1516)
- `get_viewport_screenshot` writes a temporary PNG, captures an off-screen 3D
  viewport when possible (window-grab fallback), and returns it as an MCP image.
  The server default is a 1000-pixel largest dimension. This is viewport
  evidence, not a camera/Cycles still. [server.py](https://github.com/ahujasid/mcp-for-blender/blob/main/src/blender_mcp/server.py#L536-L628),
  [addon.py](https://github.com/ahujasid/mcp-for-blender/blob/main/addon.py#L1397-L1488)
- Current server instructions encode the useful loop: inspect add-on/scene,
  make a change, then take a screenshot and inspect scene info again. They also
  recommend node/API introspection before guessing sockets or enum values.
  [server instructions](https://github.com/ahujasid/mcp-for-blender/blob/main/src/blender_mcp/server.py#L297-L327)

## Limits that affect iteration

| Layer | Current behavior | Practical consequence |
|---|---|---|
| Codex MCP | Startup timeout defaults to 10 s; per-tool timeout defaults to 60 s. `tool_timeout_sec` overrides the latter. Per-tool `output_token_limit` is available and has a standard 20% serialization allowance. [MCP docs](https://developers.openai.com/codex/mcp#configure-with-configtoml), [config reference](https://developers.openai.com/codex/config-reference#mcp_serversidtool_timeout_sec) | Break edits and renders into short calls; raise the timeout only for a known long interactive render. |
| MCP ↔ Blender socket | Commands and responses are JSON over TCP; reads use 8192-byte chunks and the MCP client waits up to 180 s. A lock serializes send/receive on the persistent connection. There is no declared maximum JSON payload in this protocol. [README protocol](https://github.com/ahujasid/mcp-for-blender/blob/main/README.md#communication-protocol), [server.py](https://github.com/ahujasid/mcp-for-blender/blob/main/src/blender_mcp/server.py#L89-L225) | Do not run overlapping calls on one server; return compact summaries rather than dumping meshes or pixels as text. |
| Blender add-on | Commands are queued from socket threads and drained on Blender's main thread every 0.05 s. The add-on refuses `blender -b` background mode because its main-thread/event-loop work would not execute. [addon.py](https://github.com/ahujasid/mcp-for-blender/blob/main/addon.py#L645-L680), [addon.py](https://github.com/ahujasid/mcp-for-blender/blob/main/addon.py#L772-L800) | Use a GUI Blender session for MCP; use the repository's headless Blender process for final baking. |
| Tool-specific/state capture | `get_scene_info` lists 10 objects; trajectory snapshots cap at 4,000 objects and 1,000 selected names, with a 4.5 MB snapshot budget and a 400,000-byte observation budget. [trajectory.py](https://github.com/ahujasid/mcp-for-blender/blob/main/src/blender_mcp/trajectory.py#L55-L97), [addon.py](https://github.com/ahujasid/mcp-for-blender/blob/main/addon.py#L1204-L1267) | Query only the objects relevant to the change; do not mistake a truncated observation for a truncated scene. |

## Security, lifecycle, and telemetry

The MCP server keeps one persistent socket connection, performs a one-time
add-on handshake, reuses the connection without a probe on every call, and
disconnects it on shutdown. The add-on's socket has no authentication or
encryption; keep it on localhost and run only one MCP server instance. [README security](https://github.com/ahujasid/mcp-for-blender/blob/main/README.md#environment-variables),
[server lifecycle](https://github.com/ahujasid/mcp-for-blender/blob/main/src/blender_mcp/server.py#L89-L225),
[server connection reuse](https://github.com/ahujasid/mcp-for-blender/blob/main/src/blender_mcp/server.py#L359-L381)

`BLENDER_MCP_SAFE_MODE=1` adds a deny-by-default AST check before code crosses
the socket. It blocks raw file/process/network access, interpreter escapes,
handlers/timers/drivers, persistent class/property registration, and external
`.blend` datablock loading, while allowing normal Blender modeling, rendering,
saving, import, and export. It protects the MCP path only: another local
process that can reach the unauthenticated add-on socket can still send raw
commands. [README safe mode](https://github.com/ahujasid/mcp-for-blender/blob/main/README.md#safe-mode),
[safe_mode.py](https://github.com/ahujasid/mcp-for-blender/blob/main/src/blender_mcp/safe_mode.py)

Telemetry consent is enabled by default in the add-on. With consent, upstream
documents collection of prompts, generated code, scene metadata, viewport
screenshots, trajectory data, and manual operator/undo signals; without consent,
the server retains only minimal anonymous usage fields. Disable it in Blender or
set `DISABLE_TELEMETRY=true` (the source also accepts
`BLENDER_MCP_DISABLE_TELEMETRY` and `MCP_DISABLE_TELEMETRY`). [Terms and privacy](https://github.com/ahujasid/mcp-for-blender/blob/main/TERMS_AND_CONDITIONS.md),
[README telemetry](https://github.com/ahujasid/mcp-for-blender/blob/main/README.md#telemetry-control),
[telemetry.py](https://github.com/ahujasid/mcp-for-blender/blob/main/src/blender_mcp/telemetry.py)

## Efficient project workflow

1. In the GUI session, call `get_addon_status`, `get_scene_info`, and targeted
   object/API introspection before editing.
2. Use one small `execute_blender_code` mutation at a time. Print only the
   values needed to verify it; use a viewport screenshot for shape/placement,
   and a Cycles still for camera, lighting, material, or bake decisions.
3. Serialize calls and keep each interactive operation below the Codex timeout.
   If a render or export is predictably longer, configure that server's
   `tool_timeout_sec`; do not make every call long by default.
4. Once the result is accepted, copy the logic into [scripts/room.py](../../scripts/room.py)
   or [scripts/bake-room.mjs](../../scripts/bake-room.mjs). Run the required
   headless bake, inspect each changed GLB, and run the browser check. The live
   `.blend` is a disposable preview; checked-in scripts and baked assets are the
   reproducible source of truth. [room workflow](../room-workflow.md), [asset reference](../../src/assets/room/README.md)

In short: MCP is for fast inspection, small reversible experiments, and visual
feedback. Checked-in `bpy`/headless scripts are for deterministic geometry,
materials, camera, baking, export, and anything that must survive a session or
fit reliably in CI.
