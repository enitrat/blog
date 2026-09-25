# Change the living room

[ADR 0002](./adr/0002-room-bake-pipeline.md) explains why the pipeline works
this way.

## Patterns

- Edit the scripts, never the GLBs. `scripts/room.py` builds and bakes the
  scene. `scripts/bake-room.mjs` lays out and draws books and manuscripts,
  then runs Blender.
- Bake only the groups you changed. A full bake takes about 6 minutes, a
  covers bake about 40 seconds.
- For a refactor, bake with `--no-publish` and read its `DIFF` lines. At
  production samples, expect only Cycles noise. `--samples` and `--size` work
  only with `--no-publish` or `--preview`.
- For a loading or asset change, compare `bun run room:check chromium
  --metrics` before and after. Keep only measurable gains.

## Pitfalls

- **Stale MCP scene.** Check that MCP reports Blender 4.5.3 and has this
  run's `.blend` open. A preview writes `living-room.blend`. A bake of only
  `books`, `covers`, `bookmark`, or `sheets` writes `living-room-<groups>.blend`.
- **glTF Transform `optimize` or default `prune`.** Never run them. Both delete nodes
  and names that `scene.ts` looks up.
- **Meshopt moves node origins.** Keep it on `shell`, `furniture`, and
  `objects`. The browser moves the other groups by their origins.
- **No `timeout` on macOS.** The checks carry their own limits. `room:check`
  stops after 10 minutes.
- **WebKit relaunch race.** WebKit relaunched after a WebGL session can exit
  mid-boot and hang Playwright. `room:check` runs each engine in its own
  process, with one browser per engine.
- **MCP timeouts.** MCP calls time out before a bake ends. Use
  `bun run room:bake`.

## Preview a change

1. Install Blender 4.5.3. The bake refuses other versions. Set
   `BLENDER=/Applications/Blender.app/Contents/MacOS/Blender`.
2. Edit `scripts/room.py`.
3. Run `bun run room:bake --preview --samples 32`.
4. Open `preview.png` in the work directory it prints.

Each run gets a new work directory, unless you set `ROOM_WORK=/tmp/room`.
For close-ups or many small edits, open the `.blend` through
[Blender MCP](#set-up-blender-mcp) and copy what works into `room.py`.

## Bake the groups you changed

| If you changed | `bun run room:bake` |
|---|---|
| A light, a book's position, the cabinet, or anything shared | no flag |
| Geometry or materials, no lights | `--only shell,furniture,objects,moving,spines` |
| Spine artwork | `--only spines` |
| Book bodies | `--only books` |
| A note, added or removed | `--only covers` |
| An English post, added or removed | `--only sheets` |
| The bookmark | `--only bookmark` |

A partial bake stops if book positions or cabinet dimensions changed. The
current read needs no bake. A failed bake leaves the
repository unchanged. If export fails, the work directory keeps a
`-baked.blend`.

## Check the result

```sh
bun run build                  # includes the baked-content check
bun run room:check chromium    # a real browser
```

- `build` fails if a book, note, or post has no baked node, or if the GLBs
  exceed 11 MB. `bun run room:assets` runs only that check, in under a
  second.
- `room:check` loads the room at desktop and laptop sizes and checks that a
  phone keeps the poster. `--shots` saves frames in `/tmp/room-check`, and
  `--metrics` prints load time and transfer size. `webkit` tests Safari.
- If the overview changed, run `bun run room:poster`.

A change is done when its bake, `build`, and `room:check` pass.

## Set up Blender MCP

MCP is optional.

1. Install `uv`, then run
   `uvx --python 3.11 mcp-for-blender install-addon`. If the addon folder is
   missing, create
   `~/Library/Application Support/Blender/4.5/scripts/addons`, set
   `BLENDERMCP_ADDONS_DIR` to it, and retry.
2. Register the server. For Codex, run `codex mcp add blender` with the same
   arguments.

   ```sh
   claude mcp add blender -e BLENDER_MCP_SAFE_MODE=1 -e DISABLE_TELEMETRY=true \
   	-- uvx --python 3.11 mcp-for-blender
   ```

3. In Blender, enable **Interface: MCP for Blender**. Click **Start MCP
   Server** in the 3D View sidebar.
