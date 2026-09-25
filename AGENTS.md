# Repository instructions

## Authored content

- Keep Mathieu's prose, claims, contact details, and artwork as he wrote them.
  Ask before you replace it.
- Change `DESIGN.md`, `CONTEXT.md`, and `PRODUCT.md` only through the
  impeccable skill. `DESIGN.md` generates `src/styles/tokens.css`.

## Blender room

Read `docs/room-workflow.md` before any room change.

- Change `scripts/room.py` or `scripts/bake-room.mjs`, not live Blender.
- Publish assets only with `bun run room:bake --only <changed groups>`.
- A visual change is done when its bake, `bun run build`, and
  `bun run room:check chromium` pass.
