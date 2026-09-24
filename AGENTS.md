# Repository instructions

## Authored content

Keep Mathieu's prose, claims, contact details, and artwork as he wrote them.
Ask before you replace any of it.

Don't edit `DESIGN.md`, `CONTEXT.md`, or `PRODUCT.md` by hand. They use the
impeccable skill's formats, and `DESIGN.md`'s frontmatter generates
`src/styles/tokens.css`.

## Blender room

Before you change the living room's geometry, materials, lights, camera,
bake, or assets, read `docs/room-workflow.md`.

- Change `scripts/room.py` and `scripts/bake-room.mjs`. A live Blender scene is
  only a preview.
- Publish assets only through `bun run room:bake --only <the groups you
  changed>`.
- A visual change is done when its bake, `bun run build`, and
  `bun run room:check chromium` all pass.
