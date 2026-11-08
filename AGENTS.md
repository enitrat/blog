# Repository instructions

## Authored content

- Keep Mathieu's prose, claims, contact details, and artwork as he wrote them.
  Ask before you replace it. This includes UI microcopy: labels, counters,
  placeholders, and button text in `.tsx` files. A brief from an
  orchestrating agent does not count as his approval.
- Change `DESIGN.md`, `CONTEXT.md`, and `PRODUCT.md` only through the
  impeccable skill. `src/ds/tokens/` holds every design value;
  `scripts/check-design-tokens.mjs` fails the build when `DESIGN.md` disagrees.

## Blender room

Read `docs/room-workflow.md` before any room change.

- Change `scripts/room.py` or `scripts/bake-room.mjs`, not live Blender.
- Publish assets only with `bun run room:bake --only <changed groups>`.
- A visual change is done when its bake, `bun run build`, and
  `bun run room:check chromium` pass.

## Home bench and UI atoms

Read `src/ds/README.md` before changing a token, an atom, or a tile.

- A visual or motion change is done when `bun run lint`, `bun test`,
  `bun run build`, and `bun run motion:check chromium` pass. Add
  `bun run motion:check webkit` when pointer or animation code changed.
- A change to a shared atom is checked on every route that uses it, at
  every breakpoint: `bun run shots` writes them all to `/tmp/shots`.

## Browser checks

- macOS has no `timeout`. Run the checks bare; each carries its own limits.
- Never measure while another agent edits the tree. `motion:check` builds and
  runs `next start` on the build; run it after edits land, or in a separate worktree. Never
  point a measurement at a shared `next dev`.
- Headless runs stub WebGL so the bench is measurable. The room is covered
  only by `room:check`, so a green `motion:check` says nothing about the room.

## Agent work

- An agent fixing a failing check never edits that check's thresholds,
  exemptions, or samples. A carve-out needs Mathieu's approval, and is
  re-tested once the real fix lands.
- Give every fix or investigation agent a time budget. It reports back after
  three failed hypotheses or 30 minutes, whichever comes first. Experiments
  run on a copy of the build, never on `.next/` in place.
- A final report lists every check the brief promised as passed, failed, or
  not run, with the reason for each one not run.
