# Repository instructions

## Authored content

- Keep Mathieu's prose, claims, contact details, and artwork as he wrote them.
  Ask before you replace it. This includes UI microcopy: labels, counters,
  placeholders, and button text in `.tsx` files. A brief from an
  orchestrating agent does not count as his approval.
- Change `CONTEXT.md` and `PRODUCT.md` only through the impeccable skill.
- `src/app/globals.css` holds every design value as semantic tokens: palettes,
  colour roles, type roles, radii, spacing and widths. Components name roles
  only; `src/lib/design-tokens.test.ts` enforces it. `/design` shows the
  system, and `docs/design/palette.md` records why the palette is Ink.

## Blender room

Read `docs/room-workflow.md` before any room change.

- Change `scripts/room.py` or `scripts/bake-room.mjs`, not live Blender.
- Publish assets only with `bun run room:bake --only <changed groups>`.
- A visual change is done when its bake and `bun run build` pass.

## UI

- Components follow shadcn conventions in `src/components/ui/`, styled with
  stock Tailwind utilities on the `globals.css` palette. Springs come from
  `src/lib/motion.ts`; never inline one.
- A visual or motion change is done when `bun run lint`, `bun run test`, and
  `bun run build` pass, and the changed pages have been looked at in light
  and dark mode at phone and desktop widths.

## Browser checks

- macOS has no `timeout`. Run checks bare.
- Never measure while another agent edits the tree. Measure a `next start`
  of a finished build, never a shared `next dev`. Kill the server by port
  when done: `next start` leaves a `next-server` child running.

## Agent work

- An agent fixing a failing check never edits that check's thresholds,
  exemptions, or samples. A carve-out needs Mathieu's approval, and is
  re-tested once the real fix lands.
- Give every fix or investigation agent a time budget. It reports back after
  three failed hypotheses or 30 minutes, whichever comes first. Experiments
  run on a copy of the build, never on `.next/` in place.
- A final report lists every check the brief promised as passed, failed, or
  not run, with the reason for each one not run.

<!-- BEGIN:nextjs-agent-rules -->

## This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
