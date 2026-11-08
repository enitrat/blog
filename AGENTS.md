# Repository instructions

## Authored content

- Keep Mathieu's prose, claims, contact details, and artwork as he wrote them.
  Ask before you replace it. This includes UI microcopy: labels, counters,
  placeholders, and button text in `.tsx` files. A brief from an
  orchestrating agent does not count as his approval.
- Change `DESIGN.md`, `CONTEXT.md`, and `PRODUCT.md` only through the
  impeccable skill. The `@theme` block in `src/app/globals.css` holds every
  design value; keep `DESIGN.md` in step with it.

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
