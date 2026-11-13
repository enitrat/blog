# Behaviours

`globals.css` decides what things look like: palettes, roles, type, radii. This page covers what
things do: focus, pressing, motion, sizing and formatting. Each behaviour is decided once and
inherited everywhere. A component never re-decides one.

The rule behind every entry: if every component has to remember something, a component will
forget it. So the site does it once, globally, and `src/lib/behaviours.test.ts` fails on the
local workaround the global rule replaced.

## The behaviours

| Job | Decided in | Components write | Never write |
| --- | --- | --- | --- |
| Keyboard focus | `globals.css` `:focus-visible`: a 2px `--ring` outline, offset 2px | nothing | `outline-none`, `focus-visible:ring-*` |
| Reduced motion, CSS | `globals.css`, one unlayered `prefers-reduced-motion` block: keyframes stop; only colour, opacity and shadow transition | nothing | `motion-reduce:*`, `motion-safe:*` |
| Reduced motion, Motion | `<MotionConfig reducedMotion="user">` in `Providers` | `transition={spring.x}` | `reduce ? { duration: 0 } : …` |
| Decorative poses (a tile's object waking) | `useTileAwake()` in `bento-tile.tsx`: held, and motion welcome | `useTileAwake() ? POSE : REST` | `held && !reduce` in each tile |
| An element that only moves (a ping) | `motion-only` utility | `motion-only` | `motion-reduce:hidden` |
| 44px press target | `hit-area` utility: computed from the control's own box | `relative … hit-area` | `after:absolute after:-inset-1.5` |
| Whole card is one link | `link-cover` utility | `link-cover` on the link | `after:absolute after:inset-0` |
| Dates, counts, ratings | `src/lib/format.ts` | `formatDate(d)`, `formatRating(r)` | `Intl.*`, `toLocaleDateString` |
| Title typesetting | `typeset()` in `src/lib/typeset.ts`, applied by the content schema | nothing | per-title `&nbsp;` |
| Tile layout | Container queries on `@container/bento` and `@container/slide` | `@[36rem]/bento:…` | `sm:` / `lg:` inside a tile |
| Book covers | portrait by contract: `covers:fetch` rejects landscape scans, a test checks the folder | `object-cover` | `object-right` for wraparounds |
| Code highlighting | `markdown.ts` picks themes whose every token clears 4.5:1 on `--card` | nothing | — |

## Compensating code, and what replaced it

Each entry was a local fix for a missing global decision. Every one of them would come back the
next time a component forgot.

- **Focus rings by hand.** 32 copies of `outline-none focus-visible:ring-2 focus-visible:ring-ring`,
  plus `FOCUS` constants in five files, plus `ring-offset-well` tuned per surface. One missed copy
  meant a control with no visible focus. Replaced by one `:focus-visible` rule. An outline
  follows the radius and never collides with a component's `box-shadow`.
- **Reduced motion by hand.** About 60 checks: `motion-reduce:transition-none` on each transition,
  `reduce ? { duration: 0 } : spring` on each motion prop, `held && !reduce` on each pose. Replaced by
  one CSS block, `MotionConfig`, and `useTileAwake()` for the one place where intent (no pose at
  all) differs from "animate instantly".
- **Hit areas by guesswork.** `after:-inset-1`, `-inset-1.5`, `-inset-2`, `-inset-x-1 -inset-y-3`,
  each worked out by hand for one control's size, and wrong as soon as the size changed. Replaced
  by `hit-area`, which computes its overhang from the control's own box: `min(0px, (100% - 44px) / 2)`.
- **Viewport breakpoints in a sidebar-narrowed bento.** `lg:grid-cols-6` gave each tile about 190px
  at 1024px, because the sidebar takes 20rem the window breakpoint cannot see. The tiles
  compensated with `truncate` on titles ("Boun…"), a `title` tooltip, and container queries inside
  tiles to survive being too small. Replaced by container queries on the bento itself. The tiles
  never get smaller than their content, so the truncation went.
- **Odd tile count.** In two columns the tiles left a hole. A layout's cell count is now part of
  its specification (see the comment in `src/app/page.tsx`).
- **Rendering the sidebar twice.** `HomeSidebar` and `HomeFactsBelow` rendered the facts and the room
  twice: two ids (`facts-title`, `facts-below-title`) and two copies of the LCP image, one hidden.
  Replaced by one aside that is `display: contents` below `lg`, so its blocks join the page grid in
  their own order.
- **Hiding the footer on home.** It hid at `lg` because the bento had to fit one screen, and when it
  showed it pushed home's pinned sidebar up by its own height. Its contact links repeated home's
  Elsewhere tile, so the footer is gone and every page ends where its content does.
- **Holding a place for missing banners.** Writing rows kept an empty 224px slot, and titles capped at
  44ch, so "Cairo / 1.0" split. Rows without a banner now use the full width. Numbers are bound
  to the word before them at the content layer.
- **Bottom-anchored text in a tall tile.** The Writing slide pinned its text to the bottom of a
  two-row tile, which left the top half blank. The slide now reads like a magazine card: date and
  title at the top, summary and Read more at the bottom, a size up when the slide is wide.
- **Focus mirrored into React state.** `WorkRow` kept `focused` state so `whileFocus` would fire on an
  `<li>`. It is CSS now (`group-focus-visible:`), and the component is a server component.
- **A live iframe of another app.** The Project tile booted morph.msaug.dev at 250% and scale 0.4,
  with `tabIndex={-1}` and `aria-hidden` to keep it out of the way. It is now a screenshot,
  `src/assets/projects/morph.png`. Retake it when morph's landing page changes.
- **A second full-size image for a blur.** `GlowImage` loaded the same rendition twice, one of them
  blurred by 22px. The glow now asks for a 2rem rendition.
- **Dead motion components.** `tilt`, `magnetic`, `hover-scale`, `equalizer` and `book-spine` were
  never rendered. They are deleted.

## Adding something new

- A new control: make it `relative`, give it `hit-area` if it is drawn under 44px, and do nothing
  for focus.
- A new animation: take a spring from `src/lib/motion.ts` and do nothing for reduced motion. If the
  motion is a decorative pose rather than a state change, gate it the way `useTileAwake()` does.
- A new component that sits inside a layout it does not own: size it with a container query.
- A new date or number: add a function to `src/lib/format.ts` if no existing one fits.
- When a fix needs the same three classes in a second file, it is a behaviour. Name it here, put it
  in `globals.css` or `src/lib`, and add a test for the workaround.
