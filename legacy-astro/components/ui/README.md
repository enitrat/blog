# UI atoms

Atoms live here. Section docs for the `/system` page live in `src/components/system/`.

## Three token layers

1. **Reference** (`DESIGN.md` frontmatter, generated into `src/styles/tokens.css`): raw values. `--color-grey-*`, `--color-orange-*`, `--font-geist*`, `--font-size-*`, `--space-*`, `--layout-*`, `--radius-*`, `--duration-*`, `--ease-*`.
2. **Semantic** (`src/styles/theme.css`, hand-written): intent, with light and dark values. `--surface`, `--muted-foreground`, `--text-label`, `--radius-tile`, `--font-code`. May reference reference tokens only.
3. **Component** (custom properties inside a component's `<style>`): may reference semantic tokens only.

**Layer rule.** Colour, font families and text roles must come from semantic tokens (`--foreground`, `--font-code`, `--text-ui`). Scale primitives may be used directly: spacing (`--space-*`), radius (`--radius-*`) and motion (`--duration-*`, `--ease-*`). No literal colors, font sizes or durations outside `tokens.css` and `theme.css`.

Semantic sizing tokens replace magic numbers: `--accent-graphic` (the accent for small marks such as dots and underlines, 3:1 on the surface; `--accent` stays for fills), `--list-max` (readable cap for index lists), `--target-min` (44px touch target), `--dot-size` (status dot), `--opacity-disabled`, `--control-height-sm` and `--control-height-md` (44px, the default `Button`). `--scrim` is a fixed dark overlay that does not flip with the theme (text over photos).

## Inverse surfaces

Atoms never take an inverse prop. A dark container re-maps the paper names once, in `theme.css`:

```css
[data-surface="inverse"], .tile[data-variant="inverse"], .tile[data-variant="media"] {
	--surface: var(--surface-inverse); --foreground: var(--surface-inverse-foreground); /* ... */
}
```

Inside it, `--surface`, `--surface-foreground`, `--foreground`, `--muted-foreground`, `--border`, `--link` and `--ring` already hold the inverse values, so atoms written against them just work. Rules:

- Write atoms against the paper names. An "inverse fill" is `background: var(--foreground); color: var(--surface)` (Pill link, Tag neutral, Button inverse); it flips by itself.
- Never swap `--surface` and `--surface-inverse` inside the rule: each would reference the other and create a cycle.
- Use `data-surface="inverse"` on any plain container (give it `background: var(--surface)`), or an inverse or media `Tile`.
- A `Tile` (paper) nested in an inverse one resets through `--paper-*`, frozen at `:root`.
- `DitherImage` reads colours from its own box, so it follows the surface too.

## Variants

`variant` and `size` props become `data-variant` and `data-size` attributes, styled in the component's scoped CSS. No cva, no tailwind.

```astro
---
/**
 * Button
 * @prop variant "primary" | "ghost" | "inverse" (default "primary")
 * @prop size "md" | "sm" (default "md")
 * @prop href Renders an <a> when set, a <button> otherwise
 */
interface Props { variant?: 'primary' | 'ghost' | 'inverse'; size?: 'md' | 'sm'; href?: string }
const { variant = 'primary', size = 'md', href } = Astro.props;
---
<style>
	[data-variant="primary"] { background: var(--accent); color: var(--accent-foreground); }
</style>
```

## Tiles, links and headings

A `Tile` with `href` is one link: it must not contain links or buttons. Pass `labelAs` ("span", "h2" or "h3") when the tile label is a real heading, and `dot` (with `dotTone`) for the standard status dot beside the label. Molecules never restyle an atom from outside: `Label` exposes `grow` for flex headers.

## Pages

Every page body is `Page` (`width="wide"` for index pages at `--page-max`, `"measure"` for essay, about and work detail) opening with `PageTitle`, the one h1 voice. Rows go in `ListRows`; a book is a `BookLine`. A page never reaches into an atom with `:global(...)`: use the atom's props (`Label space`, `Label separated`, `ListRow headingLevel`).

## Touch targets

Interactive atoms reach 44px (`--target-min`): `Button` md is 44px tall, and `Button` sm, `Pill` and non-inline `Link` extend an invisible 44px hit area in both axes with `::after`.

## One of each

Look here before creating anything. If it exists, add a variant. When a new component replaces an old one, delete the old one and its dead CSS. Molecules compose atoms and never restyle their internals; they pass `variant`, `size` or a class.

## Elevation and motion

Tiles sit on the sand; objects sit on tiles. Shadows come only from the semantic elevation tokens, which flip for dark mode:

| Token | Use |
|---|---|
| `--elevation-tile` | every `Tile` at rest (set by `Tile`, never by a tile's content) |
| `--elevation-tile-lift` | a linked `Tile` on hover (set by `Tile`) |
| `--elevation-object` | a physical thing on a tile: book cover, set artwork, writing card (`Plate`) |
| `--elevation-object-lift` | that object while it moves or is open (`Deck`) |

`--lift-distance` (-2px) and `--press-scale` are the only hover/press magnitudes. **Only links lift.** Text, labels and pills never cast a shadow; no glass, no blur, no zero-offset halo except `Plate`'s `glow`, which is sampled from its own image.

Motion vocabulary (reference scale, used directly):

- `--duration-fast` + `--ease-out`: state (colour, border, press).
- `--duration-base` + `--ease-out`: hover lift and every exit.
- `--duration-moderate` + `--ease-spring`: an object opening (a deck fanning, a glyph reacting). Stagger 40ms per item on enter (`STAGGER` in `motion.ts`), none on exit.
- Springs live in `src/lib/motion.ts` (`SPRING_LAYOUT`, `SPRING_PRESS`, `SPRING_SETTLE`, `SPRING_ROLL`, ...), never inline. A script animates only `transform`, `opacity`, `box-shadow` (and `clip-path`), on the element itself: never `filter` or layout properties. Dim with an overlay's opacity, not a filter.
- Motion that retargets every pointer frame animates single axes (`x`, `y`, `rotateX`, ...) or motion values, never a composed `transform` string: a string target re-reads computed style on each retarget and drops the spring's velocity, which WebKit shows as a jump.
- Hover motion lives inside `@media (hover: hover)`. Touch gets a static pose that already shows everything.
- `prefers-reduced-motion: reduce`: no translation, rotation or spin; colour and opacity changes stay. State still changes: a deck still fans, its poses are painted instantly.
- **One entrance, and only one**: `Bench settle` (home page only) is the single page-load entrance animation: tiles rise 12px and fade, 30ms apart for the first four, under `--duration-moderate` (350ms ease-out, so the last tile lands at 440ms). Nothing else enters on load, and no other list or page may copy it.

## Shared primitives for bench objects

- **Plate** (atom): one physical object (image or slotted content) with `--elevation-object`, a radius, a fixed `ratio`, and an optional `glow` colour behind it.
- **Deck** (molecule): `<Deck items={xs}>{(x, i) => <Card />}</Deck>` renders a hand of cards and wraps each in its own card element (index and rest pose included; no `DeckCard`). All cards share one grid cell, so they match the tallest. It stacks at rest and lifts one card while its host (`[data-deck-host]`, default the `Deck` itself) is hovered or has focus inside. The active card is chosen from the pointer's x over the deck's fixed footprint (one cached rect, dropped on scroll, resize and when the host's own motion ends), never from `:hover` on a moving card. Back cards stay readable: a light dim (about 7%), a smaller scale and their shadow carry the depth. Where a hover pointer can fan the hand, the back cards' lazy images are decoded at idle so the first fan has no long frame; touch never fans and keeps lazy loading. A long frame on the first interaction is a cold resource to fix, never a sample to exempt.
  - `hold="fan"` (default): the hand spreads sideways like playing cards (Writing's index cards). `hold="pull"`: the card is drawn up and out of the pile, turned in perspective (Reading's books); `lift` is how far it rises, `spread` and `angle` are ignored. The pile peeks more (12% right, 7% up), and the pointer picks the book physically: the card whose resting box is hit first from the front wins, so the peeking strip of a back book pulls it; off the pile, position along the pile's offset decides (`pullZone`). `ratio="5/7"` gives cards a minimum shape (taller than their content). Poses come from the item count.
  - **Set down**: when the host is left the cards gather with the pile held `KNOCK`% high, then drop with one small knock on `SPRING_SETTLE` (overshoot about 1px). Reduced motion paints the rest pose instantly.
  - `cycle={s}` deals the front card to the back every `s` seconds while the deck is in view and untouched (`SPRING_DEAL`), and stops for good at the first pointer hover or keyboard focus (a mouse click's focus does not count).
  - Reporting the held item: give the Deck an `id` and `captions={[{ title, date }, ...]}` (one record per item). It fires `deck:active` (bubbles, `detail: { item, caption }`) whenever the held card changes; the front card is the held one at rest, so a deal changes it too. A `Readout` with `follow="<id>"` and `field="title"` shows it with no tile script. `--deck-fit` lets the hand fill more than 96% of its container (a tile bleeds it into its padding). `data-open` and `data-active` expose state.
- **Readout** (atom): text that reports a held object's state and changes by rolling, never by cutting: the old value leaves up and fades, the new one enters from below, through a one-line window. `mode="text"` (default) rolls the whole string and truncates; `mode="char"` rolls only the characters that change, aligned from the right, with tabular numerals (a clock, a count). It renders plain text, so it works without JS. Interruptible: a change mid-roll retargets from where the glyph is (an outgoing glyph finishes fading on its own); `SPRING_ROLL`, transform and opacity only; reduced motion is a crossfade (`ROLL_FADE`). The real text sits in a visually hidden node and the rolling glyphs are `aria-hidden`; `live` adds `aria-live="polite"` for a value that changes by itself and matters, never for hover-driven text.
  - Props: `value`, `as` (default `span`), `mode`, `live`, `follow`, `field`, `id`, `class`; `--readout-justify: end` right-aligns text in a wider box.
  - From a script: `import { setReadout } from '../ui/readout'` then `setReadout(el, "1:07")`. Call it on every tick; an unchanged value does nothing. `el` is the Readout's root (`[data-readout]`, give it an `id` or `data-` hook). No animation code of your own.
  - Following a Deck: `<Readout follow="writing-deck" field="title" value={first.title} />`. Server-render `value` with the front item's caption.
- **Equalizer** (atom): small bars in `--accent-graphic` that move only inside a `[data-playing]` ancestor. Orange means playing.
- **set player** (`src/lib/set-player.ts`): the one SoundCloud widget and its playback state. The room and the set tile are two views of it.

## Pointer primitives

Objects react to the pointer; nothing else does. Springs come from `src/lib/motion.ts`, never inline.

- **Tilt** (`max` = 7.5 degrees at the edge, `glare`): wraps a `Plate`. It leans toward the pointer in perspective and settles flat on leave. Spin or animate on an inner element, never on the Tilt.
- **Magnetic** (`strength` = 0.35; use 0.15 to 0.2 on small targets): its child drifts toward the pointer inside its box plus 16px and springs home.
- **Glide** (`shape` = "pill" | "row", `as`): a group of `data-glide-item`s sharing ONE highlight (`--highlight`) that glides to the hovered or focused item on `SPRING_GLIDE` (port of beUI `SharedLayoutBg` and `Dock`). Appears in place, fades on leave; clip-path and opacity only. Items keep their own hover without JS (`data-glide-ready` switches it off). The header nav and the Work tile's rows are Glide groups; `glide.ts` drives them.
- **Record** (the Set tile's artwork): turning speed is a motion value on `SPRING_RECORD`, integrated per frame into an angle, so play spins up from wherever the record rests and pause coasts down and stays at that angle (nothing snaps back to 0). The frame step clamps its `dt`, so a tab returning from hidden does not jump the record. Square to round is a `clip-path` transition on the face, with two static shadows (square, round) crossfaded by opacity, since a shadow cannot follow a clip-path; the face turns, the light does not. Switching reduced motion on mid-play stops the turn and keeps the angle. Reduced motion: round, no spin. Its elapsed time is a `Readout` in `char` mode.
- Tilt and Magnetic are driven by one script (`pointer.ts`) over `[data-tilt]` and `[data-magnetic]`. Inert on touch and under reduced motion.
- **Layout reads**: a pointer script never reads layout per move. It measures once (Tilt on the first move after `pointerenter`, Magnetic and Glide lazily) and drops the cached rect on scroll, resize and `transitionend`, like Deck. `bun run motion:check` fails above one read per `pointermove`.
- **Press**: `Button` and interactive `Pill` scale to `--press-scale-control` on `:active` (CSS, `--duration-fast`, `--ease-out`; none under reduced motion). `Tile` keeps its own `--press-scale`.
