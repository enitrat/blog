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
- Springs live in `src/lib/motion.ts` (`SPRING_LAYOUT`, `SPRING_PRESS`, ...), never inline. A script animates only `transform`, `opacity`, `box-shadow` (and `clip-path`), on the element itself: never `filter` or layout properties. Dim with an overlay's opacity, not a filter.
- Hover motion lives inside `@media (hover: hover)`. Touch gets a static pose that already shows everything.
- `prefers-reduced-motion: reduce`: no translation, rotation or spin; colour and opacity changes stay. State still changes: a deck still fans, its poses are painted instantly.
- **One entrance, and only one**: `Bench settle` (home page only) is the single page-load entrance animation: tiles rise 12px and fade, 30ms apart for the first four, under `--duration-moderate`. Nothing else enters on load, and no other list or page may copy it.

## Shared primitives for bench objects

- **Plate** (atom): one physical object (image or slotted content) with `--elevation-object`, a radius, a fixed `ratio`, and an optional `glow` colour behind it.
- **Deck** (molecule): `<Deck items={xs}>{(x, i) => <Card />}</Deck>` renders a hand of cards and wraps each in its own card element (index and rest pose included; no `DeckCard`). It stacks at rest and fans in place while its host (`[data-deck-host]`, default the `Deck` itself) is hovered or has focus inside. Poses come from the item count. The active card is chosen from the pointer's x over the deck's fixed footprint, never from `:hover` on a moving card. Opt-in `cycle={s}` deals the front card to the back every `s` seconds while the deck is in view and untouched (`SPRING_DEAL`), and stops for good at the first pointer hover or keyboard focus (a mouse click's focus does not count). Cards on Writing (links) and covers on Reading (a linked tile) are both a `Deck`. `data-open` and `data-active` expose state.
- **Equalizer** (atom): small bars in `--accent-graphic` that move only inside a `[data-playing]` ancestor. Orange means playing.
- **set player** (`src/lib/set-player.ts`): the one SoundCloud widget and its playback state. The room and the set tile are two views of it.

## Pointer primitives

Objects react to the pointer; nothing else does. Springs come from `src/lib/motion.ts`, never inline.

- **Tilt** (`max` = 7.5 degrees at the edge, `glare`): wraps a `Plate`. It leans toward the pointer in perspective and settles flat on leave. Spin or animate on an inner element, never on the Tilt.
- **Magnetic** (`strength` = 0.35; use 0.15 to 0.2 on small targets): its child drifts toward the pointer inside its box plus 16px and springs home.
- **Glide** (`shape` = "pill" | "row", `as`): a group of `data-glide-item`s sharing ONE highlight (`--highlight`) that glides to the hovered or focused item on `SPRING_GLIDE` (port of beUI `SharedLayoutBg` and `Dock`). Appears in place, fades on leave; clip-path and opacity only. Items keep their own hover without JS (`data-glide-ready` switches it off). The header nav and the Work tile's rows are Glide groups; `glide.ts` drives them.
- Tilt and Magnetic are driven by one script (`pointer.ts`) over `[data-tilt]` and `[data-magnetic]`. Inert on touch and under reduced motion.
- **Press**: `Button` and interactive `Pill` scale to `--press-scale-control` on `:active` (CSS, `--duration-fast`, `--ease-out`; none under reduced motion). `Tile` keeps its own `--press-scale`.
