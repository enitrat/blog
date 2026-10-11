---
name: Mathieu Saugier
description: A cool grey bench of flat tiles, Geist type, one signal orange for state, and one dark lit room.
# Reference tokens: generated into src/styles/tokens.css by scripts/generate-tokens.mjs.
colors:
  grey-50: "#f5f5f3"
  grey-100: "#eaeaed"
  grey-200: "#dcdde0"
  sand-200: "#ebe7da"
  grey-300: "#c3c5c9"
  grey-400: "#8e939a"
  grey-500: "#6b6f76"
  grey-600: "#575b62"
  grey-700: "#3f4247"
  grey-800: "#26282c"
  grey-850: "#1b1c1f"
  grey-900: "#16171a"
  grey-950: "#0e0e10"
  orange-300: "#ff9a70"
  orange-400: "#ff7a47"
  orange-500: "#ff5b1f"
  orange-600: "#c2410c"
  manuscript-paper: "#f5f0e5"
  manuscript-ink: "#242018"
  manuscript-scroll: "#94856b"
  manuscript-rule: "#d8cfbd"
  manuscript-shade: "#766647"
  manuscript-shadow: "rgb(25 18 10)"
  manuscript-scrim: "rgb(28 23 17)"
  lamp-floor: "rgb(255 196 120)"
  lamp-banker: "rgb(150 255 170)"
  lamp-amber: "rgb(255 190 110)"
  lamp-glow: "rgb(255 214 150)"
  lamp-ember: "rgb(255 180 110)"
fonts:
  geist: '"Geist Variable", system-ui, -apple-system, "Segoe UI", sans-serif'
  geist-mono: '"Geist Mono Variable", ui-monospace, SFMono-Regular, Menlo, monospace'
fontSize:
  xs: "0.6875rem"
  sm: "0.8125rem"
  base: "0.9375rem"
  md: "1.125rem"
  lg: "1.5rem"
  xl: "1.75rem"
  2xl: "clamp(1.75rem, 1rem + 2.4vw, 2.75rem)"
  3xl: "clamp(2.5rem, 1.5rem + 4vw, 4.5rem)"
leading:
  tight: "1.1"
  snug: "1.25"
  normal: "1.5"
  relaxed: "1.62"
weight:
  regular: "400"
  medium: "500"
  semibold: "600"
tracking:
  tight: "-0.03em"
  snug: "-0.02em"
  normal: "0"
  wide: "0.09em"
space:
  1: "0.25rem"
  2: "0.5rem"
  3: "0.75rem"
  4: "1rem"
  5: "1.25rem"
  6: "1.5rem"
  8: "2rem"
  10: "2.5rem"
  12: "3rem"
  16: "4rem"
  20: "5rem"
  24: "6rem"
layout:
  page-gutter: "clamp(1rem, 2.5vw + 0.25rem, 1.75rem)"
  bench-gap: "1.25rem"
  tile-pad: "1.625rem"
  page-max: "77.5rem"
  measure: "38rem"
rounded:
  sm: "6px"
  md: "10px"
  lg: "14px"
  xl: "18px"
  full: "999px"
motion:
  duration:
    fast: "150ms"
    base: "250ms"
    slow: "600ms"
    slower: "1200ms"
    moderate: "350ms"
  ease:
    out: "cubic-bezier(0.2, 0.8, 0.2, 1)"
    in-out: "cubic-bezier(0.65, 0, 0.35, 1)"
    spring: "cubic-bezier(0.34, 1.4, 0.64, 1)"
elevation:
  tile: "0 1px 2px rgb(22 23 26 / 0.04), 0 2px 6px rgb(22 23 26 / 0.04)"
  tile-lift: "0 2px 4px rgb(22 23 26 / 0.05), 0 14px 28px -10px rgb(22 23 26 / 0.18)"
  object: "0 1px 2px rgb(22 23 26 / 0.12), 0 6px 16px -4px rgb(22 23 26 / 0.22)"
  object-lift: "0 2px 4px rgb(22 23 26 / 0.12), 0 20px 36px -12px rgb(22 23 26 / 0.34)"
  tile-dark: "0 1px 2px rgb(0 0 0 / 0.4), inset 0 1px 0 rgb(255 255 255 / 0.04)"
  tile-lift-dark: "0 2px 4px rgb(0 0 0 / 0.45), 0 16px 32px -10px rgb(0 0 0 / 0.6), inset 0 1px 0 rgb(255 255 255 / 0.06)"
  object-dark: "0 1px 2px rgb(0 0 0 / 0.5), 0 8px 18px -4px rgb(0 0 0 / 0.55)"
  object-lift-dark: "0 2px 4px rgb(0 0 0 / 0.5), 0 22px 40px -12px rgb(0 0 0 / 0.7)"
---

# Design System: Mathieu Saugier

## Overview

**Creative North Star: "The Bench"**

The site is a sand-tinted workbench on which small, real objects from Mathieu's life are laid out: a bio, what he is doing now, the book he is reading, the latest essay, a trip, the work. Each object is a flat tile and each tile is a door. Recognisable with the content removed: a sand field, flat near-white rectangles of varied size, mono caps labels, one orange dot. Geist does all the talking; Geist Mono labels and counts.

The system is flat on purpose. Tiles are separated by space and a 1px hairline, never by shadow, glass or gradient. Orange is a signal for state (playing, current, latest, pressed, focus), not decoration. The only dark, textured, lit object on the bench is the 3D living room. Everything else is paper and graphite.

The essay breaks no rule of the bench; it only removes the bench. Reading pages are a single plain column, no banner, no sidebar, no cards.

**Key Characteristics:**
- Flat tiles on a grey field: 1px hairline, no drop shadow, no glass, no blur.
- One family (Geist) plus its mono (Geist Mono) for labels, data and code.
- Orange only for state; `--accent-graphic` for small marks (dots, underlines), `--accent` for fills.
- Light and dark are both first-class; a tile can locally invert without any prop on its atoms.
- Photos are ordered-dither duotones that reveal true colour on hover, focus or tap.
- The room is the one dark object, painted with its own art-direction colours that never change with the theme.

## Colors

A cool grey scale does nearly all the work; one signal orange marks state. Two sets of fixed art colours belong to the room and are not part of the interface palette.

### Primary
- **Signal Orange** (`orange-500`, `orange-400` and `orange-300` for lighter steps): state fills (primary button, pressed toggle, text selection) and, via `--accent`, the one accent on a bench. Dark text sits on it (`--accent-foreground`).
- **Burnt Orange** (`orange-600`): the accent for small marks and focus rings on paper, so dots, underlines and rings reach 3:1 on the near-white surface. Exposed as `--accent-graphic` and `--ring`; on dark and inverse surfaces they swap to `orange-500` and `orange-400`.

### Neutral
- **Bench Sand** (`sand-200` light, `grey-950` dark): the page field behind the tiles (`--background`), a light grey with a slight yellow tint. Every page sits on it, essays included.
- **Paper** (`grey-50` light, `grey-850` dark): the tile surface (`--surface`).
- **Graphite** (`grey-850` light, `grey-100` dark): the inverse surface (`--surface-inverse`), the one graphite tile family.
- **Ink** (`grey-900` light, `grey-100` dark): text and neutral link colour. Links are not blue.
- **Muted** (`grey-600` light, `grey-400` dark for text; `grey-100` / `grey-800` for fills): metadata and secondary lines.
- **Hairline**: `--border` is `grey-900` at 12% (light) or `grey-50` at 14% (dark). `grey-300` through `grey-700` fill out the scale.
- **Scrim** (`grey-950` at 75% and 94%): a fixed dark overlay that never flips, so text over photos stays readable in both themes.

### Art colours (the room only)
- **Manuscript** (`manuscript-paper`, `-ink`, `-scroll`, `-rule`, `-shade`, `-shadow`, `-scrim`): the warm parchment of the open book and bookshelf inside the room.
- **Lamp** (`lamp-floor`, `-banker`, `-amber`, `-glow`, `-ember`): the lamplight painted into the render and its overlays.

These are aliased once in `src/components/home/room/room.css` and used only by room components. They are the one sanctioned place a component reads a reference colour directly.

### Named Rules
**The Orange Is State Rule.** Orange means playing, current, latest, pressed or focused. If removing it loses no meaning, it should not be there.
**The Small Marks Rule.** Dots and underlines use `--accent-graphic`; fills use `--accent`. Never paint a small orange mark with `--accent` on paper.
**The Fixed Scrim Rule.** Text over a photo uses `--scrim` and `--scrim-foreground`, which do not change with the theme.

## Typography

**Display / Body / UI Font:** Geist Variable (system-ui, sans-serif fallbacks)
**Label / Mono / Code Font:** Geist Mono Variable (ui-monospace fallbacks)

**Character:** One quiet grotesque for everything, with mono as the instrument-panel voice for labels, dates, counts and code. Hierarchy comes from size, tracking and weight, not from a second display face.

### Hierarchy
- **Display** (`--text-display`, 2xl: clamp 1.75rem to 2.75rem, leading 1.1, tracking -0.03em): the bio statement.
- **Title** (`--text-title`, xl: 1.75rem, leading 1.25, tracking -0.02em, semibold, balanced wrap): the one `PageTitle` h1 voice on every page; large `BookLine` titles.
- **Heading** (`--text-heading`, lg: 1.5rem, leading 1.25): essay section headings, page lead line, book titles.
- **Body** (`--text-body`, md: 1.125rem, leading 1.62, regular): essay prose, `--measure` 38rem.
- **UI** (`--text-ui`, base: 0.9375rem, leading 1.5, medium for controls): default text, buttons, rows.
- **Small** (`--text-small`, sm: 0.8125rem): descriptions, dates, captions.
- **Label** (`--text-label`, xs: 0.6875rem, Geist Mono, tracking 0.09em, uppercase): tile headers, tags, pills, meta.

### Named Rules
**The One Family Rule.** Geist for everything, Geist Mono for labels, data and code. No second display face, no serif.
**The Semantic Size Rule.** Components use the `--text-*` roles, never raw `--font-size-*` or literal sizes.

## Layout

A flat bench: one column on phones, a 6-column grid from 40rem, a 12-column grid from 64rem (`Bench`). Each `Tile` declares its `span` at md and lg, optional `rows` at lg, and an `order` for the phone stack. The gap is `--gap-bench` (1.25rem); tile padding is `--pad-tile` (1.625rem); the page gutter is `--page-gutter` (clamp 1rem to 1.75rem); the page container is `--page-max` (77.5rem).

Spacing steps are `--space-1` to `--space-24` (0.25rem to 6rem). A `Page` is a single grid column with `--space-10` between sections. Width `wide` fills `--page-max` for index pages; width `measure` centres a 38rem column for essays, about and work detail. Index lists cap at `--list-max` (1.5 times the measure) so rows stay readable.

Every interactive target reaches 44px (`--target-min`); small controls extend an invisible hit area with a pseudo-element.

### Essay reading rules
- **In short** opens every essay: a mono label and a numbered list of section titles, each with its first sentence, each linking to its section. Built at build time, works without JS. Each section links back to it.
- Body is justified with automatic hyphenation (left-aligned where hyphenation is unsupported, and on very narrow screens) at Geist `--text-body`, measure 38rem.
- No banner image, no sidebar, no floating contents, no cards. The page is the title, a dot-separated meta line, In short, and the prose.

### Named Rules
**The Bench Grid Rule.** Tiles are placed with `span`, `rows` and `order`; pages never write their own grid for a bench.
**The Plain Reading Rule.** The essay is one column. Apparatus (In short, back links) is orientation, never the only path.

## Elevation & Depth

Two levels, both soft and offset. Tiles sit on the sand with a barely-there shadow (`--elevation-tile`) and a 1px hairline; a linked tile lifts 2px on hover to `--elevation-tile-lift` and settles on press. Objects inside tiles (book covers, set artwork, writing cards) are physical things on the tile and carry `--elevation-object`, deepening to `--elevation-object-lift` when they move. Dark mode swaps in deeper black shadows with a faint top highlight. Controls draw their outline with an inset 1px ring, not a shadow.

### Named Rules
**The Objects Cast Shadows Rule.** Shadow says "this is a thing you can pick up": tiles and the objects on them. Text, labels, pills and the reading pages never cast one. No glass, no blur, no zero-offset coloured halos except the cover glow sampled from the cover itself.
**The Lift Means Link Rule.** Only something that goes somewhere when clicked lifts on hover.

## Shapes

Soft but not round: tiles use `--radius-tile` (18px, `xl`); smaller scale steps are sm 6px, md 10px, lg 14px. Pills and buttons are fully round (`--radius-pill`, 999px). Hairlines are always 1px. Photos inherit the tile's corner via `border-radius: inherit` and `overflow: hidden`.

## Components

Layering runs one way: atoms, then molecules, then tiles, then pages. Look before creating; add a variant instead of a new component; when something replaces another, delete the old one. `src/components/ui/README.md` is the convention document and `/system` is the living render of it.

### Three token layers
1. **Reference** (this file's frontmatter, generated into `src/styles/tokens.css`): raw values `--color-*`, `--font-*`, `--font-size-*`, `--space-*`, `--layout-*`, `--radius-*`, `--duration-*`, `--ease-*`. Primitives never reference each other.
2. **Semantic** (`src/styles/theme.css`, hand-written): intent with light and dark values (`--surface`, `--muted-foreground`, `--text-label`, `--radius-tile`). May reference reference tokens only.
3. **Component** (custom properties inside a component's `<style>`): may reference semantic tokens only.

Colour, font families and text roles come from semantic tokens. Scale primitives (spacing, radius, motion) may be used directly. No literal colours, font sizes or durations outside `tokens.css` and `theme.css`; the room's art colours are the one declared exception.

### Scoped inverse surface
Atoms never take an inverse prop. `[data-surface="inverse"]` and inverse or media `Tile`s re-map the paper names (`--surface`, `--foreground`, `--muted-foreground`, `--border`, `--link`, `--ring`, `--accent-graphic`) once, in `theme.css`. Write atoms against the paper names; an inverse fill is `background: var(--foreground); color: var(--surface)` and flips by itself. A paper tile nested in an inverse one resets through the `--paper-*` names frozen at `:root`. Never swap `--surface` and `--surface-inverse` inside the rule: it creates a cycle.

### Atoms (`src/components/ui/`)
- **Tile / Bench**: the flat surface and its grid. Variants `paper`, `inverse`, `media` (zero padding, full-bleed, header over a scrim). A tile with `href` is one link and holds no links or buttons.
- **Button**: pill action, 44px tall. `primary` (orange fill), `ghost` (hairline outline), `inverse` (foreground fill). Sizes `md`, `sm`.
- **Pill**: small mono capsule. `link` (foreground fill, orange on hover), `chip` (outlined), `toggle` (outlined, orange when pressed).
- **Label**: mono caps caption with optional status dot and right-aligned meta.
- **Link**: `inline`, `quiet`, `arrow`, `external`.
- **Tag**, **StatusDot** (7px), **Icon**, **Rule**.
- **ListRow / ListRows**: mono date column, title, muted description, optional tag; arrow appears on hover and focus.
- **PageTitle / Page**: the one h1 voice and the one page layout.
- **BookLine / BookSpine / Shelf**: the reading list.
- **DitherImage / Lightbox**: photo treatment and viewer.
- **Plate**: one physical object on a tile (cover, artwork, card image) with `--elevation-object`, a hairline, a fixed ratio and an optional `glow` sampled from its own image at build time.
- **Fan / FanItem**: a stack of objects with a rest, open and touch pose per item; it opens while its host (`data-fan-host`) is hovered or focused within, and the hovered item comes forward while its siblings recede. Reading covers and Writing cards are both a Fan.
- **PostCard**: a small index card for one piece of writing: Plate (banner or typographic fallback), title, headline.
- **Equalizer**: bars in `--accent-graphic` that move only inside `[data-playing]`.
- **IsoGlyph**: a small isometric drawing, top faces first, shaded by `--iso-top/left/right/edge`; a part with `--pose` reacts when its `[data-glyph-host]` row is hovered.
- **set player** (`src/lib/set-player.ts`): the one SoundCloud widget; the room's record player and the Last set tile are two views of it.

### Molecules and tiles
- **Reading** (`src/components/reading/`): `Prose`, `InShort`, `CopyCodeButton`.
- **Site** (`src/components/site/`): `Header`, `Footer`, inside `src/layouts/Shell.astro` (the one shell).
- **Home tiles** (`src/components/home/tiles/`): `BioTile`, `NowTile`, `ReadingTile`, `WritingTile`, `PlacesTile`, `WorkTile`, plus `RoomTile` and `LivingRoom` for the room.
- **Pages** (`src/pages/`, `src/layouts/BlogPost.astro`): compose Shell, Page, PageTitle and tiles. Pages never reach into an atom with `:global(...)`; they use its props.

### The room (signature object)
The live 3D living room sits in a 4:3 tile and opens in a dialog. It is the one dark, lit object on the bench: its tile keeps the paper hairline and fill (so dark mode shows no bright edge round the picture), and its lamp and manuscript colours do not follow the theme. The record player is the SoundCloud set; the bookshelf is the book list.

### Dither treatment
Photos are ordered-dither duotones drawn to a canvas from the container's own colours, so they follow the surface and the theme. `tone="ink"` pairs `--surface-inverse` with `--surface`; `tone="accent"` pairs `--accent` with `--surface`; the darker colour takes the shadows. Hover, keyboard focus or tap reveals the true colours (`reveal="hover"`); `reveal="never"` stays dithered. Without JS the image is an ordinary responsive image.

### Motion
Short and functional: `--duration-fast` 150ms for hover and state, `--duration-base` 250ms for fades such as the dither reveal and for every exit, `--duration-moderate` 350ms with `--ease-spring` for an object opening (a fan spreading, a glyph reacting, staggered 40ms per item), `--duration-slow` 600ms and `--duration-slower` 1200ms for scene and entrance moments. Easing is `--ease-out` for state changes and `--ease-in-out` for larger moves. Under `prefers-reduced-motion: reduce` every duration collapses to near zero and smooth scrolling stops.

## Do's and Don'ts

### Do:
- **Do** separate tiles with space and a 1px `--border` hairline.
- **Do** write atoms against the paper semantic names so they retint on any inverse surface.
- **Do** use `--accent-graphic` for dots and underlines, `--accent` for fills, and only for state.
- **Do** use `Bench`, `Tile`, `Page` and `PageTitle` for layout; add a variant before adding a component.
- **Do** keep every target 44px and keep light and dark identical in structure.
- **Do** keep the essay a single justified, hyphenated 38rem column opening with In short.

### Don't:
- **Don't** add drop shadows, glass, blur or gradients to tiles, controls or pages (the lit room scene is the exception, inside the room).
- **Don't** use orange as decoration or for large areas outside primary buttons and pressed toggles.
- **Don't** introduce a second typeface or colour literals outside `tokens.css` and `theme.css`.
- **Don't** nest links or buttons inside a linked `Tile`.
- **Don't** swap `--surface` and `--surface-inverse` inside the inverse rule.
- **Don't** add a banner, sidebar or card to the essay.
