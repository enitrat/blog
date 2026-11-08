---
name: Mathieu Saugier
description: The living room by day and by night. Cream paper, walnut-black ink and oxblood in light; the room after dark, lit by brass. A bento of cards on the home page, calm reading pages everywhere else.
colors:
  paper-0: "oklch(0.995 0.003 85)"
  paper-50: "oklch(0.972 0.006 85)"
  paper-100: "oklch(0.945 0.008 80)"
  paper-300: "oklch(0.82 0.012 70)"
  paper-400: "oklch(0.7 0.013 65)"
  paper-500: "oklch(0.58 0.014 60)"
  paper-700: "oklch(0.38 0.014 50)"
  paper-850: "oklch(0.245 0.013 50)"
  paper-900: "oklch(0.205 0.012 50)"
  paper-950: "oklch(0.165 0.01 50)"
  oxblood-100: "oklch(0.93 0.035 25)"
  oxblood-600: "oklch(0.47 0.15 25)"
  brass-300: "oklch(0.82 0.125 78)"
  brass-800: "oklch(0.32 0.06 75)"
  bottle-400: "oklch(0.68 0.09 162)"
  bottle-600: "oklch(0.43 0.075 162)"
  green-500: "#00c758"
  background: "oklch(0.972 0.006 85)"
  foreground: "oklch(0.205 0.012 50)"
  card: "oklch(0.995 0.003 85)"
  muted: "oklch(0.945 0.008 80)"
  muted-foreground: "oklch(0.38 0.014 50)"
  faint: "oklch(0.58 0.014 60)"
  border: "oklch(0.22 0.02 50 / 0.09)"
  border-strong: "oklch(0.22 0.02 50 / 0.16)"
  primary: "oklch(0.47 0.15 25)"
  primary-foreground: "oklch(0.995 0.003 85)"
  primary-soft: "oklch(0.93 0.035 25)"
  secondary: "oklch(0.43 0.075 162)"
  ring: "oklch(0.47 0.15 25)"
  scrim: "oklch(0.165 0.01 50)"
  scrim-foreground: "oklch(0.972 0.006 85)"
  background-dark: "oklch(0.165 0.01 50)"
  foreground-dark: "oklch(0.945 0.008 80)"
  card-dark: "oklch(0.205 0.012 50)"
  muted-dark: "oklch(0.245 0.013 50)"
  muted-foreground-dark: "oklch(0.82 0.012 70)"
  faint-dark: "oklch(0.7 0.013 65)"
  border-dark: "oklch(0.95 0.02 85 / 0.075)"
  border-strong-dark: "oklch(0.95 0.02 85 / 0.14)"
  primary-dark: "oklch(0.82 0.125 78)"
  primary-foreground-dark: "oklch(0.205 0.012 50)"
  primary-soft-dark: "oklch(0.32 0.06 75)"
  secondary-dark: "oklch(0.68 0.09 162)"
  ring-dark: "oklch(0.82 0.125 78)"
typography:
  display:
    fontFamily: "Instrument Serif, ui-serif, Georgia, serif"
    fontSize: "3.5rem"
    fontWeight: 400
    lineHeight: 1
    letterSpacing: "-0.015em"
  headline:
    fontFamily: "Instrument Serif, ui-serif, Georgia, serif"
    fontSize: "2.25rem"
    fontWeight: 400
    lineHeight: 1.1
  title:
    fontFamily: "Instrument Serif, ui-serif, Georgia, serif"
    fontSize: "1.5rem"
    fontWeight: 400
    lineHeight: 1.25
  body:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.625
  label:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.71875rem"
    fontWeight: 500
    lineHeight: "1.25rem"
    letterSpacing: "0.07em"
  meta:
    fontFamily: "Geist Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.33
    letterSpacing: "-0.01em"
rounded:
  card: "1.5rem"
  media: "0.5rem"
spacing:
  inset: "1rem"
  gutter: "0.75rem"
components:
  card:
    backgroundColor: "{colors.card}"
    rounded: "{rounded.card}"
    padding: "{spacing.inset}"
  card-media:
    rounded: "{rounded.media}"
  button-default:
    backgroundColor: "{colors.card}"
    textColor: "{colors.foreground}"
    rounded: "9999px"
    padding: "8px 16px 6px 14px"
    height: "36px"
  button-default-hover:
    backgroundColor: "{colors.muted}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.primary-foreground}"
    rounded: "9999px"
    padding: "8px 16px 6px 14px"
    height: "36px"
  button-glass:
    textColor: "{colors.foreground}"
    rounded: "9999px"
    height: "36px"
  button-ghost:
    textColor: "{colors.muted-foreground}"
    rounded: "9999px"
    height: "36px"
  button-ghost-hover:
    backgroundColor: "{colors.muted}"
    textColor: "{colors.foreground}"
  badge:
    backgroundColor: "{colors.primary-soft}"
    textColor: "{colors.primary}"
    rounded: "9999px"
    padding: "2px 8px"
  eyebrow:
    textColor: "{colors.faint}"
    typography: "{typography.label}"
  list-row:
    textColor: "{colors.foreground}"
    rounded: "0.75rem"
    padding: "16px 12px"
  list-row-hover:
    backgroundColor: "{colors.muted}"
---

# Design System: Mathieu Saugier

## Overview

**Creative North Star: "The Living Room, Day and Night"**

Every colour on the site is an ink sampled from one picture: the render of Mathieu's living room (`src/assets/living-room.png`). By day the room is cream paper, walnut-black ink and an oxblood accent. By night it is the same room after dark, and brass, the lamp, takes over from oxblood. Dark mode is a second lighting of the room. It is not an inverted light mode.

The site has two speeds, and the visual system has two registers to match. The home page is a bento: a grid of cards with fixed-height rows, each card one real object from Mathieu's life (about, latest writing, the last set, reading, work, Elsewhere, and Places, Now and Bench once they have data). The inner pages are calm reading pages. They hold hairline-separated rows and long text, not cards.

Type pairs a serif for display with a grotesk for everything else, and a mono for dates and counts. Motion is springy and comes from one file of presets.

**Key Characteristics:**
- Two token layers: primitive OKLCH ramps that never change with the theme, and semantic roles that pick a primitive per theme. Components name roles only.
- Oxblood accent by day, brass accent by night.
- Instrument Serif for display, Geist for UI and reading, Geist Mono for metadata.
- One card grammar on the home page: one inset, one eyebrow row, media to the edge.
- Depth from a hairline ring folded into a soft shadow, never from a separate border.
- No footer on home; the Elsewhere card holds the contact links.

## Colors

Warm neutrals from the room's paper and wood, one accent per theme, and a night scrim that never lightens.

### Primary
- **Oxblood** (`primary`, oxblood-600): the day accent. Inline links, the name in the About card, icons in the About list, solid primary buttons, focus rings, text selection, the default status dot.
- **Oxblood Wash** (`primary-soft`, oxblood-100): the badge fill by day.
- **Brass** (`primary-dark`, brass-300): the night accent. It takes every role oxblood holds by day. Text on it is dark paper (`primary-foreground-dark`, paper-900).
- **Brass Shadow** (`primary-soft-dark`, brass-800): the badge fill by night.

### Secondary
- **Bottle Green** (`secondary`, bottle-600 by day; bottle-400 by night): a defined role from the room's bottle green. No component uses it yet.

### Neutral
- **Cream Paper** (`background`, paper-50): the page by day.
- **Card Paper** (`card`, paper-0): card, button and code-block surfaces by day.
- **Muted Paper** (`muted`, paper-100): hover fills on rows and ghost buttons, image placeholders, the nav highlight.
- **Walnut Ink** (`foreground`, paper-900): headings and primary text.
- **Muted Ink** (`muted-foreground`, paper-700): descriptions, body prose, secondary text.
- **Faint Ink** (`faint`, paper-500): eyebrows, dates, counts, captions, trailing arrows.
- **Hairline** (`border`) and **Strong Hairline** (`border-strong`): translucent walnut by day and translucent cream by night, so the line takes the colour of the surface under it. `border-strong` marks the lifted state and prose table and quote rules.
- **Night Room** (`background-dark`, paper-950), **Night Card** (`card-dark`, paper-900), **Night Muted** (`muted-dark`, paper-850): the same roles after dark, on the same paper ramp read from the dark end.
- **Night Scrim** (`scrim`, paper-950, with `scrim-foreground`, paper-50): the overlay behind photos, the writing card's banner and the lightbox.
- **Live Green** (green-500): the live ping of the status dot, and nothing else.

### Named Rules
**The Roles Only Rule.** Components name roles (`bg-card`, `text-faint`, `border-border`, `text-primary`), never a ramp step and never a `dark:` colour variant. `src/lib/design-tokens.test.ts` fails the build otherwise. Retint the site by editing a ramp; re-role it by editing the `:root` and `.dark` blocks.

**The Lamp Rule.** The accent changes hue with the light: oxblood by day, brass by night. Never carry oxblood into dark mode or brass into light mode.

**The Night Scrim Rule.** The scrim is night in both themes. `.dark` does not re-role it, so text over a photo is always light on dark.

**The Dither Rule.** Place photographs are drawn as an ordered-dither duotone on a canvas. Its two inks come from `--dither-ink` and `--dither-paper` in `:root` and `.dark` (paper-850 on paper-50 by day, paper-900 on paper-300 by night), so every photo, whatever its source, reads as one set.

## Typography

**Display Font:** Instrument Serif (with ui-serif, Georgia), weight 400 only, roman and italic
**Body Font:** Geist (with ui-sans-serif, system-ui)
**Label/Mono Font:** Geist Mono (with ui-monospace, Menlo)

**Character:** A tall, narrow serif sets names and titles; a neutral grotesk carries the interface and the reading; a mono sets anything that is a number or a date.

### Hierarchy
- **Display** (Instrument Serif 400, 3.5rem, line-height 1, tracking -0.015em): the greeting in the About card from 1100px up. It steps down to 2.25rem below that.
- **Headline** (Instrument Serif 400, 2.25rem, 3rem from 640px): the page title on every inner page, essays and Field Reports included.
- **Title** (Instrument Serif 400, 1.5rem, leading tight): essay titles in the writing list.
- **Body** (Geist 400, 0.875rem, line-height 1.625): UI text, row titles (at 600), descriptions. Essays use Tailwind Typography at `prose-sm`, `prose-base` from 640px, inside a 36rem column.
- **Label** (Geist 500, 0.71875rem, line-height 1.25rem, tracking 0.07em, uppercase): the eyebrow. Card headers and section headings on inner pages.
- **Meta** (Geist Mono 400, 0.75rem, tracking -0.01em, tabular figures): dates, durations, years, the In short numbering.

### Named Rules
**The One Serif Weight Rule.** Instrument Serif is loaded at 400 only. Hierarchy in the serif comes from size, never weight.

**The Serif for Names Rule.** The serif sets page titles, essay titles, the greeting and a company's initial when its logo is missing. Interface text, labels and running prose stay in Geist.

## Layout

The shell is one centred column, at most 90rem wide, with 1rem side padding (2rem from 640px). A sticky header sits 0.75rem from the top.

**The home bento.** A dense grid with a 0.75rem gutter (`gap-gutter`) and fixed-height rows, so the grid keeps its proportions on any window and spare height sits below it. Phones: 2 columns of 11rem rows. From 640px: 4 columns of 10rem rows. From 1100px (68.75rem): 8 columns of 12.125rem rows, which fill a 1440 by 900 window under the header. About takes 2 by 2, Writing 6 by 2, Set 4 by 1, Reading 2 by 2, Elsewhere 2 by 1, Work 6 by 1. Places, Now and Bench share a fifth row at widths 3 : 2 : 1, and only when they have data.

**Inner pages.** One column with a 2.5rem rhythm between blocks, 2rem above and 4rem below. Three widths: 36rem for reading (essays, Field Reports, About), 48rem for lists (writing, work), 80rem for wide pages (reading, places).

**The Fixed Row Rule.** Bento rows have fixed heights. A card fits its cell; it never grows the row.

## Elevation & Depth

Depth is a hairline and a soft, warm shadow folded into one value. Cards carry no separate border: the first layer of every elevation is a 1px ring in the border colour. By day the shadows are tinted walnut. By night they are black, with a faint top highlight, and the lifted state adds a brass glow, the lamp catching the card.

### Shadow Vocabulary
- **Card** (`shadow-card`): every card, default and glass buttons, the nav pill, code blocks, banner thumbnails. Hairline plus a 1 to 2px contact shadow.
- **Raised** (`shadow-raised`): solid primary buttons and dialogs. Adds a 12 to 16px soft drop.
- **Lifted** (`shadow-lifted`): a linked home card under the pointer. Strong hairline and a 24 to 28px drop; at night, a brass glow.

Exact values for both themes live in `.impeccable/design.json`.

### Named Rules
**The Hairline in the Shadow Rule.** Never add `border` to a card. The ring is part of the shadow, so a card has one edge in both themes.

**The Glass Rule.** Glass is the card colour at 60% with a 4px blur and 150% saturation, used for the nav pill, the glass button and carousel arrows. Dialogs are the card at 80% with an 8px blur.

## Shapes

Corners are generous on cards and tighten as they nest. A card is 1.5rem (`rounded-card`). Media inside a card is the card radius minus the inset (`rounded-media`, 0.5rem), so the inner curve stays concentric with the outer one. Buttons, the nav, badges and chips are full pills. Interactive rows on inner pages round their hover fill at 0.75rem and bleed 0.75rem past the column on each side, so text stays aligned while the hover reads as a soft plate. Company logos are rounded squares: 0.5rem at 44px in rows, 1rem at 64px in Field Report headers, each with a hairline.

**The Nested Radius Rule.** Inner radius equals outer radius minus the gap between them. Change `--radius-card` or `--spacing-inset` and media follows.

## Components

### Buttons
- **Shape:** full pill, 36px tall (28px small), with optically weighted padding: 8px top, 16px right, 6px bottom, 14px left. Icon buttons are 36px or 28px squares.
- **Default:** card fill, foreground text, semibold, card shadow; hover fills with muted.
- **Primary:** accent fill, primary-foreground text, raised shadow; hover at 90%.
- **Glass:** card at 60% with blur and saturation (see the Glass Rule); for chrome over images and sticky bars.
- **Ghost:** muted-foreground text; hover fills with muted and darkens the text.
- **Focus:** a 2px ring in the ring role, offset 2px from the background, on every interactive element.

### Cards (home)
- **Corner Style:** 1.5rem.
- **Background:** card role. Photo cards put the image full-bleed under a night scrim.
- **Shadow Strategy:** card shadow at rest; a linked card rises 3px on the hover spring and takes the lifted shadow, then presses on tap.
- **Internal Padding:** one inset, 1rem.
- **Entry:** cards pop in on mount (fade, 14px rise, 0.98 scale) on the enter spring, 45ms apart.

**The One Inset, One Eyebrow Rule.** Every home card opens with the same 20px header row: a 14px icon, the uppercase label that names the card, and an optional trailing action at 0.8125rem. Content follows at the one inset. Media runs to the card edge or sits inside it at the media radius.

### Eyebrow
The uppercase label (label typography, faint ink). It names a card, heads a section on an inner page, or holds real metadata above a page title (byline and date on an essay, role and period on a Field Report). It can carry a status dot, an icon and right-aligned meta.

### Badge
A small non-interactive pill in primary-soft with primary text, 0.75rem medium, for flags such as "Reading" and "Latest".

### List Row
The inner-page unit. Rows are separated by a top hairline, with a date column on the left (8rem wide in plain rows, 6rem in writing rows), a semibold title, and a muted description. A linked row fills with muted on hover and slides a trailing arrow in. Writing rows set the title in the serif and carry the essay's banner as a 14rem by 8.75rem thumbnail at the media radius; the row lifts 2px and the banner zooms on the hover spring. Work rows lead with the company logo, which pops on hover.

### Navigation
A glass pill of text links beside the name in the header. A muted highlight follows the pointer between links on the layout spring. The current page is foreground; others are muted-foreground and darken on hover. A theme toggle sits beside it.

### In short
At the top of an essay: an eyebrow, then a numbered list (mono numerals) of section titles with each section's first sentence, separated from the essay by a hairline. A title underlines in the accent on hover.

### Book covers and spines
On the reading page, the current book and the top-rated books show their real covers. The archive is drawn as piles of spines.

### Motion
Every spring comes from `src/lib/motion.ts`; none is inlined. Hover scales are 1.02 for surfaces and 1.08 for small things such as logos and icons. Under reduced motion, state still changes and nothing moves.

## Do's and Don'ts

### Do:
- **Do** name roles in components, and add a role before reaching for a ramp.
- **Do** give every home card the one inset and the one eyebrow row.
- **Do** derive nested radii from `rounded-card` minus `p-inset`.
- **Do** use the serif at 400 for page titles and essay titles, Geist for everything else, and Geist Mono with tabular figures for dates and counts.
- **Do** keep inner pages as hairline rows on the page background.
- **Do** put the scrim behind any text set on a photograph.
- **Do** show a 2px ring-role focus ring on every interactive element.

### Don't:
- **Don't** use a ramp step or a `dark:` colour variant in a component class.
- **Don't** add a border to a card; the hairline lives in the shadow.
- **Don't** wrap inner-page content in cards.
- **Don't** set the serif bold, or set UI text in the serif.
- **Don't** put an invented label above a page title. That slot holds real metadata only.
- **Don't** let a bento card grow its row.
- **Don't** inline a spring.
