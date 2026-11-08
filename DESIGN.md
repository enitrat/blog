---
name: Mathieu Saugier
description: A living desk of small springy cards on cool gray paper, Figtree type, an iron dark mode, and violet for the brand.
colors:
  gray-25: "#fcfcfd"
  gray-50: "#f9fafb"
  gray-100: "#f2f4f7"
  gray-200: "#eaecf0"
  gray-300: "#d0d5dd"
  gray-400: "#98a2b3"
  gray-500: "#667085"
  gray-600: "#475467"
  gray-700: "#344054"
  gray-800: "#182230"
  gray-900: "#101828"
  iron-100: "#f4f4f5"
  iron-200: "#e4e4e7"
  iron-300: "#d1d1d6"
  iron-400: "#a0a0ab"
  iron-500: "#70707b"
  iron-600: "#51525c"
  iron-700: "#3f3f46"
  iron-800: "#27272a"
  iron-900: "#1a1a1e"
  iron-950: "#131316"
  iron-fade: "#17171a"
  brand-200: "#d9d6fe"
  brand-300: "#bdb4fe"
  brand-400: "#9b8afb"
  brand-500: "#7a5af8"
  brand-600: "#6938ef"
  brand-700: "#5925dc"
  brand-800: "#4a1fb8"
  brand-900: "#3e1c96"
  red-500: "#f04438"
  green-500: "#00c758"
  white: "#ffffff"
typography:
  title:
    fontFamily: "Figtree, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "-0.025em"
  heading:
    fontFamily: "Figtree, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 600
    lineHeight: 1.55
    letterSpacing: "-0.025em"
  body:
    fontFamily: "Figtree, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.625
    letterSpacing: "-0.025em"
  label:
    fontFamily: "Figtree, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.33
    letterSpacing: "-0.025em"
  code:
    fontFamily: "Geist Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.7
rounded:
  xs: "2px"
  sm: "4px"
  md: "6px"
  lg: "8px"
  xl: "12px"
  2xl: "16px"
  full: "9999px"
spacing:
  0.5: "2px"
  1: "4px"
  1.5: "6px"
  2: "8px"
  4: "16px"
components:
  button-default:
    backgroundColor: "{colors.white}"
    textColor: "{colors.gray-800}"
    rounded: "{rounded.full}"
    padding: "8px 16px 6px 14px"
    height: "36px"
  button-default-hover:
    backgroundColor: "{colors.gray-100}"
  button-brand:
    backgroundColor: "{colors.brand-500}"
    textColor: "{colors.white}"
    rounded: "{rounded.full}"
    padding: "8px 16px 6px 14px"
    height: "36px"
  button-glass:
    textColor: "{colors.gray-800}"
    rounded: "{rounded.full}"
    height: "36px"
  card:
    backgroundColor: "{colors.white}"
    rounded: "{rounded.xl}"
    padding: "16px"
  card-dark:
    backgroundColor: "{colors.iron-900}"
    rounded: "{rounded.xl}"
    padding: "16px"
  nav-pill:
    textColor: "{colors.gray-500}"
    rounded: "{rounded.full}"
    padding: "6px 12px"
  badge:
    textColor: "{colors.brand-600}"
    rounded: "{rounded.full}"
    padding: "2px 8px"
---

# Design System: Mathieu Saugier

## Overview

**Creative North Star: "The Living Desk"**

The site is a tidy desk of small, live objects from Mathieu's life: the last DJ set, the essay on top of the pile, the book being read, the work, the places. Each one is a white card with a hairline and the faintest shadow, laid on cool gray paper. Nothing on the desk is loud at rest. The personality arrives when a hand reaches for it: cards spring up a little, small things pop, decks fan open, and objects settle back with a light overshoot.

The system is restrained in colour, size and weight, and generous in motion. Type is one family, Figtree, set small and tight. Violet carries the brand: links, emphasis, badges, selection, and the occasional gradient tile. Dark mode is not an inverted light mode; it is a separate desk in neutral iron, with violet stepping one shade lighter so it keeps its weight.

Glass is thin. Chrome that floats over content (the nav pill, dialogs, composer-style bars) is a translucent fill with a low-alpha hairline, a small 4 to 8px blur and 150% saturation. Saturation, not blur, is what makes it read as glass.

**Key Characteristics:**
- Cool, blue-tinted gray paper in light mode; a separate neutral iron ramp in dark mode.
- White cards, 1px hairlines, and `shadow-xs` as the signature depth.
- Figtree throughout, text-sm as the workhorse, tracking-tight, medium and semibold for emphasis.
- Violet as the brand voice, in light and dark.
- Personality lives in spring motion, not in colour or size.
- Radius shrinks as nesting deepens.

## Colors

A cool gray paper desk by day, a neutral iron desk by night, and one violet that carries the brand across both.

### Primary
- **Desk Violet** (brand-500 / brand-600): The brand. Links and their hover step (600 → 700 in light, 400 → 300 in dark), badges, the text selection, focus rings, and brand gradient tiles (brand-300 → 400 or 400 → 500).
- **Deep Violet** (brand-700 to brand-900): Pressed and hovered brand text, selection text, and the bottom of gradients.
- **Pale Violet** (brand-200 / brand-300): Selection fill in light mode and the top of light gradient tiles.

### Neutral
- **Paper** (gray-50): The light page background, and the colour edge fades dissolve into.
- **Card White** (white): Card and button surfaces in light mode.
- **Hairline Gray** (gray-200 / gray-100): Borders and dividers in light mode, at full strength or 40 to 80% alpha.
- **Ink Steps** (gray-400 → gray-900): Text, from muted captions (400) through secondary text (500, 600) to headings (800, 900).
- **Iron Night** (iron-950): The dark page background, fading into iron-fade at edges.
- **Iron Card** (iron-900): Card surfaces in dark mode.
- **Iron Steps** (iron-400 → iron-100): Dark-mode text. Body prose is iron-300; muted captions iron-400; headings iron-100 or white.
- **Signal Red / Live Green** (red-500, green-500): Errors and live-status dots only.

### Named Rules
**The Two Desks Rule.** Dark mode uses the iron ramp, never an inverted gray. Light gray-600 text becomes iron-300, not gray-400.

**The Lighter Violet Rule.** Wherever light mode uses brand-600 or brand-700 for emphasis, dark mode uses brand-500 or brand-400.

**The Dark Hairline Rule.** Dark borders are white/10, or iron-800 at 15 to 60% alpha. Never a solid light border on iron.

## Typography

**Body Font:** Figtree (with ui-sans-serif, system-ui)
**Code Font:** Geist Mono (with ui-monospace, Menlo)

**Character:** One geometric grotesk with a friendly, rounded voice, set small, tight and antialiased. Hierarchy comes from weight and gray step, not from size jumps.

### Hierarchy
- **Title** (600, 1.25rem): Page titles. Used once per page.
- **Heading** (600, 1.125rem): Section headings in prose and large card titles.
- **Body** (400, 0.875rem, line-height 1.625): The workhorse for UI and card text. Prose uses leading-relaxed and `text-wrap: pretty`, measured to max-w-xl.
- **Label** (400, 0.75rem): Card eyebrows, dates, metadata, counts.
- **Code** (Geist Mono 400, 0.8125rem): Code blocks and inline code only.

### Named Rules
**The Small Type Rule.** text-sm is the default. Reach for text-lg or text-xl only for page and section titles.

**The Two Weights Rule.** Emphasis is medium or semibold. Regular is for running text only; bold is not used.

## Layout

Spacing runs on a 4px base. gap-1.5 (6px) is the default rhythm between cards and controls; gap-1, gap-2, gap-4 and gap-0.5 follow. Containers are max-w-xl (36rem) for reading, max-w-3xl (48rem) for lists, and max-w-7xl (80rem) for the shell.

The home page is one viewport on desktop. A sidebar on the left holds the introduction; a dense grid of small cards fills the rest, sized so nothing scrolls from 1024px up. On phones the sidebar and cards stack in one column and the page scrolls.

Pills use asymmetric, optically weighted padding: tighter at the bottom and left, looser at the top and right (8px 16px 6px 14px).

## Elevation & Depth

Depth is quiet and mostly structural. Cards sit on the paper with a hairline and `shadow-xs`; motion, not shadow, signals that something can be picked up.

### Shadow Vocabulary
- **Signature** (`box-shadow: 0 1px 2px rgb(16 24 40 / 0.05)`): Every card, button and composer bar. The default.
- **Stacked** (`shadow-sm` tinted zinc-800/15): Cards stacked on cards, such as decks and book piles.
- **Brand lift** (`shadow-md`): Brand gradient tiles only.
- **Image glow**: A duplicate of the image behind it, blurred 16px, saturated ×6, at 70% opacity and shifted 8px down; in dark mode 15% opacity and saturation ×12.

### Named Rules
**The Faint Shadow Rule.** shadow-xs is the ceiling for ordinary surfaces. Larger shadows belong to brand tiles and stacked objects only.

**The Saturation Glass Rule.** Glass is 50 to 80% fill, a low-alpha hairline, 4 to 8px blur and backdrop-saturate-150. Never a heavy blur.

## Shapes

Corners are soft and get tighter as nesting deepens: pills and avatars are fully round, cards and media frames are 12px, chat bubbles and glow tiles 16px, mini cards 6px, and the lowest piece of a stack 2px at the bottom. Media frames carry a translucent hairline (iron-950/10). Edge fades are three-stop and eased (from paper, through paper/50 at 75%, to transparent), never a plain two-stop fade.

## Components

### Buttons
- **Shape:** Full pill (9999px), 36px tall; 28px in the small size.
- **Default:** White fill, gray-200 hairline, shadow-xs, gray-800 semibold text; hover lifts to gray-100/50. Dark: iron-900 on a white/10 hairline.
- **Brand:** Violet gradient (brand-400 → 500) with white text and shadow-md.
- **Glass:** white/50 fill, gray-200/50 hairline, 4px blur, saturate 150; dark iron-950/65 on iron-600/25.
- **Ghost:** Text only, with a gray-100 (dark iron-800) fill on hover.
- **Focus:** A 2px brand-500/60 ring with an offset, on every interactive element.

### Cards / Containers
- **Corner Style:** 12px.
- **Background:** White in light mode, iron-900 in dark mode.
- **Shadow Strategy:** Signature shadow-xs (see Elevation & Depth).
- **Border:** 1px gray-200, or white/10 in dark mode.
- **Internal Padding:** 16px, with gap-4 between the eyebrow header and the body.
- **Motion:** A linked card springs to scale 1.02 on hover (stiffness 450, damping 18); cards pop in on load with a short stagger.

### Navigation
- **Style:** A light glass pill holding text links. A gray-100 (dark iron-800) highlight follows the pointer between links on a layout spring (400 / 40).
- **States:** Links step one gray darker on hover; the current page is gray-900 (dark white).
- **Mobile:** The same pill with tighter link padding.

### Badge
- **Style:** A violet tint pill (brand-500/10 fill, brand-600 text; dark brand-400/15 and brand-400), text-xs, for flags such as "Latest".

### Live Dot
- **Style:** A 6px green-500 dot with the `ping` keyframe ring behind it, the only CSS keyframe in the system. Hidden ring under reduced motion.

### Deck
- **Character:** A stack of mini cards that fans open like a hand when reached for, on the fan spring (stiffness 320, damping 18, 33.5ms stagger per card); the held card rises and its caption pops in.

## Do's and Don'ts

### Do:
- **Do** use shadow-xs and a 1px hairline for every card.
- **Do** take every spring from `src/lib/motion.ts`: hover 1.02 on large surfaces, 1.08 on small ones, tap 0.95, exit to y −30 and scale 0.9.
- **Do** keep stiffness between 250 and 550 and damping between 12 and 40; bounce stays between 0.1 and 0.25.
- **Do** step text links one gray darker on hover in light mode and one iron lighter in dark mode, with `transition-colors` (150ms).
- **Do** set prose links underlined at a 22% offset, with the decoration colour shifting on hover.
- **Do** give every interactive element a visible focus-visible ring.
- **Do** respect reduced motion: state still changes, nothing springs or translates.

### Don't:
- **Don't** invert gray for dark mode; use iron.
- **Don't** use heavy blur for glass; saturation carries it.
- **Don't** use shadow-md outside brand gradient tiles.
- **Don't** add CSS keyframes beyond `ping`; motion belongs to springs.
- **Don't** set type above text-xl, or use bold.
- **Don't** remove focus outlines without a replacement ring.
