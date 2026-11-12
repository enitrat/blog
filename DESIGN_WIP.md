# Home page: design decisions (work in progress)

This file records how the home page was redesigned in November 2026: what Mathieu liked, what he
rejected, and what he used as a reference. Read it before changing the home page. The tokens it
names live in `src/app/globals.css`.

## The brief

Tech people are the main visitors. Most of them arrive because of Mathieu's work. The home page
should show them the rest of him too: DJing, training, reading, writing. A resume of links and short
text is what this page must not become.

The page itself is the proof of craft. Visitors judge the work by how well the site is made, so
every part of it shares one visual language, as if one person made all of it.

## The reference: ana.sh

Ana Howard's site was the starting point. These are the parts Mathieu pointed to, in design terms:

| What he noticed | Name | What we kept |
| --- | --- | --- |
| A thick light border with grey inside, which reads as raised | Bezel: a thick frame around a nested well | Every home tile |
| The grey of the page bleeding into an image | Edge fade, done with `mask-image` | The Writing banner, the Morph preview |
| Light behind an image, in its own colours | Ambient glow | The set art, the book covers, the room render |
| Blog card: text, an image, then "Read more" | Split card with an explicit call to action | The Writing tile |
| Clothes and bits that scroll by themselves | Auto-advancing carousel | Writing cycles through the last four posts |
| Products kept in a folder | Folder stack leading to a portfolio | The Work tile, which opens `/work` |
| A little motion on the cards | Hover lift and press feedback | `BentoTile`, with springs from `src/lib/motion.ts` |
| An About column taking the whole left side | Sticky sidebar, full height | `HomeSidebar` |

## What we decided

- **Layout.** A sidebar on the left (mark, nav, About, fun facts, a picture) beside a bento on the
  right. From 64rem the sidebar is full height and sticky, so it stays put while the page scrolls.
  The nav lives in the sidebar, so the home page has no top bar.
- **Hierarchy.** Writing is the largest tile. Last set, Reading, Work, Elsewhere and Project share
  the rest. No tile competes with Writing for attention.
- **Fill.** From 64rem the bento fills one screen in three rows. Each row is clamped between 13rem
  and 17rem: a tall monitor leaves space under the grid, and a short window never crushes a tile.
- **Surface.** The bezel. A card-coloured frame (`p-bezel`, 5px) around a grey well (`bg-well`).
  The well's radius nests: `--radius-well` is `--radius-bezel` (22px) minus the frame.
- **Elevation.** A 1px ring, a tight contact shadow and a soft ambient shadow
  (`--elevation-bezel`), set where the prototype's elevation slider sat at 0.25. Shadows don't show
  on a dark page, so in dark mode the bezel drops them and keeps the ring plus a 1px top highlight.
- **The well colour** is `--muted` mixed 75% into `--card`, in oklab. Never mix it in oklch: an
  oklch mix of near-greys drops the hue, and the leftover chroma renders pink.
- **Images** keep their own colours. Each one gets a 1px inset outline (`shadow-image-edge`) so its
  edge holds against the well: pure black at 10% by day, pure white at 10% by night, never tinted.
- **Ambient glow** is a copy of the image behind it, blurred 22px, saturated to 1.7, scaled to 108%
  and set at 75% opacity (`ambient-glow`). The playing record pushes it to 130% and 90%.
- **Gap** between tiles is 15px (`gap-bento`).
- **Motion.** Tiles pop in once on load, with a 45ms stagger. After that, all motion follows one
  rule, described below.

## Motion: one rule for every tile

A tile is **held** while a mouse or trackpad hovers it, or while keyboard focus is inside it. A tap
on a touch screen does not hold a tile. Holding wakes the tile in three layers, and each layer
always moves the same way:

| Layer | What moves | How |
| --- | --- | --- |
| Surface | The bezel rises 2px, and a deeper shadow (`shadow-bezel-raised`) fades in over the resting one | `spring.hover`. The shadow change is an opacity cross-fade; `box-shadow` is never animated. Tiles that are one big link (Writing, Reading, Project) also press in on tap. |
| Object | The tile's one signature object | `spring.fan`, with small movements: at most about a fifth of the object's width, 8° of rotation, 1.04× scale |
| Affordance | Arrows, chevrons, titles and glows | CSS, 150ms ease-out; glows take 300ms |

The signature object of each tile:

| Tile | What happens when it's held |
| --- | --- |
| Writing | The banner slowly pushes in (1.04×, 500ms), and the title turns to the accent colour |
| Last set | The record slides a fifth of the way out of its sleeve, and the sleeve tips back 3°. While playing, it becomes the spinning record instead. |
| Reading | The two covers fan apart |
| Work | The sheets rise out of the folder |
| Elsewhere | The icons rise in a ripple, left to right; the one under the pointer rises and tilts further |
| Project | The live Morph preview scrolls up, as if someone were reading the page |

The two slower movements, the banner (500ms) and the preview scroll (500ms), break the rule that
UI motion stays under 300ms. Both cover a large distance and run on images, not controls, so a
faster move would read as a jump.

In the sidebar, the nav pills press in to 96% on tap, and the room render's glow brightens on
hover. Every spring comes from `src/lib/motion.ts`. Under reduced motion nothing moves, and the
Writing carousel never advances on its own.

These values came from live sliders in the prototype, where Mathieu's final settings were:
gap 15px, radius 22px, elevation 0.25, bezel 5px, glow 0.75.

## What we rejected, and why

- **A duotone tint on every image**, to make covers and photos match the palette. Mathieu disliked
  it. Covers with type and UI screenshots also lose legibility under it. If images need a shared
  treatment later, the candidate is a real shader (see paper.design shaders), tested first on the
  five worst assets.
- **A hero layout** with a large display headline over the bento. The headline was far too big.
- **An index layout** with no cards, just type on hairlines. There was no focal point, so the eye
  had nowhere to land.
- **The old bento**, with nine near-equal tiles and full-bleed covers and banners. It read as
  aggressive: oversized media and no hierarchy.
- **Flat cards** with layered shadows, tested against the bezel. The bezel won.
- **A fixed row height** where the page scrolls past one screen. It differed from the
  fill-one-screen version only when the window height changed, so we replaced both with the clamp.

## Open items

- **Portrait.** The sidebar shows an `MS` monogram until Mathieu supplies a photo.
- **Sidebar picture.** It shows the Blender room render (`src/assets/living-room.png`). It could be
  replaced by a photo of Mathieu at the decks, in the gym, or travelling.
- **Contact line.** Ana's sidebar ends with a sentence on how to reach her. Mathieu has not written
  one; until he does, the Elsewhere tile carries the links.
- **New labels** waiting for Mathieu's approval: "Fun facts", "Project", "Read more", and the
  caption "The room, built in Blender".
- **Morph** is not in `src/content/work` yet. The Project tile reads its name and URL from
  `PROJECT` in `src/consts.ts`.
- **Work page.** The Work folder should lead to a portfolio-style `/work`, more focused on
  engineering than the current list.
- **The headline claim.** The About copy says "An engineer working in DeFi". If the site is meant
  to prove design-engineering craft, that line may need to change. This is Mathieu's call.
