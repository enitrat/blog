# Pléiade spines

Every book on the site wears the spine of the *Bibliothèque de la Pléiade*,
Gallimard's leather-bound classics collection. The style turns a reading list
into an object people recognize, and it keeps the real title and author
readable. We interpret the style. We never copy Gallimard's or NRF's marks,
logos, or covers.

Three places draw spines:

| Where | Drawn by | How |
|---|---|---|
| Homepage "Bookshelf" row | `PleiadeShelf.astro`, `PleiadeSpine.astro` | HTML and CSS |
| `/bookshelf/` bookcase | `src/components/bookshelf/library/pleiade-paint.ts` | Canvas textures in Three.js |
| Living room bookcase | `scripts/spine-art.mjs` | SVG, baked into `spines.glb` |

All three take their colors from `PLEIADE_HEX` in `src/utils/pleiade.ts`. When
the palette was duplicated, the copies drifted apart.

## What a Pléiade spine looks like

These traits come from a reference photograph of 26 real volumes:

- A smooth leather back with slightly rounded corners. The curve makes the
  center lighter than the edges.
- Fine horizontal gold rules, packed close, over most of the height.
- A plain panel that interrupts the rules, about a quarter to a third of the
  height. The author and then the title sit in it, centered, in capitals, on
  several lines.
- A small mark under the title when the volume belongs to a series.
- Almost no variation in height. The width varies with the page count.

## The rules every renderer follows

- **One anatomy.** Color, width, title, and author are the only values that
  change from book to book.
- **Color comes from the author.** `PLEIADE_AUTHOR_COLORS` gives each author
  one of eight leathers, so two books by the same author always match. The
  colors are screen-tuned starting points, not measurements of real leather.
- **Width comes from the page count.** On the homepage, spines run from 42 to
  90 pixels wide. In the living room, every volume is 237 mm tall and 17 to 47
  mm thick.
- **Text stays text** wherever the renderer allows it. The CSS spines use real,
  selectable HTML, and every spine's accessible name holds the full title and
  author.
- **Lettering fits the spine.** The living room's artwork is drawn at each
  spine's own proportions and sized to fit. The renderer behind `sharp`
  ignores SVG's `textLength`, so text too long for the spine would run off the
  leather rather than shrink.

## How to judge a change

Put a row of six to ten spines next to the reference photograph, then check:

1. The rule density and the panel position are recognizable before you read a
   title.
2. The row looks like one editorial collection, not an assortment of covers.
3. A long title stays inside its spine.
4. All eight colors stay distinct under normal screen light.
5. At 390 pixels wide, no tap target falls under 44 pixels.

## References

- [Bibliothèque de la Pléiade, photograph by LPLT](https://commons.wikimedia.org/wiki/File:Biblioth%C3%A8que_de_la_Pl%C3%A9iade.JPG),
  CC BY-SA 3.0. It shows 26 real volumes from the front.
- [Gallimard's 2025 Pléiade catalogue](https://www.gallimard.fr/system/files/inline-files/Catalogue-Pleiade-2025.pdf),
  for the collection's naming.
- [petargyurov/virtual-bookshelf](https://github.com/petargyurov/virtual-bookshelf),
  Unlicense. Its idea of building a spine from CSS gradients shaped the CSS
  spines. We took the idea, not the code.
