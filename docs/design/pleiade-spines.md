# Pléiade spines

Every book on the site wears the spine of Gallimard's *Bibliothèque de la
Pléiade*. The style turns a reading list into a recognizable object and keeps
the real title and author readable. We interpret it and never copy
Gallimard's or NRF's marks, logos, or covers.

| Where | Drawn by | How |
|---|---|---|
| Homepage "Bookshelf" row | `PleiadeShelf.astro`, `PleiadeSpine.astro` | HTML and CSS |
| `/bookshelf/` bookcase | `src/components/bookshelf/library/pleiade-paint.ts` | Canvas textures in Three.js |
| Living room bookcase | `scripts/spine-art.mjs` | SVG, baked into `spines.glb` |

All three take colors from `PLEIADE_HEX` in `src/utils/pleiade.ts`, because
copies drifted apart.

## What a Pléiade spine looks like

The reference photograph shows:

- a smooth leather back with slightly rounded corners, lighter at the center
- fine, closely packed gold rules over most of the height
- a plain panel, a quarter to a third of the height, holding the author and
  then the title, centered in capitals on several lines
- a small series mark under the title, if any
- near-constant height, and width that follows page count

## Rules every renderer follows

- **One anatomy.** Only color, width, title, and author change per book.
- **Color comes from the author.** `PLEIADE_AUTHOR_COLORS` gives each author
  one of eight leathers, so one author's books match. The colors are
  screen-tuned starting points, not measurements.
- **Width comes from page count.** Homepage spines are 42 to 90 pixels wide.
  Living room volumes are 237 mm tall and 17 to 47 mm thick.
- **Text stays text** where possible. CSS spines use selectable HTML. Every
  spine's accessible name holds the full title and author.
- **Lettering fits the spine.** Living room lettering is sized to fit each
  spine's proportions. The renderer behind `sharp` ignores SVG's
  `textLength`, so long text would overflow, not shrink.

## How to judge a change

Compare a row of six to ten spines with the photograph:

1. Rule density and panel position read as Pléiade before any title does.
2. The row looks like one editorial collection, not assorted covers.
3. A long title stays inside its spine.
4. All eight colors stay distinct under normal screen light.
5. At 390 pixels wide, no tap target is under 44 pixels.

## References

- [Bibliothèque de la Pléiade, photograph by LPLT](https://commons.wikimedia.org/wiki/File:Biblioth%C3%A8que_de_la_Pl%C3%A9iade.JPG),
  CC BY-SA 3.0, 26 volumes from the front.
- [Gallimard's 2025 Pléiade catalogue](https://www.gallimard.fr/system/files/inline-files/Catalogue-Pleiade-2025.pdf),
  for the collection's naming.
- [petargyurov/virtual-bookshelf](https://github.com/petargyurov/virtual-bookshelf),
  Unlicense. We took its idea of a CSS-gradient spine, not its code.
