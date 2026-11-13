# Reading: piles and the dock

Decided 13 November 2026, over six rounds of live prototypes on real books, covers and notes.

## Chosen

- **Piles, not a list.** Every shelved book is a cover in a pile. Piles sort by author by
  default, and by length, rating or year on request. The books being read open the author sort,
  and authors read once share one "One book each" pile.
- **Sorting** uses the header nav's grammar: muted words, and a 2.5px accent underline that
  slides under the current one, after a "Piled by" label.
- **Re-sorting deals.** Each cover travels to its new pile, staggered by 10ms and capped at 20
  covers, so the last one leaves within 0.2s.
- **Arrival is quiet.** Piles fade in and rise 8px, staggered by 25ms. The page settles in about
  0.4s.
- **Holding a pile** fans it like a hand of cards and opens the **dock** at the bottom of the
  screen. The dock names the book (title, author, date, pages, rating) and
  holds the pile as a hand of at most nine cards. Pointing at a card lifts it, and the same book
  rises in the pile. A sideways trackpad scroll over the dock, or a sideways swipe across the hand,
  turns it one book per 40px; on a pile bigger than the hand, the hand re-centres, so the arc turns
  like a dial. Clicking the lifted card turns it over to show its back and note.
- **The dock holds.** Leaving a pile gives 450ms to reach the dock. Leaving the dock closes it
  after 250ms. Another pile takes the dock only after the pointer rests on it for 140ms, so
  crossing piles on the way down does not hijack it. Escape, or a tap anywhere else, closes it.
- **The rail** reaches big piles. Under a hand of more than nine, a strip made of a sliver of
  every cover jumps to any book in one sweep and re-centres the hand. The hand itself moves only
  when the chosen book leaves it, so it never slides under the pointer.
- On a phone, a tap opens the pile and the dock. The hand's arc and cards scale to the dock's
  width.

Code: `src/lib/piles.ts` (the piles for each sort, with tests), `src/components/books/`
(`pile-table`, `fan-pile`, `pile-dock`, `book-cover`).

## Rejected

- **The old page** (a year-by-year list with a strip per year): read as a list, not a collection.
- **Pléiade shelves** (spines by year, pulled out to a card): titles unreadable on a spine; did
  not hold up on a website.
- **Card catalogue** (stamped index cards): legible but no covers, and covers are the point.
- **Desk** (covers scattered per year, draggable): liked for its spirit, but tied to dates and a
  heap on big years. It led to the piles.
- **Sorter, Poster, Drawer, Cascade**: a tidy grid, a gallery wall, an inline tray and uneven
  towers. All lost to the fanned pile.
- **The heap intro** (all covers in one heap, then dealt): too loud and too long.
- **A floating, filled sort control**: read as app chrome. Also rejected: an inline sentence,
  a title menu, and a re-deal button.
- **Loupe alternatives**: lifting the cover in place, or captioning it under the pile, left
  buried covers hard to see. Other docks: a sliding strip (click to move), a magnifying row, a
  detail card, a hand of every book (did not scale), a wheel (hard to navigate), and side decks
  (too slow to reach book 40 of 45).
