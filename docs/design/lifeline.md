# Lifeline ("Me in 10s")

Decided 13 November 2026 from a four-way prototype (`/proto/lifeline`, since deleted). It
replaces the fun facts in home’s sidebar, under the heading “Me in 10s”.

## Chosen: Scaled, with Ledger's dots

- Years in a right-aligned mono column (`type-meta`, `text-faint`), a hairline rail
  (`border-strong`), the entry in the body role (`type-body`, `text-muted-foreground`).
- The rail is to scale: before each entry it runs `--spacing-lifeline-year` per year since the
  previous one, so 2001 to 2012 reads as a long stretch and 2024 to 2026 as a short one. The step is
  0.3rem on tall windows and compresses with the window's height (down to 1px), so the pinned
  sidebar fits a 768px laptop without scrolling.
- Dots are 6px in `faint`, not the accent. The accent stays with the name in the intro.
- Entries live in `src/data/lifeline.ts`, in Mathieu's words. The component is
  `src/components/home/lifeline.tsx`.

## Rejected

- **Ledger** on its own: right grammar, but even spacing loses the sense of time passing.
- **Editorial**: the serif reading voice above each entry was warm, but at 17.5px it is louder
  than the intro and taller than the sidebar can afford on a laptop.
- **Now**: the live green dot on the current year and fading past entries put the weight on the
  present, but it read as a status widget, and the past entries in the caption role were too faint
  to enjoy.
- **Blue dots** (Scaled as prototyped): the accent on six dots competed with "Mathieu" for
  attention.
