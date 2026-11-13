# Wordmark: Sticker

Decided 13 November 2026, from a logo lab of eleven directions and then a live tuner.

## Chosen

- **Sticker.** Titan One, "MSAUG" stacked over a sparkle and "DEV": white letters in a fat
  outline, with a hard shadow, tilted like a sticker slapped on. It replaces the serif M tile and
  "Mathieu Saugier" in the header. (The home monogram it left alone has since become a photo, `src/assets/portrait.jpg`.)
- **Favicon.** The M alone, as its own sticker (`StickerMark`), since the two-line wordmark
  cannot be read at 16px; compared against the sparkle and the full wordmark at 16, 32 and 64px
  on light and dark tab bars. It replaced the serif M tile on 13 November 2026.
- **Tuner settings**, in em of the font size: indent 1.75, line gap −0.06, tracking −0.02, tilt
  −8°, slant −4.5°, outline 0.09 on each side, shadow 0.075 down and 0.0525 right, no die-cut
  border.
- **Colours** are the sticker roles in `src/app/globals.css`: fill neutral-0, outline accent-600,
  shadow accent-800. They are the same in both themes, like a printed sticker. The tuner used
  oklch(0.53 0.145 258) for the outline and oklch(0.32 0.109 258) for the shadow; the nearest
  palette steps replaced them.

## Where it lives

- `src/components/site/wordmark.tsx` draws the outlined paths, so the site loads no Titan One.
- `docs/design/wordmark.svg` is the same mark as a standalone file, with the tuner's colours.
- Paths were outlined with opentype.js from Google Fonts' `TitanOne-Regular.ttf` at 1em = 100
  units, with line-height 1 and the font's hhea metrics, to match the tuner's CSS.

## Rejected

Gumdrop (Modak), Bounce (Chewy), Inkwell (Shrikhand), Stamp, Keycaps, Blob, Wave (Bagel Fat One),
Echo (Rammetto One), Prompt (Fragment Mono) and House (Instrument Sans with the MS tile).
