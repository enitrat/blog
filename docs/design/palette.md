# Palette: Ink

Decided 11 November 2026, from a live picker over every page, in both themes.

## Chosen

- **Ink.** Cool, near-monochrome neutrals (hue 250, chroma 0.006 to 0.016) and one fountain-pen
  blue accent (hue 258, chroma up to 0.13). The same accent in both themes: step 600 by day, step
  300 by night.
- **Night offsets** picked on sliders: surfaces −0.05 lightness from the proposal, text −0.01.
  Baked into the steps night uses: background 0.185, card 0.225, muted 0.27; text 0.89, muted text
  0.81, faint text 0.71.
- **Positive** (live dots) at hue 155, kept apart from the accent.
- **Focus rings** are neutral (neutral-950 by day, neutral-100 by night), and text selection is a
  tint of `primary-soft`.
- Every text role passes WCAG AA in both themes. Night body text is APCA Lc 84; it was 96.

Values live in `src/app/globals.css`, layer 1 (palettes) and layer 2 (roles).

## Rejected

- **Current** (warm paper, oxblood by day, brass by night): the red read wrong, the night was too
  dark (L 0.17) and too contrasty (Lc 96), and switching accent hue between themes weakened the
  brand.
- **Moss** (warm cream, forest green): literary, but green is a stock "calm" colour and it pushed
  the live signal to amber.
- **Ochre** (warm tonal, burnt amber): cosy, but the accent sat too close to the surfaces and links
  stood out less.
- **Ultramarine** (untinted neutrals, vivid blue): crisp, but cold, and close to the browser's
  default link blue.
