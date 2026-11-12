# Typography: Instrument × Serif

Decided 12 November 2026, from a live type lab over a real essay, in both themes.

## Chosen

- **Instrument Sans** for headings and interface, at 600 for every heading: `type-headline`
  46px (36px on phones) for the page title, `type-title` 27px for sections, `type-subtitle` 20px
  below them. Tracking −0.025em on the headline, easing off as headings get smaller.
- **Source Serif 4** for reading text: 17.5px with a line height of 1.6, and oldstyle
  proportional figures, so numbers sit inside the sentence.
- **Fragment Mono** for meta, figures and code. It ships only a regular (400) weight, so every
  mono role is set at 400.
- **Labels** (eyebrows): Instrument Sans 11px, uppercase, +0.1em tracking.
- **Reading column** is 44.5rem. That gives about 90 characters per line, past the usual 60–75
  range. Mathieu chose the wider column knowing that.

## Which role a piece of text takes

Decided 12 November 2026, in a consistency pass across every page.

- **Facts are mono** (`type-meta`): dates, bylines, read times, roles, periods, counts. The same
  fact looks the same in a list and on its own page.
- **Uppercase labels are for section names only** (`type-eyebrow`): "In short", "More work",
  "Writing" on a home tile. Never a date or a role.
- **A row title in an index is `type-subtitle`** (20px): Writing and Work rows alike. Long titles
  take two lines on desktop. The books being read on Reading stay larger, as featured items.
- **Home's title stays `type-title`**: it sits beside the monogram in a 20rem sidebar, where the
  page headline does not fit.
- **Typographer's quotes everywhere**: essay bodies get remark-smartypants; frontmatter titles,
  descriptions and roles go through `smartQuotes` (`src/lib/typeset.ts`); UI strings are typed
  with `’`. Only the characters change, never the words.
- **Tight tiles respond to their own width**, not the window's (container queries): the Reading
  tile puts the title under the covers, Elsewhere goes two by two with smaller icons, and a tile
  header truncates its trailing note instead of wrapping.

## Where it lives

- `src/lib/type-families.ts` names the three families once.
- `src/app/layout.tsx` loads them with next/font. `src/app/globals.css` maps them onto
  `--font-sans`, `--font-serif` and `--font-mono` in `@theme`, and sets every size, weight,
  tracking and figure style in the `type-*` roles. `type-code` sizes code in em, so it keeps pace
  with the text around it.
- The `essay` utility maps rendered markdown onto the roles element by element. The typography
  plugin's `prose prose-lg` only spaces the blocks.
- OG cards (`src/lib/og.tsx`) draw static Instrument Sans TTFs from `src/assets/fonts/`, because
  Satori reads no CSS or woff2. They restate the roles by hand.
- `src/lib/design-tokens.test.ts` fails on any type set outside a role: arbitrary values, figure,
  case and style utilities, inline font styles (except the OG card), and type in `prose-*`
  modifiers. It also fails when the layout, CSS, OG fonts or `/design` specimen drift from the
  family list and the roles.

## Rejected

- **Newsreader with Geist and Geist Mono** (the previous system): three faces with overlapping
  jobs. The serif carried titles and text while a sans carried the "In short" block, so an essay
  read as two designs. It also looked like a default setup, and the column ran about 80
  characters.
- **Instrument Serif headlines** (Editorial): condensed and striking, but not wanted for
  headings. It ships only one weight.
- **One serif for everything** (Book, Literata): coherent, but too quiet, and the navigation
  became serif.
- **IBM Plex Sans, Serif and Mono** (Engineer, Plex family): one coherent family, but common on
  technical blogs. Kept as the runner-up.
- **Bricolage Grotesque headlines** (Masthead): its Source Serif 4 reading text was liked and
  kept. The heavy, quirky display face was not.
- **Small caps for labels**: none of the Google-served fonts tried ship small caps in their
  Latin subset, and `font-synthesis: none` (rightly) forbids faked ones.
