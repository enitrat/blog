# Handoff: Living Index visual QA

## Mission for the next session

Perform computer-vision-assisted QA of the implemented personal homepage, concentrating on responsive composition, visual ordering, sticky-stage behavior, image loading, overflow, accessibility-adjacent layout states, and fidelity to the approved Living Index direction. Fix only confirmed material defects; preserve the content and authorship boundaries.

## Canonical project references

Do not reconstruct the product or design rationale from this handoff. Read the existing artifacts:

- `PRODUCT.md` — audience, positioning, content boundaries, and product goals.
- `CONTEXT.md` — working context and prior decisions.
- `DESIGN.md` — shipped visual system and responsive rules.
- `.impeccable/surfaces/src-pages-index-astro.md` — approved surface brief.
- `.impeccable/mocks/living-index-b-horizontal-sequence.webp` — approved composition.
- `.impeccable/mocks/living-index-c-art-stage.webp` — approved interaction character.
- `docs/design/explorations/README.md` — preserved Living Index and Editorial Standard directions.
- `docs/adr/0001-keep-astro-as-the-experience-shell.md` — stack decision.
- `src/assets/home/PLACEHOLDERS.md` — generated asset provenance and replacement status.

The implemented surface is primarily:

- `src/pages/index.astro`
- `src/components/home/LivingIndex.astro`
- `src/components/SiteHeader.astro` and `src/components/SiteFooter.astro`
- `src/styles/site.css` and `src/styles/home.css`
- `src/components/BaseHead.astro`
- `src/consts.ts`

## Current status

- Astro production build passes with `bun run build`.
- Impeccable's mechanical design detector returned `[]`.
- The independent Impeccable finish reviewer initially failed the mobile stacking behavior, then returned `PASS` after the fix and after `DESIGN.md` was created.
- Desktop hover/focus changes `data-active-id` and the active frame's `aria-hidden` state.
- Mobile stage/trigger geometry and hit testing were probed after the stacking fix.
- Direction seed `526476f5` is present in built `dist/index.html`.
- All visible prose not owned by the user is explicitly marked as placeholder copy. Current-role and contact details remain unresolved intentionally.

Run locally with:

```bash
bun install
bun run dev
```

Then inspect `http://localhost:4321`. For the production path, use `bun run build && bun run preview`.

## Important implementation complexities

### The same content has visual and DOM order differences on mobile

`LivingIndex.astro` keeps the semantic DOM order as orientation → Work/Writing groups → art stage. At `760px` and below, CSS flex ordering changes the visual sequence to orientation → art stage → Work/Writing groups. This is intentional because artwork is supplementary; essential links remain earlier in source order and work without JavaScript. A QA agent must distinguish an accessibility-safe DOM difference from an accidental visual-order bug.

### The mobile stage is a sticky viewport, not a static image

The mobile stage is `position: sticky`, `z-index: 3`, and `height: min(42svh, 24rem)` with an `18rem` floor. The index groups use `z-index: 2`. This was changed because the opaque groups previously covered the stage precisely while its image changed.

The correct live state should show:

- the artwork pinned across the top portion of the viewport;
- one or more index rows still visible and tappable below it;
- the active artwork changing as rows pass through the observer band;
- no index row hidden behind the artwork when it is expected to be selected or tapped.

The `IntersectionObserver` maintains ratios for all triggers and uses a band below most of the sticky stage (`rootMargin: -48% 0px -18% 0px`). Equal ratios can legitimately leave the earlier adjacent row active until the next row becomes the stronger candidate. Test natural slow scrolling, not only `scrollIntoView()`.

### Desktop and mobile use different activation mechanisms

- Desktop: pointer hover and keyboard focus activate matching artwork.
- Mobile: visibility while scrolling activates matching artwork.
- Links remain normal links; the artwork is never required navigation.
- Exactly one figure should have `data-active="true"` and `aria-hidden="false"`.

### Several images occupy the same stage coordinates

All stage frames are absolutely stacked. The initial frame loads eagerly; alternatives are lazy. A blank flash during the first hover/scroll activation is a risk on slow networks. Test with cache disabled and a throttled connection.

The detailed Work images are also lazy. Playwright full-page screenshots sometimes captured blank reserved image boxes even though a later runtime probe showed `complete: true` and non-zero `naturalWidth`. Prefer ordinary viewport screenshots after actually scrolling each section into view and waiting for decoding. Do not diagnose missing production media from one full-page capture alone.

### Existing screenshots are useful but not definitive

`.impeccable/screenshots/` has been deleted: those captures predated the final mobile z-index/observer correction, and ~16MB of stale, regenerable PNG is not worth carrying. The directory is gitignored, so a fresh QA run can write there freely. Full-page capture also does not faithfully demonstrate sticky positioning. Generate fresh viewport screenshots for the QA pass.

An apparent small dark control strip appeared over the panoramic image in screenshot viewing, while direct inspection of `stage-builder.webp` showed no such pixels. Confirm in a normal browser before treating it as baked-in artwork or a UI defect.

### The design relies on layered global and homepage CSS

`BaseHead.astro` imports the existing global stylesheet, while `index.astro` imports `home.css`. Most homepage behavior is anchored under `.home-page`, but the page-specific stylesheet also contains conventional class selectors and root tokens. Check for inherited global typography, link, header, image, and mobile rules overriding the intended homepage styling.

### Image generation is intentionally provisional

The paper texture and four narrative images are generated placeholders. Prompts, sidecars, originals, and replacement guidance live under `src/assets/home/`. QA should assess composition, crop, contrast, and responsiveness—not whether these are the user's final art choices.

Astro currently generates many responsive image variants during a cold build, so the first build can be noticeably slower. Cached builds are much faster.

## Required visual ordering

### Wide desktop

1. Thin identity/header row: Mathieu identity left; Work, Writing, Bookshelf, About right.
2. First editorial field:
   - orientation/name on the left;
   - Work index in the middle;
   - Writing index on the right.
3. One panoramic art stage spanning the full content width beneath that field.
4. Detailed Work section using unequal editorial columns.
5. Detailed Writing section using ruled ledger rows.
6. Closing statement/contact placeholder section.
7. Minimal footer.

Work and Writing must never be merged into a mixed feed. Bookshelf must remain navigation-only on the homepage.

### Intermediate width (`761px–980px`)

1. Orientation remains beside the index area.
2. Work and Writing index groups stack within their side of the opening field.
3. The art stage remains below the opening field.
4. Detailed Work becomes a two-column layout, with its lead entry spanning both columns.

Pay special attention around both breakpoints; this intermediate state has not had the same depth of visual inspection as the endpoints.

### Mobile (`≤760px`)

1. Identity followed by a complete four-item navigation row.
2. Name, role, descriptor placeholder, and introduction placeholder.
3. Sticky art stage.
4. Separate Work index.
5. Separate Writing index.
6. Single-column detailed Work entries.
7. Ruled detailed Writing list.
8. Closing/contact placeholders.
9. Footer.

The sticky stage should remain visible while index rows scroll below it. All four header destinations must remain present; do not hide Bookshelf to make the row fit.

## Computer-vision QA matrix

Capture ordinary viewport screenshots—not only full-page images—at minimum:

- 1440×900 and 1440×1100
- 1024×768
- 981×900 and 979×900
- 768×1024
- 761×900 and 760×900
- 390×844
- 375×667
- 320×568

For each size, check:

- no horizontal overflow or clipped navigation;
- the name does not collide with Work/Writing indexes;
- rules join cleanly without doubled or missing borders;
- long English and French article titles wrap without arrow collisions;
- the panoramic crop retains a meaningful focal figure;
- artwork captions remain legible and inside the image;
- section headings do not orphan from their content;
- detailed Work entries align intentionally rather than appearing as a generic card grid;
- Writing dates, titles, descriptions, and arrows preserve a clear ledger rhythm;
- closing copy and contact placeholders do not overlap at narrow widths;
- paper texture remains subtle and does not reduce text contrast;
- there is no unexplained empty media box after scrolling and waiting for image decode;
- focus outlines are visible and not clipped;
- at 200% browser zoom, navigation and major content remain usable.

For interaction capture:

1. Desktop initial state.
2. Desktop after hovering every Work item and at least two Writing items.
3. Desktop keyboard-only traversal through the same entries.
4. Mobile with the first Work row below the sticky stage.
5. Mobile midway through Writing rows, showing the active artwork and an accessible row together.
6. Mobile at the point the sticky stage releases at the end of the Living Index parent.
7. Reduced-motion mode at desktop and mobile widths.
8. JavaScript disabled: the first artwork and every link must still form a coherent page.
9. Slow-network/cache-disabled activation: no unacceptable blank stage.

Where possible, pair visual screenshots with DOM probes for bounding boxes, `data-active-id`, active-frame count, `aria-hidden`, image `complete`, and `naturalWidth`.

## Main risks

- Sticky-stage stacking can regress easily if either z-index or group backgrounds change.
- A sticky stage that is too tall can cover active/tappable rows on short phones.
- Observer selection may feel one row behind during fast or programmatic scrolling; judge natural scroll behavior.
- Hidden lazy-loaded stage frames may briefly reveal the dark fallback on slow connections.
- The 761/760 and 981/979 transitions may expose ordering or spacing discontinuities.
- Real authored prose may be longer than placeholders and could break the carefully tuned first viewport.
- Final artwork may have different focal points and require per-image `object-position` or crop metadata.
- Generated image output count and cold-build time may grow as more artwork is added.
- Existing global CSS can create subtle regressions when reused components or selectors change.
- The current visible contact controls are deliberately dashed non-links; they must become real X/Telegram links when destinations are supplied.

## Content and scope boundaries

- Do not invent final prose, project outcomes, current-role details, availability language, or contact destinations.
- Do not foreground ZK. The intended positioning is broad software engineering and product translation, with DeFi/blockchain leadership and applied ZK as supporting evidence.
- Do not make the site appear job-seeking.
- Do not add a career-history browser or recruitment funnel.
- Do not add Bookshelf content to the homepage opening sequence.
- Preserve unrelated pre-existing/untracked files such as the POC pages and assets unless the user explicitly brings them into scope.

## Tooling notes

- `bun run build` is the reliable implementation check.
- `git diff --check` passed at handoff time.
- Biome can report Astro frontmatter imports/variables as unused even when Astro templates use them. Do not apply unsafe auto-fixes blindly.
- Reduced-motion CSS uses `!important` to defeat transition timing, which Biome warns about. Treat it as intentional unless replacing it with an equally reliable scoped mechanism.
- Impeccable's detector has already been run once for this implementation and returned clean; do not rerun it merely to reproduce the result unless the QA session materially changes the interface.

## Suggested skills

- `impeccable` — primary skill for visual critique, bounded iteration, and finish quality.
- `prototype` — useful if a sticky-stage alternative needs to be tested without committing to the production structure.
- `review-animations` — use if changing the crossfade, image scale, hover, or reduced-motion behavior.
- `diagnosing-bugs` — use for reproducible breakpoint, image-loading, observer, or stacking defects.
- `impl-validator` — request a fresh second opinion after any material QA fixes.

## Recommended next action

Start the dev server, produce fresh viewport screenshots at 1440×900, 979×900, 760×900, 390×844, and 320×568, and inspect the sticky stage during real mobile scrolling. Record evidence before editing. If the composition passes, test long replacement copy and varied artwork crops as adversarial fixtures without publishing invented content.
