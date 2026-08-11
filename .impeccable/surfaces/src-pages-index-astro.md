---
version: 1
slug: "src-pages-index-astro"
primary_target: "src/pages/index.astro"
related_targets: []
---

# Homepage

## Scope and mode

- Target: `src/pages/index.astro`
- Mode: Experience, with immediate professional orientation.
- Purpose: establish who Mathieu is, demonstrate how he thinks and works, and create reasons to explore or begin a conversation.

## Audience, job, and proof

- Primary visitor: a technically sophisticated founder, executive, CTO, peer, or potential collaborator.
- Visitor job: understand Mathieu's professional center quickly, then discover credible evidence through projects and authored writing.
- Proof: selected Work, dated Writing, authored artwork, and direct access to deeper pages.
- Desired continuation: explore a Work or Writing entry, or contact Mathieu through X or Telegram.

## Approved direction

- Visual world: Living Index.
- Approved compositional base: `.impeccable/mocks/living-index-b-horizontal-sequence.webp`.
- Interaction reference: `.impeccable/mocks/living-index-c-art-stage.webp`.
- Composition: recognizable personal-site header and orientation, followed by separately labelled Work and Writing indexes, a panoramic narrative artwork stage, and enough second-fold content to establish density and depth.
- Memorable moment: the panoramic artwork changes as a Work or Writing entry becomes active, borrowing Composition C's art-stage behavior without turning imagery into hidden navigation.

## Information architecture

- Work and Writing are distinct segments with distinct labels, ordering, and destinations.
- Work contains professional projects and Field Reports.
- Writing contains articles, technical guides, project reflections, and Book Reflections.
- The Living Index must never interleave Work and Writing in a single undifferentiated list.
- Bookshelf remains a dedicated destination and may remain in global navigation, but it is not a concrete homepage-intro section.
- About remains a concise destination rather than a résumé timeline.

## Constraints

- All published prose is written or explicitly approved by Mathieu; unknown copy remains a placeholder during implementation.
- Essential identity, navigation, Work, and Writing content is usable without animation, WebGL, hover, or client-side hydration.
- Artwork supports and responds to the index but does not carry essential labels or navigation.
- Motion must respect reduced-motion preferences and preserve a stable still composition.
- The light resting state, quiet continuous energy, cobalt structure, and stronger orange narrative events carry forward.

## Unresolved

- Final homepage introduction and professional descriptor.
- Which real artwork maps to each Work and Writing entry.
- Exact initial Work and Writing selections.
- Mobile translation of the panoramic art stage and index interaction.

## Implementation fidelity inventory

| Ingredient | Commitment | Medium |
| --- | --- | --- |
| Global header | Mathieu's name at left; Work, Writing, Bookshelf, and About at right; one fine rule; no app chrome | Semantic HTML and CSS |
| Display type | Sculptural literary serif with an unmistakable silhouette; maximum display size below 6rem | Self-hosted Literata variable font |
| Reading and navigation type | Clear, compact humanist sans with strong small-size rendering | Self-hosted Commissioner variable font |
| Introductory copy | Mathieu's authored descriptor and short introduction; explicitly marked placeholders until supplied | Semantic HTML |
| Work index | Separate labelled group for Kakarot, Starknet Agent, and Cairo Coder; title, placeholder summary, real destination | Semantic HTML and CSS |
| Writing index | Separate labelled group populated from the real Astro content collection | Semantic HTML and CSS |
| Panoramic art stage | One dominant image covering roughly one-third of the first desktop viewport; changes with active Work or Writing entry | Optimized generated placeholder rasters through `astro:assets` |
| Active-stage interaction | Hover and keyboard focus on desktop; intersection-driven selection on narrow screens; links remain normal links | Small vanilla TypeScript enhancement |
| Cobalt structure | Fine horizontal rules, active markers, and focus geometry; no decorative grid | CSS borders and authored geometry |
| Paper material | Cool-white uncoated fiber visible only at close inspection | Generated seamless raster texture |
| Work detail section | Three unequal editorial columns with project titles and explicit copy placeholders | Semantic HTML and CSS |
| Writing detail section | Dated article rows with real titles and descriptions from the repository | Semantic HTML and CSS |
| Contact close | Quiet invitation with unresolved X and Telegram destinations clearly marked for replacement | Semantic HTML |

### Compositional commitments

- Desktop header occupies a thin top band; the orientation and two indexes form the first information field.
- Work and Writing are visually and semantically separate before and after the panoramic stage.
- The panoramic image spans the content width and remains the only dominant image in the opening sequence.
- The lower page alternates the denser Work treatment with the quieter Writing rows; it does not become a uniform card grid.
- On mobile, identity leads, the stage follows, then Work and Writing stack separately. The stage may become sticky only while its related index is in view.
- Generated imagery, descriptions, contact destinations, and professional descriptor remain visibly replaceable; no placeholder is presented as fact.
