# Handoff: Living Index

## Current state

The design/layout pass is complete. `bun run build`, `bun run tokens:check`, `bun run lint`, and `git diff --check` pass. The checked viewport matrix has no horizontal overflow. About now uses a normal reading flow, the 979px homepage has no empty identity track, and article pages use one contents list without section marks or a progress rail.

## Canonical references

- `PRODUCT.md` — audience, positioning, and scope.
- `CONTEXT.md` — project vocabulary.
- `DESIGN.md` — shipped design system and Atomic Design rules.
- `.impeccable/homepage-surface-brief.md` — homepage direction.
- `.impeccable/surfaces/` — non-home surface constraints.
- `src/assets/home/PLACEHOLDERS.md` — replaceable artwork.

## Homepage invariants

- Header: Mathieu, Work, Writing, Bookshelf, About.
- Opening field: orientation, separate Work/Projects and Writing indexes, then one art stage.
- At `980px` and below, orientation becomes a full-width band; the two index channels remain paired.
- At `760px` and below, the page stacks and the art stage becomes sticky. Essential links remain ordinary links and usable without JavaScript.
- Work and Writing are never merged. Bookshelf is navigation-only on the homepage.

## Reading invariants

- Shared masthead and ruled sheet.
- `52ch` reading measure.
- Optional, bounded margin apparatus with one contents list.
- Prose and heading links remain usable without the apparatus or JavaScript.
- No homepage artwork on reading/index surfaces.

## Safe checks

```bash
bun run build
bun run tokens:check
bun run lint
git diff --check
```

When changing the homepage, inspect at least `1440×900`, `979×900`, `760×900`, `390×844`, and `320×568`. Check overflow, navigation, stage stacking, focus, reduced motion, and the no-script reading path. Do not invent authored copy, role details, contact destinations, project claims, or final artwork.
