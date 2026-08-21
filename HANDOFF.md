# Handoff: Quiet personal index

## Current state

The design/layout pass is complete. `bun run build`, `bun run tokens:check`, `bun run lint`, and `git diff --check` pass. The checked viewport matrix has no horizontal overflow. About now uses a normal reading flow, the 979px homepage has no empty identity track, and article pages use one contents list without section marks or a progress rail.

## Canonical references

- `PRODUCT.md` — audience, positioning, and scope.
- `CONTEXT.md` — project vocabulary.
- `DESIGN.md` — shipped quiet personal-index design system.
- `.impeccable/homepage-surface-brief.md` — homepage direction.
- `.impeccable/surfaces/` — non-home surface constraints.
- The homepage is intentionally text-first; no artwork bundle is required for the shipped direction.

## Homepage invariants

- Header: Mathieu, Work, Writing, Bookshelf, About.
- Opening field: identity, role, two orientation paragraphs, and a small Now row.
- The first page exposes separate Work, Projects, Writing, and Bookshelf sections with View all routes.
- Rows remain ordinary links and usable without JavaScript; mobile stacks the Now row and section content.
- Work and Writing are never merged. The homepage does not need artwork to communicate.

## Reading invariants

- Shared masthead and measured reading sheet.
- `66ch` reading measure on wide screens, with a readable mobile band.
- Optional, bounded margin details with one contents list.
- Prose and heading links remain usable without the apparatus or JavaScript.
- No homepage artwork on reading/index surfaces.

## Safe checks

```bash
bun run build
bun run tokens:check
bun run lint
git diff --check
```

When changing the homepage, inspect at least `1440×900`, `979×900`, `760×900`, `390×844`, and `320×568`. Check overflow, navigation, section stacking, focus, reduced motion, and the no-script reading path. Do not invent authored copy, role details, contact destinations, or project claims.
