---
name: Mathieu Saugier
description: A quiet personal index — warm white, near-black type, restrained blue links, and generous space.
colors:
  cobalt-structure: "#3567d8"
  orange-event: "#b86f45"
  cool-paper: "#fbfbfa"
  near-black-ink: "#171717"
  muted-ink: "#737373"
  cobalt-rule: "#dededc"
  cobalt-wash: "#f1f4fb"
  cobalt-underline: "color-mix(in srgb, #3567d8 45%, transparent)"
typography:
  headline:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "clamp(2.8rem, 7vw, 6.4rem)"
    fontWeight: 620
    lineHeight: 0.95
    letterSpacing: "-0.065em"
  headline-article:
    fontFamily: "Literata Variable, Georgia, serif"
    fontSize: "clamp(2.7rem, 6vw, 5.2rem)"
    fontWeight: 560
    lineHeight: 0.98
    letterSpacing: "-0.04em"
  title:
    fontFamily: "Literata Variable, Georgia, serif"
    fontSize: "clamp(1.8rem, 3.5vw, 2.7rem)"
    fontWeight: 540
    lineHeight: 1.05
    letterSpacing: "-0.04em"
  title-work:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "1.08rem"
    fontWeight: 560
    lineHeight: 1.25
  prose-h2:
    fontFamily: "Literata Variable, Georgia, serif"
    fontSize: "clamp(1.7rem, 3vw, 2.25rem)"
    fontWeight: 560
    lineHeight: 1.14
    letterSpacing: "-0.025em"
  title-ledger:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "1.08rem"
    fontWeight: 560
    lineHeight: 1.25
  prose-h3:
    fontFamily: "Literata Variable, Georgia, serif"
    fontSize: "clamp(1.25rem, 2vw, 1.5rem)"
    fontWeight: 560
    lineHeight: 1.14
    letterSpacing: "-0.025em"
  identity:
    fontFamily: "Literata Variable, Georgia, serif"
    fontSize: "1.15rem"
    fontWeight: 560
    letterSpacing: "-0.025em"
  title-onward:
    fontFamily: "Literata Variable, Georgia, serif"
    fontSize: "clamp(1.1rem, 1.8vw, 1.4rem)"
    fontWeight: 550
    lineHeight: 1.14
    letterSpacing: "-0.02em"
  prose-h4:
    fontFamily: "Literata Variable, Georgia, serif"
    fontSize: "1.05rem"
    fontWeight: 560
    lineHeight: 1.14
    letterSpacing: "-0.025em"
  role-line:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "clamp(1.35rem, 2.7vw, 2.15rem)"
    fontWeight: 480
    lineHeight: 1.15
    letterSpacing: "-0.04em"
  body-about:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "clamp(1.05rem, 1.5vw, 1.2rem)"
    lineHeight: 1.68
  standfirst:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "clamp(1rem, 1.4vw, 1.15rem)"
    lineHeight: 1.55
  body-article:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "clamp(1.02rem, 0.686rem + 0.372vw, 1.15rem)"
    lineHeight: 1.7
  body:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "1rem"
    lineHeight: 1.55
  table:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "0.86rem"
  intro:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "clamp(1.05rem, 1.8vw, 1.35rem)"
    lineHeight: 1.55
  meta:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "0.82rem"
    lineHeight: 1.35
  apparatus:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "0.78rem"
    lineHeight: 1.35
  action-link:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "0.78rem"
    fontWeight: 620
  data:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "0.72rem"
    fontWeight: 620
    letterSpacing: "0.04em"
  label:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "0.7rem"
    fontWeight: 620
    letterSpacing: "0.04em"
  caption-label:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "0.68rem"
    fontWeight: 620
    letterSpacing: "0.04em"
  micro:
    fontFamily: "Commissioner Variable, Arial, sans-serif"
    fontSize: "0.64rem"
    fontWeight: 620
    letterSpacing: "0.04em"
  code:
    fontFamily: "Roboto Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.82rem"
    lineHeight: 1.6
  code-inline:
    fontFamily: "Roboto Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.86em"
spacing:
  page-gutter: "clamp(1.25rem, 5vw, 5rem)"
  page-gutter-narrow: "1rem"
  section-block: "clamp(5rem, 10vw, 8rem)"
  surface-head-block: "clamp(4rem, 8vw, 7rem)"
  sheet-head-block: "clamp(4.5rem, 10vw, 8rem)"
  reading-block: "clamp(2.5rem, 5vw, 4rem)"
  column-gap: "clamp(1.5rem, 4vw, 4rem)"
  ledger-row: "clamp(1.2rem, 2.5vw, 1.8rem)"
---

# Design System: Mathieu Saugier

## Direction

This is a personal website, not a publication template or a SaaS product. The
visual system is a quiet index: warm white space, near-black type, restrained
blue links, thin neutral rules, and a small number of carefully chosen serif
moments. The home page makes the person and the things he is doing visible in
one scrollable surface; archive pages provide depth without changing worlds.

## Principles

- Let content and sequence create interest. Do not decorate empty space with
  gradients, cards, badges, dashboards, or artificial activity signals.
- Use Commissioner for direct, contemporary reading and Literata for section
  headings, article titles, and a few editorial anchors.
- Keep rows open and tactile: a title, a useful description, and a clear
  destination. Blue is for interaction, not for every piece of metadata.
- Prefer whitespace and hierarchy to borders. Rules are quiet separators, not a
  repeated newspaper grid.
- Motion should clarify navigation: small row translations, opacity changes,
  and page transitions only. Everything must remain useful with reduced motion
  and without JavaScript.

## Content constraints

Mathieu's own prose is rendered from Markdown and must not be rewritten as part
of a visual change. Work, writing, bookshelf, and about remain real routes;
the home page is a curated index into them, not a replacement for them.
